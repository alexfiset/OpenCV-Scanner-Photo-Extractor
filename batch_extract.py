import cv2
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BLANK_PATH  = "Blank_Scanner_Bed.jpg"
PHOTOS_DIR  = "Photos"
OUTPUT_DIR  = "output_photos"

# ── Tunable parameters ─────────────────────────────────────────────────────
DIFF_THRESH       = 5    # LAB weighted diff threshold; raise for noisy scanner, lower for dark photos
CLOSE_K           = 10   # px; fills holes inside photos without bridging inter-photo gaps
OPEN_K            = 10   # px; removes thin noise strips at scanner edges
DIST_PEAK_THRESH  = 0.20  # fraction of max distance; raise if single photo splits, lower if touching photos merge
MIN_AREA_RATIO    = 0.03  # smallest allowed photo as fraction of image area
MAX_AREA_RATIO    = 0.80  # largest allowed photo as fraction of image area
ASPECT_MIN        = 0.20
ASPECT_MAX        = 5.00
DESKEW_ANGLE_MIN  = 3.0   # degrees; only warp-correct if rotation exceeds this
CROP_PAD          = 10    # px border added around each extracted photo
JPEG_QUALITY      = 95
IOU_MERGE_THRESH  = 0.20  # IoU above this → merge over-segmented regions


def iou(a, b):
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_to_rect(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def extract_crop(scan, label_mask, box, label_id):
    """Extract crop for one region, with optional deskew."""
    x1, y1, x2, y2 = box
    region = (label_mask == label_id).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        rect = cv2.minAreaRect(cnts[0])
        angle = rect[2]
        # cv2.minAreaRect returns angles in [-90, 0); normalize to [-45, 45)
        if angle < -45:
            angle += 90
        if abs(angle) > DESKEW_ANGLE_MIN:
            box_pts = cv2.boxPoints(rect).astype(np.float32)
            return warp_to_rect(scan, box_pts), angle

    # Simple rectangular crop with padding
    H, W = scan.shape[:2]
    x1c = max(0, x1 - CROP_PAD)
    y1c = max(0, y1 - CROP_PAD)
    x2c = min(W, x2 + CROP_PAD)
    y2c = min(H, y2 + CROP_PAD)
    return scan[y1c:y2c, x1c:x2c], 0.0


def save_scan_results(scan_path, scan, annotated, merged, crops):
    scan_name = os.path.splitext(os.path.basename(scan_path))[0]
    out_dir   = os.path.join(OUTPUT_DIR, scan_name)
    os.makedirs(out_dir, exist_ok=True)

    for i, crop in enumerate(crops):
        path = os.path.join(out_dir, f"photo_{i+1}.jpg")
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    cv2.imwrite(os.path.join(out_dir, "_annotated.jpg"), annotated,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    print(f"Saved {len(crops)} photos + annotated scan → {out_dir}")


def process_scan(scan_path, blank_orig, verbose=True):
    """Full pipeline for a single scan. Returns (annotated_img, crops, boxes)."""
    scan = cv2.imread(scan_path)
    if scan is None:
        print(f"  ERROR: cannot load {scan_path}")
        return None, [], []

    print(f"Working on scan : {scan_path}")

    sh, sw = scan.shape[:2]
    bh, bw = blank_orig.shape[:2]

    # Align
    if (bh, bw) != (sh, sw):
        blank = cv2.resize(blank_orig, (sw, sh), interpolation=cv2.INTER_LINEAR)
    else:
        blank = blank_orig.copy()

    # ── Pre-blur both to suppress JPEG block artifacts ─────────────────────────
    blank_blur = cv2.GaussianBlur(blank, (7, 7), 0)
    scan_blur  = cv2.GaussianBlur(scan,  (7, 7), 0)

    # ── Compute weighted LAB difference ───────────────────────────────────────
    blank_lab = cv2.cvtColor(blank_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    scan_lab  = cv2.cvtColor(scan_blur,  cv2.COLOR_BGR2LAB).astype(np.float32)

    dL = np.abs(scan_lab[:,:,0] - blank_lab[:,:,0])
    da = np.abs(scan_lab[:,:,1] - blank_lab[:,:,1])
    db = np.abs(scan_lab[:,:,2] - blank_lab[:,:,2])

    diff = (0.05 * dL + 0.5 * da + 0.5 * db).astype(np.uint8)

    # ── Threshold to binary mask ───────────────────────────────────────────────
    _, raw_mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    fg_pct = 100 * np.sum(raw_mask > 0) / raw_mask.size
    print(f"Raw mask foreground: {fg_pct:.1f}% of image")
    if fg_pct < 1:
        print("WARNING: Almost nothing detected — blank may be wrong file or DIFF_THRESH too high")
    if fg_pct > 80:
        print("WARNING: Nearly everything detected — blank may not match scan (different scanner / DPI)")

    def fill_mask(binary):
        """Fill interior of all external contours to produce solid regions."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    # ── Close: fill holes within photos ───────────────────────────────────────
    ck = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_K, CLOSE_K))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, ck)

    # ── Fill any remaining interior holes via contour drawing ──────────────────
    filled = fill_mask(closed)

    # ── Open: remove thin noise strips at scanner edges ───────────────────────
    ok = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_K, OPEN_K))
    clean_mask = cv2.morphologyEx(filled, cv2.MORPH_OPEN, ok)

    fg_pct = 100 * np.sum(clean_mask > 0) / clean_mask.size
    print(f"Clean mask foreground: {fg_pct:.1f}% of image")

    # ── Distance transform ─────────────────────────────────────────────────────
    dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Sure foreground: peaks = centers of each photo ────────────────────────
    _, sure_fg = cv2.threshold(dist_norm,
                               int(DIST_PEAK_THRESH * 255), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # ── Sure background: slightly dilated mask ────────────────────────────────
    sure_bg = cv2.dilate(clean_mask, np.ones((3, 3), np.uint8), iterations=3)

    # ── Unknown zone: boundary between photos ─────────────────────────────────
    unknown = cv2.subtract(sure_bg, sure_fg)

    # ── Label markers ─────────────────────────────────────────────────────────
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # background → 1, not 0
    markers[unknown == 255] = 0    # unknown → 0 (watershed will decide)

    # ── Run watershed (needs 3-channel image) ─────────────────────────────────
    markers_ws = cv2.watershed(scan.copy(), markers.copy())

    n_regions = len(np.unique(markers_ws)) - 2   # subtract background (1) and boundary (-1)
    print(f"Watershed found {n_regions} regions (including oversegmented candidates)")

    # Filter regions
    img_area = sh * sw
    candidates = []
    for label in np.unique(markers_ws):
        if label <= 1:
            continue
        region = (markers_ws == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        x, y, w, h = cv2.boundingRect(cnts[0])
        area_ratio = (w * h) / img_area
        aspect = max(w, h) / max(min(w, h), 1)
        if MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO and aspect <= ASPECT_MAX:
            candidates.append([x, y, x+w, y+h])

    # Merge overlapping
    candidates.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    merged_boxes = []
    used = [False] * len(candidates)
    for i, a in enumerate(candidates):
        if used[i]:
            continue
        group = [a]
        for j, b in enumerate(candidates):
            if i == j or used[j]:
                continue
            if iou(a, b) > IOU_MERGE_THRESH:
                group.append(b)
                used[j] = True
        merged_boxes.append([min(r[0] for r in group), min(r[1] for r in group),
                              max(r[2] for r in group), max(r[3] for r in group)])
        used[i] = True

    # Extract crops
    crops = []
    for box in merged_boxes:
        x1, y1, x2, y2 = box
        roi = markers_ws[y1:y2, x1:x2]
        valid = roi[roi > 1]
        if len(valid) == 0:
            crop = scan[max(0,y1-CROP_PAD):y2+CROP_PAD, max(0,x1-CROP_PAD):x2+CROP_PAD]
        else:
            dominant = np.bincount(valid.flatten()).argmax()
            crop, _ = extract_crop(scan, markers_ws, box, dominant)
        crops.append(crop)

    # Annotate
    ann = scan.copy()
    for i, (x1, y1, x2, y2) in enumerate(merged_boxes):
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 25)
        cv2.putText(ann, str(i+1), (x1+150, y1+400),
                    cv2.FONT_HERSHEY_SIMPLEX, 16, (50,50,10), 40)

    return ann, crops, merged_boxes


if __name__ == "__main__":
    blank_orig = cv2.imread(BLANK_PATH)
    if blank_orig is None:
        raise FileNotFoundError(f"Cannot load blank reference: {BLANK_PATH}")

    scan_files = sorted(f for f in os.listdir(PHOTOS_DIR)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    print(f"Processing {len(scan_files)} scans...\n")
    for fname in scan_files:
        scan_path = os.path.join(PHOTOS_DIR, fname)
        ann, crops, boxes = process_scan(scan_path, blank_orig)
        if ann is not None:
            save_scan_results(scan_path, cv2.imread(scan_path), ann, boxes, crops)
            print(f"  {fname}: {len(crops)} photos")
        print()
