# OpenCV-Scanner-Photo-Extractor

A Python tool that automatically extracts individual photos from flatbed scanner images using OpenCV background subtraction, morphological filtering, and watershed segmentation.

**Primary use case:** Digitize a batch of old prints by placing multiple photos on a flatbed scanner at once, then let this tool split them into separate, deskewed image files.

Created with the use of Claude Code.
---

## Sample Output

The annotated scan shows each detected photo labeled and bounded. Individual crops are saved as separate JPEGs.

![Annotated scan output](output_photos/IMG_01/annotated.jpg)

---

## How It Works

1. **Background subtraction** — compares your scan against a reference blank-bed image in LAB color space to isolate photo regions
2. **Morphological cleanup** — closing fills holes inside photos; opening removes scanner-noise strips
3. **Watershed segmentation** — distance transform finds photo centers; watershed separates touching photos
4. **Deskewing** — perspective warp corrects rotation if a photo was placed at an angle
5. **Cropped output** — each detected photo is saved as a numbered JPEG

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `opencv-python`, `numpy`
Notebook only: `matplotlib`, `pillow`, `jupyter`

---

## Usage

### Batch processing

1. Place your blank-bed reference scan as `Blank_Scanner_Bed.jpg` in the project root
2. Put your scan images in the `Photos/` directory
3. Run:

```bash
python batch_extract.py
```

Extracted photos and an annotated preview are saved to `output_photos/<scan_name>/`.

### Interactive notebook

Open `Photo_Identifier_NB.ipynb` in Jupyter to step through the pipeline visually and tune parameters.

---

## Scanning Tips

- Space photos **3–5 mm apart** on the bed for reliable separation
- Use **identical scanner settings** for the blank reference and all scans (resolution, color mode, compression)
- JPEG artifacts are suppressed by a Gaussian blur pre-pass, but higher scan quality always helps

---

## Tunable Parameters

Edit the constants at the top of `batch_extract.py` (or the second notebook cell):

| Parameter | Default | Effect |
|---|---|---|
| `DIFF_THRESH` | `5` | Raise if your scanner introduces noise or color cast |
| `DIST_PEAK_THRESH` | `0.20` | Raise if one photo splits; lower if two photos merge |
| `IOU_MERGE_THRESH` | `0.20` | Controls how aggressively over-segmented regions are merged |
| `MIN_AREA_RATIO` | `0.03` | Ignore regions smaller than this fraction of the image |
| `DESKEW_ANGLE_MIN` | `3.0°` | Only deskew photos rotated more than this angle |
| `CROP_PAD` | `10 px` | Border padding added around each extracted photo |
| `JPEG_QUALITY` | `95` | Output JPEG compression quality |

---

## Output Structure

```
output_photos/
└── IMG_01/
    ├── photo_1.jpg
    ├── photo_2.jpg
    ├── photo_3.jpg
    └── _annotated.jpg   ← original scan with detection boxes
```
