# Grounded SAM 2 Inference

Automatically detect and segment objects in images using natural language prompts — no manual annotation needed.

This notebook combines two powerful AI models:
- **Grounding DINO** — finds objects in an image based on a text description (e.g. `"hands."`)
- **SAM 2 (Segment Anything Model 2)** — draws precise pixel masks around those objects

The result: colorful segmentation overlays saved as image files, ready to use.

---

## What it does

1. You provide a **folder of images** and a **text prompt** (like `"hands."` or `"person. dog."`)
2. Grounding DINO detects matching objects and draws bounding boxes
3. SAM 2 refines those into detailed segmentation masks
4. Results are saved as annotated images with colored overlays, labels, and confidence scores

---

## Example output

Each image gets bounding boxes + colored masks for every detected object, labeled with the object name and confidence score.

---

## How to run (Google Colab)

This notebook is designed to run on **Google Colab** with a free or Pro GPU.

### Step 1 — Open the notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook → GitHub**
3. Paste your GitHub repo URL and select `Grounded_SAM2_Inference.ipynb`

> Or use **File → Upload notebook** to upload it directly from your computer.

### Step 2 — Enable GPU

In Colab: **Runtime → Change runtime type → T4 GPU** → Save

### Step 3 — Run the setup cells

Run the first four cells in order. They will:
- Clone the Grounded SAM 2 repository
- Install PyTorch and dependencies
- Compile the Grounding DINO CUDA extension (~3–5 minutes)
- Download the model checkpoints (~2 GB)

### Step 4 — Mount Google Drive

Run the Drive mount cell and authorize access. Your images should be in a zip file on your Drive (e.g. `novel.zip`).

### Step 5 — Configure your run

In the config section, set:

```python
IMAGE_FOLDER = "/content/novel/novel"   # path to your images after unzipping
TEXT_PROMPT  = "hands."                 # what to detect — end each word/phrase with a period
OUTPUT_FOLDER = "/content/seg_results"  # where results are saved
```

**Prompt tips:**
- End each term with a period: `"cat."`, `"person. car."`
- Multiple objects: `"hands. face. phone."`
- Be specific for best results: `"coffee mug."` works better than `"object."`

### Step 6 — Run the inference cell

The final cell processes every image and saves results to `OUTPUT_FOLDER`. Progress is printed for each image.

---

## Configuration reference

| Variable | Description | Default |
|---|---|---|
| `SAM2_CHECKPOINT` | Path to SAM 2 model weights | `sam2.1_hiera_large.pt` |
| `GDINO_MODEL_ID` | Grounding DINO model from HuggingFace | `grounding-dino-tiny` |
| `BOX_THRESHOLD` | Minimum detection confidence (0–1) | `0.35` |
| `TEXT_THRESHOLD` | Minimum text-box alignment score (0–1) | `0.25` |
| `TEXT_PROMPT` | Object(s) to detect | `"hands."` |

Raise the thresholds if you're getting too many false detections. Lower them if objects are being missed.

---

## Requirements

- Google Colab (recommended) or a Linux machine with an NVIDIA GPU
- CUDA 12.1+
- Python 3.10+
- ~5 GB free disk space for model checkpoints

All Python dependencies are installed automatically by the notebook cells.

---

## Credits

- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) by IDEA-Research
- [SAM 2](https://github.com/facebookresearch/sam2) by Meta AI
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA-Research
