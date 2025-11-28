# Urban Litter Detection from POV Walking Routes Using YOLOv8

This repository contains the code and minimal assets used to reproduce the litter-detection pipeline presented in our study on automated monitoring of urban public spaces in Santiago, Chile. The method combines point-of-view (POV) images captured during pedestrian walking routes with a YOLOv8 detector pretrained on the TACO dataset. The pipeline integrates object detection, EXIF-based geolocation, tabular outputs, and an optional evaluation module.

Due to privacy and ethical considerations, the full dataset of images used in the study cannot be released. Instead, we provide a **small illustrative sample**, the **exact scripts used for inference and data preparation**, and the **labeling schema** used by the research team.

---

## Repository contents

```
LITTER-DETECTION/
├── data/
│   └── images/
│       ├── IMG_1jpg.jpg
│       ├── IMG_2.jpg
│       ├── IMG_3.jpg
│       ├── IMG_4.jpg
│       ├── IMG_5.jpg
│       ├── IMG_6.jpg
│       ├── IMG_7.jpg
│       └── IMG_8.jpg
├── outputs/
│   ├── results.csv
│   └── results.geojson
├── weights/
│   └── litter_yolov8.pt
├── run_litter_pipeline.py
└── README.md

```

---

## Features

- **YOLOv8 inference** on POV urban images  
- **EXIF GPS extraction** for geolocation  
- **CSV output** with predictions and metadata  
- **GeoJSON export** for spatial visualization and analysis  
- **Optional evaluation** (precision, recall, TP/TN/FP/FN)  
- **Minimal sample dataset** (safe for publication)  
- **Labeling schema** describing human annotation criteria  

---

## Background

This code operationalizes the methodological components described in our paper, where litter in public spaces is identified through POV images captured during structured walking routes. The detector uses YOLOv8 weights trained on the **TACO dataset (Trash Annotations in Context)**.

The goal of this repository is to ensure reproducibility of:

1. The inference procedure  
2. The geolocation workflow  
3. The evaluation metrics  
4. The preparation of GeoJSON files used in the spatial analysis  

Rather than providing the full dataset, we provide a precise reproduction of the **processing pipeline** as used in the study.

---

## Installation

### 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install ultralytics exifread pandas
```

---

## Running the pipeline

### Basic usage

```bash
python src/run_litter_pipeline.py     --image_dir ./src/sample_data/sample_images     --weights ./weights/litter_yolov8.pt     --output_dir ./outputs     --conf 0.25
```

### With ground truth (optional)

```bash
python src/run_litter_pipeline.py     --image_dir ./images     --weights ./weights/litter_yolov8.pt     --output_dir ./outputs     --gt_csv ./ground_truth.csv     --conf 0.25
```

### Arguments

| Argument | Description |
|---------|-------------|
| `--image_dir` | Folder containing input images |
| `--weights` | Path to YOLOv8 weights (TACO-based) |
| `--output_dir` | Folder for output CSV and GeoJSON |
| `--conf` | Confidence threshold (default = 0.25) |
| `--gt_csv` | Optional ground truth file |
| `--class_ids` | Optional list of class IDs considered “litter” |

---

## Outputs

### `results.csv`
Contains per-image metadata:

| image_name | has_litter_pred | lat | lon |
|------------|-----------------|-----|-----|
| IMG_001.jpg | 1 | -33.44 | -70.65 |
| IMG_002.jpg | 0 |         |        |

### `results.geojson`
A FeatureCollection of geolocated detections suitable for:

- GIS software (QGIS, ArcGIS)
- Mapping libraries (Leaflet, Mapbox)

### `metrics.json` (if GT provided)

```json
{
  "precision": 0.89,
  "recall": 0.57,
  "accuracy": 0.72,
  "tp": 1234,
  "tn": 4567,
  "fp": 123,
  "fn": 987
}
```

---

## Labeling schema

The annotation guidelines used in the study are provided in:

```
src/labeling/schema.md
```

This includes:

- Definitions of litter vs non-litter  
- Rules for ambiguous cases  
- Contextual criteria based on the object's presence/function  
- Examples used internally by annotators  

---

## Model weights

This project uses YOLOv8 weights trained on the **TACO dataset**.

Original source:  
https://github.com/jeremy-rico/litter-detection  

Please cite the TACO dataset if reusing the weights.

---

## Acknowledgements

- TACO dataset authors  
- Volunteers who collected POV imagery  
