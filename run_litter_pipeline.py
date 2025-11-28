import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import exifread
import pandas as pd
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("litter_pipeline")


def dms_to_decimal(dms, ref) -> Optional[float]:
    """
    Convert EXIF GPS coordinates from degrees/minutes/seconds (DMS) to decimal.
    `dms` is a list of exifread.utils.Ratio values.
    `ref` is one of 'N', 'S', 'E', 'W'.
    """
    try:
        deg = float(dms[0].num) / float(dms[0].den)
        mins = float(dms[1].num) / float(dms[1].den)
        secs = float(dms[2].num) / float(dms[2].den)
        decimal = deg + mins / 60.0 + secs / 3600.0
        if ref in ["S", "W"]:
            decimal = -decimal
        return decimal
    except Exception as e:
        logger.warning(f"Could not convert DMS to decimal: {e}")
        return None


def get_gps_from_image(image_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract latitude and longitude from the image EXIF.
    Returns (lat, lon) or (None, None) if GPS info is missing.
    """
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    gps_lat = tags.get("GPS GPSLatitude")
    gps_lat_ref = tags.get("GPS GPSLatitudeRef")
    gps_lon = tags.get("GPS GPSLongitude")
    gps_lon_ref = tags.get("GPS GPSLongitudeRef")

    if not all([gps_lat, gps_lat_ref, gps_lon, gps_lon_ref]):
        return None, None

    lat = dms_to_decimal(gps_lat.values, gps_lat_ref.values)
    lon = dms_to_decimal(gps_lon.values, gps_lon_ref.values)
    return lat, lon


def load_ground_truth(gt_path: Path) -> Dict[str, int]:
    """
    Load ground truth from a CSV file.
    Assumes at least the following columns:
        image_name, truth
    where `truth` is 0/1 and `image_name` matches the file name.
    """
    df = pd.read_csv(gt_path)
    if "image_name" not in df.columns or "truth" not in df.columns:
        raise ValueError("Ground truth CSV must contain 'image_name' and 'truth' columns.")
    df["image_name"] = df["image_name"].astype(str)
    df["truth"] = df["truth"].astype(int)
    return dict(zip(df["image_name"], df["truth"]))


def compute_metrics(truths: List[int], preds: List[int]) -> Dict[str, float]:
    """
    Compute precision, recall, accuracy and TP/TN/FP/FN.
    """
    if len(truths) == 0:
        return {}

    tp = sum(1 for t, p in zip(truths, preds) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(truths, preds) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(truths, preds) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truths, preds) if t == 1 and p == 0)

    total = len(truths)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_samples": total,
    }


def build_geojson(rows: List[Dict]) -> Dict:
    """
    Build a GeoJSON FeatureCollection from a list of rows containing 'lat' and 'lon'.
    """
    features = []
    for r in rows:
        lat = r.get("lat")
        lon = r.get("lon")
        if lat is None or lon is None:
            continue

        properties = {k: v for k, v in r.items() if k not in ["lat", "lon"]}
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": properties,
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def run_pipeline(
    image_dir: Path,
    weights_path: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
    gt_path: Optional[Path] = None,
    class_ids: Optional[List[int]] = None,
):
    """
    Run the full pipeline:
    - Run YOLOv8 on images under `image_dir`.
    - Set has_litter_pred = 1 if there is at least one detection of the given class(es).
    - Extract lat/lon from EXIF.
    - If ground truth is provided, compute metrics.
    - Export results.csv and results.geojson.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    logger.info(f"Model loaded from: {weights_path}")
    logger.info(f"Model classes: {model.names}")

    if gt_path is not None:
        gt_dict = load_ground_truth(gt_path)
        logger.info(f"Ground truth loaded for {len(gt_dict)} images")
    else:
        gt_dict = {}

    rows = []
    y_true = []
    y_pred = []

    image_paths = sorted(
        [p for p in image_dir.glob("**/*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    logger.info(f"Found {len(image_paths)} images to process in {image_dir}")

    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False,
        )

        litter_pred = 0
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.boxes.cls is not None:
                if class_ids is None:
                    litter_pred = 1 if len(r.boxes.cls) > 0 else 0
                else:
                    detected_classes = r.boxes.cls.int().tolist()
                    litter_pred = 1 if any(c in class_ids for c in detected_classes) else 0

        lat, lon = get_gps_from_image(img_path)

        img_name = img_path.name
        truth = gt_dict.get(img_name, None)
        if truth is not None:
            y_true.append(truth)
            y_pred.append(litter_pred)

        row = {
            "image_path": str(img_path),
            "image_name": img_name,
            "has_litter_pred": litter_pred,
            "lat": lat,
            "lon": lon,
        }
        if truth is not None:
            row["truth"] = truth
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    geojson_obj = build_geojson(rows)
    geojson_path = output_dir / "results.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f, ensure_ascii=False)
    logger.info(f"GeoJSON saved to {geojson_path}")

    if y_true:
        metrics = compute_metrics(y_true, y_pred)
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(
            "Precision: %.4f | Recall: %.4f | Accuracy: %.4f | TP: %d TN: %d FP: %d FN: %d (N=%d)",
            metrics["precision"],
            metrics["recall"],
            metrics["accuracy"],
            metrics["tp"],
            metrics["tn"],
            metrics["fp"],
            metrics["fn"],
            metrics["n_samples"],
        )
    else:
        logger.info("Metrics not computed because no ground truth was provided or matched.")


def main():
    parser = argparse.ArgumentParser(
        description="Litter detection pipeline with YOLOv8 + GeoJSON export."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Root folder with images (recursively searches for .jpg/.jpeg/.png).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLOv8 weights file trained on TACO.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder where results.csv, results.geojson and metrics.json will be saved.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLOv8 (default: 0.25).",
    )
    parser.add_argument(
        "--gt_csv",
        type=str,
        default=None,
        help="Optional path to CSV with ground truth (columns: image_name, truth).",
    )
    parser.add_argument(
        "--class_ids",
        type=int,
        nargs="*",
        default=None,
        help="Class IDs that are considered 'litter'. If omitted, any detection is counted.",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    gt_path = Path(args.gt_csv) if args.gt_csv else None

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if gt_path is not None and not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    run_pipeline(
        image_dir=image_dir,
        weights_path=weights_path,
        output_dir=output_dir,
        conf_threshold=args.conf,
        gt_path=gt_path,
        class_ids=args.class_ids,
    )


if __name__ == "__main__":
    main()
