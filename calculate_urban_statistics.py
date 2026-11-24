import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import numpy as np
import tensorflow as tf

from pathlib import Path
from scipy import ndimage
from train_model import build_model, load_split

MODEL_PATH = "model_results/experiment_1/model.weights.h5"
OUTPUT_DIR = Path("urban_statistics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.475
MIN_BUILDING_PIXELS = 3

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

print("Loading best EfficientNetB3 model...")
model = build_model(encoder_weights="imagenet")
model.load_weights(MODEL_PATH)

X_val, Y_val = load_split("val")

print("Generating GPU predictions...")
predictions = model.predict(X_val, batch_size=1, verbose=1)

structure = np.ones((3, 3), dtype=np.uint8)
records = []

for i in range(len(X_val)):
    prediction = predictions[i, ..., 0]

    pred_mask = prediction >= THRESHOLD
    true_mask = Y_val[i, ..., 0] >= 0.5

    labels, count = ndimage.label(pred_mask, structure=structure)
    sizes = np.bincount(labels.ravel())
    predicted_buildings = int(np.sum(sizes[1:] >= MIN_BUILDING_PIXELS))

    true_labels, true_count = ndimage.label(
        true_mask, structure=structure
    )
    true_sizes = np.bincount(true_labels.ravel())
    actual_buildings = int(
        np.sum(true_sizes[1:] >= MIN_BUILDING_PIXELS)
    )

    total_pixels = pred_mask.size
    built_pixels = int(pred_mask.sum())

    records.append({
        "image": i,
        "predicted_building_count": predicted_buildings,
        "ground_truth_component_count": actual_buildings,
        "built_up_pixels": built_pixels,
        "built_up_percentage": 100 * built_pixels / total_pixels,
        "ground_truth_built_up_percentage":
            100 * true_mask.sum() / total_pixels,
        "building_density_per_10000_pixels":
            predicted_buildings * 10000 / total_pixels
    })

    print(f"Processed {i+1}/{len(X_val)}")

csv_path = OUTPUT_DIR / "validation_urban_statistics.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

total_pixels = len(X_val) * 128 * 128
total_built_pixels = sum(r["built_up_pixels"] for r in records)
total_buildings = sum(r["predicted_building_count"] for r in records)
total_gt_buildings = sum(r["ground_truth_component_count"] for r in records)

mean_built_up = 100 * total_built_pixels / total_pixels
mean_gt_built_up = np.mean([
    r["ground_truth_built_up_percentage"] for r in records
])

report = f"""
URBAN STATISTICS — EFFICIENTNETB3 ATTENTION U-NET
================================================

Dataset: SpaceNet 2 Paris validation split
Images analysed: {len(records)}
Image dimensions: 128 x 128
Prediction threshold: {THRESHOLD}
Minimum component size: {MIN_BUILDING_PIXELS} pixels

PREDICTED STATISTICS
--------------------
Total detected connected components: {total_buildings}
Mean components per image: {total_buildings / len(records):.2f}
Overall built-up percentage: {mean_built_up:.4f}%
Mean built-up percentage per image: {np.mean([r['built_up_percentage'] for r in records]):.4f}%
Mean building density per 10,000 pixels:
{np.mean([r['building_density_per_10000_pixels'] for r in records]):.4f}

GROUND-TRUTH REFERENCE
---------------------
Total ground-truth connected components: {total_gt_buildings}
Mean ground-truth built-up percentage: {mean_gt_built_up:.4f}%

IMPORTANT LIMITATIONS
---------------------
Connected components are approximate building counts.
Touching buildings may merge into one component.
Small buildings below the minimum pixel threshold are excluded.
Density is pixel-based, not buildings per square kilometre.
No geospatial area conversion has been applied.
"""

(OUTPUT_DIR / "urban_statistics_report.txt").write_text(report)

print(report)
print("CSV saved:", csv_path)
print("Report saved:", OUTPUT_DIR / "urban_statistics_report.txt")
