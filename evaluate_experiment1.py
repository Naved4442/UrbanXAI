import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

from train_model import (
    build_model,
    load_split,
    binary_iou_np,
    dice_score_np,
    RESULTS_DIR,
    MODEL_DIR,
)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("Building Experiment 1 model...", flush=True)

model = build_model(encoder_weights="imagenet")

print("Loading trained weights...", flush=True)

model.load_weights(RESULTS_DIR / "model.weights.h5")

print("Loading test data...", flush=True)

X_test, Y_test = load_split("test")

print("Running predictions...", flush=True)

preds = model.predict(X_test, batch_size=1, verbose=1)

preds_bin = (preds > 0.5).astype(np.uint8).squeeze()
Y_test = (Y_test > 0.5).astype(np.uint8).squeeze()

iou = binary_iou_np(Y_test, preds_bin)
dice = dice_score_np(Y_test, preds_bin)
precision = precision_score(Y_test.ravel(), preds_bin.ravel(), zero_division=0)
recall = recall_score(Y_test.ravel(), preds_bin.ravel(), zero_division=0)
f1 = f1_score(Y_test.ravel(), preds_bin.ravel(), zero_division=0)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_DIR / "segmentation_metrics.txt", "w") as f:
    f.write("EXPERIMENT 1 — ImageNet EfficientNetB3\n")
    f.write(f"IoU:       {iou:.4f}\n")
    f.write(f"Dice:      {dice:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")

print("\n===== EXPERIMENT 1 RESULTS =====")
print(f"IoU:       {iou:.4f}")
print(f"Dice:      {dice:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nMetrics saved successfully.")
