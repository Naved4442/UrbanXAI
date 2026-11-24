import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

from train_model import (
    load_split,
    binary_iou_np,
    dice_score_np,
    save_prediction_visuals,
    RESULTS_DIR,
    MODEL_DIR,
)

# Allow TensorFlow to allocate GPU memory as needed
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("Loading saved model...")
model = load_model(
    MODEL_DIR / "best_model.keras",
    compile=False,
    safe_mode=False
)

print("Loading test dataset...")
X_test, Y_test = load_split("test")

print("Running predictions with batch size 1...")
preds = model.predict(X_test, batch_size=1, verbose=1)

preds_bin = (preds > 0.5).astype(np.uint8)
Y_true = Y_test.astype(np.uint8)

iou = binary_iou_np(Y_true, preds_bin)
dice = dice_score_np(Y_true, preds_bin)

y_true_flat = Y_true.flatten()
y_pred_flat = preds_bin.flatten()

precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)

report = f"""
TEST SET EVALUATION RESULTS
===========================
IoU       : {iou:.4f}
Dice      : {dice:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1 Score  : {f1:.4f}
"""

print(report)

with open(RESULTS_DIR / "segmentation_metrics.txt", "w") as f:
    f.write(report)

save_prediction_visuals(X_test, Y_test, preds_bin)

print("EVALUATION COMPLETED SUCCESSFULLY!")
