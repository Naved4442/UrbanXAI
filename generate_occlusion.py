import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from train_model import build_model, load_split

MODEL_PATH = "model_results/experiment_1/model.weights.h5"
OUTPUT_DIR = Path("explainability/occlusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 16
THRESHOLD = 0.475
NUM_IMAGES = 20

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

print("Loading EfficientNetB3 model...")
model = build_model(encoder_weights=None)
model.load_weights(MODEL_PATH)

X_val, Y_val = load_split("val")

def predict_score(image):
    pred = model.predict(image[None, ...], verbose=0)[0, ..., 0]
    mask = pred > THRESHOLD
    if not np.any(mask):
        return 0.0
    return float(np.mean(pred[mask]))

print("Generating patch occlusion explanations...")

for idx in range(min(NUM_IMAGES, len(X_val))):
    image = X_val[idx].astype(np.float32)
    baseline = predict_score(image)

    importance = np.zeros((128, 128), dtype=np.float32)

    for y in range(0, 128, PATCH_SIZE):
        for x in range(0, 128, PATCH_SIZE):
            occluded = image.copy()
            occluded[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :] = 0

            score = predict_score(occluded)
            importance[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = baseline - score

    importance = np.maximum(importance, 0)
    if importance.max() > 0:
        importance /= importance.max()

    np.save(OUTPUT_DIR / f'occlusion_{idx:03d}.npy', importance)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(np.clip(image / 255.0, 0, 1))
    axes[0].set_title("Original Image")

    axes[1].imshow(importance, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Occlusion Importance")

    axes[2].imshow(np.clip(image / 255.0, 0, 1))
    axes[2].imshow(importance, cmap="jet", alpha=0.5)
    axes[2].set_title("Occlusion Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"occlusion_{idx:03d}.png", dpi=150)
    plt.close(fig)

    print(f"Completed image {idx+1}/{min(NUM_IMAGES, len(X_val))}")

print("OCCLUSION EXPLANATIONS COMPLETED")
print("Saved to:", OUTPUT_DIR)
