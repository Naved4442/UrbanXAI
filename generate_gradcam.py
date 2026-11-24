import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import Model
from train_model import build_model
from train_model import load_split

MODEL_PATH = "model_results/experiment_1/models/best_model.keras"
OUTPUT_DIR = Path("explainability/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.475
NUM_IMAGES = 20

# GPU configuration
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

print("Building EfficientNetB3 architecture...")
model = build_model(encoder_weights=None)
model.load_weights("model_results/experiment_1/model.weights.h5")
print("Experiment 1 weights loaded successfully.")

grad_model = Model(
    inputs=model.inputs,
    outputs=[model.get_layer("top_activation").output, model.output]
)

print("Loading validation images...")
X_val, Y_val = load_split("val")

def generate_heatmap(image):
    image_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        features, predictions = grad_model(image_tensor, training=False)

        predicted_mask = tf.cast(
            predictions > THRESHOLD, tf.float32
        )

        count = tf.reduce_sum(predicted_mask)
        score = tf.cond(
            count > 0,
            lambda: tf.reduce_sum(predictions * predicted_mask) / count,
            lambda: tf.reduce_max(predictions)
        )

    gradients = tape.gradient(score, features)

    if gradients is None:
        raise RuntimeError("Gradients could not be calculated.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(
        features[0] * weights[0][None, None, :],
        axis=-1
    )

    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)

    cam = tf.image.resize(
        cam[..., None], (128, 128)
    ).numpy().squeeze()

    return cam, predictions[0, ..., 0].numpy()

print("Generating Grad-CAM visualizations...")

for i in range(min(NUM_IMAGES, len(X_val))):
    image = X_val[i].astype(np.float32)

    heatmap, prediction = generate_heatmap(image)
    np.save(OUTPUT_DIR / f'gradcam_{i:03d}.npy', heatmap)

    image_display = np.clip(image / 255.0, 0, 1)
    binary_mask = (prediction > THRESHOLD).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image_display)
    axes[0].set_title("Original Image")

    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title("Predicted Buildings")

    axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM Heatmap")

    axes[3].imshow(image_display)
    axes[3].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[3].set_title("Grad-CAM Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"gradcam_{i:03d}.png", dpi=150)
    plt.close(fig)

    print(f"Completed image {i + 1}/{min(NUM_IMAGES, len(X_val))}")

print("\nGRAD-CAM GENERATION COMPLETED")
print("Saved to:", OUTPUT_DIR)
