import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_model import build_model, load_split

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
RESULTS = Path("model_results/experiment_3")
MODEL_DIR = RESULTS / "models"
WEIGHTS = "model_results/experiment_1/model.weights.h5"

EPOCHS = 40
BATCH_SIZE = 4
LEARNING_RATE = 1e-6

MODEL_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(42)

# --------------------------------------------------
# FOCAL TVERSKY + BCE LOSS
# --------------------------------------------------
def focal_tversky_bce_loss(y_true, y_pred):
    smooth = 1.0
    alpha = 0.3
    beta = 0.7
    gamma = 0.75

    axes = [1, 2, 3]

    tp = tf.reduce_sum(y_true * y_pred, axis=axes)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=axes)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=axes)

    tversky = (tp + smooth) / (
        tp + alpha * fp + beta * fn + smooth
    )

    focal_loss = tf.reduce_mean(
        tf.pow(1.0 - tversky, gamma)
    )

    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )

    return bce + focal_loss


# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------
def augment(image, mask):
    # Rotate image and mask by the same random angle
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    # Synchronized horizontal flip
    flip_lr = tf.random.uniform([]) > 0.5
    image = tf.cond(
        flip_lr,
        lambda: tf.image.flip_left_right(image),
        lambda: image,
    )
    mask = tf.cond(
        flip_lr,
        lambda: tf.image.flip_left_right(mask),
        lambda: mask,
    )

    # Synchronized vertical flip
    flip_ud = tf.random.uniform([]) > 0.5
    image = tf.cond(
        flip_ud,
        lambda: tf.image.flip_up_down(image),
        lambda: image,
    )
    mask = tf.cond(
        flip_ud,
        lambda: tf.image.flip_up_down(mask),
        lambda: mask,
    )

    # Mild image-only brightness and contrast changes
    image = tf.image.random_brightness(image, max_delta=8.0)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image, mask


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("Loading training and validation data...", flush=True)

X_train, Y_train = load_split("train")
X_val, Y_val = load_split("val")

print("Training data:", X_train.shape, Y_train.shape, flush=True)
print("Validation data:", X_val.shape, Y_val.shape, flush=True)

# --------------------------------------------------
# GPU CONFIGURATION
# --------------------------------------------------
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("Available GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

# --------------------------------------------------
# TF DATA PIPELINES
# --------------------------------------------------
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    .shuffle(len(X_train), seed=42, reshuffle_each_iteration=True)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# --------------------------------------------------
# BUILD MODEL AND LOAD EXPERIMENT 1 WEIGHTS
# --------------------------------------------------
print("Building EfficientNetB3 Attention U-Net...", flush=True)

model = build_model(encoder_weights="imagenet")

print("Loading Experiment 1 trained weights...", flush=True)
model.load_weights(WEIGHTS)

print("Compiling Experiment 3 model...", flush=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=focal_tversky_bce_loss,
    metrics=["accuracy"],
)

# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_model.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1,
    ),
]

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
print("\n===== EXPERIMENT 3 TRAINING STARTED =====", flush=True)

start = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print(
    f"Training completed in {time.time() - start:.1f} seconds",
    flush=True,
)

# Save the best restored weights separately
model.save_weights(
    str(MODEL_DIR / "final_best.weights.h5")
)

# Save training history
np.savez(
    RESULTS / "training_history.npz",
    **history.history
)

print("Experiment 3 checkpoints and history saved.", flush=True)
print("===== EXPERIMENT 3 COMPLETED =====", flush=True)
