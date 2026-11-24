import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from train_model import build_model, load_split, bce_dice_loss

OUT = "model_results/experiment_4"
os.makedirs(OUT + "/models", exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-6
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Loading dataset...", flush=True)
X_train, Y_train = load_split("train")
X_val, Y_val = load_split("val")

X_train = np.asarray(X_train, dtype=np.float32)
X_val = np.asarray(X_val, dtype=np.float32)
Y_train = np.asarray(Y_train, dtype=np.float32)
Y_val = np.asarray(Y_val, dtype=np.float32)

if Y_train.ndim == 3:
    Y_train = Y_train[..., None]
if Y_val.ndim == 3:
    Y_val = Y_val[..., None]

# Divide training images by building coverage.
coverage = np.mean(Y_train > 0.5, axis=(1, 2, 3))
empty_idx = np.where(coverage == 0)[0]
sparse_idx = np.where((coverage > 0) & (coverage < 0.05))[0]
dense_idx = np.where(coverage >= 0.05)[0]

print(
    f"Sampling groups: empty={len(empty_idx)}, "
    f"sparse={len(sparse_idx)}, dense={len(dense_idx)}",
    flush=True
)

def make_group(indices):
    return tf.data.Dataset.from_tensor_slices(
        (X_train[indices], Y_train[indices])
    ).shuffle(max(len(indices), 1), reshuffle_each_iteration=True).repeat()

def augment(x, y):
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    y = tf.image.rot90(y, k)

    flip_lr = tf.random.uniform([]) > 0.5
    x = tf.cond(flip_lr, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(flip_lr, lambda: tf.image.flip_left_right(y), lambda: y)

    flip_ud = tf.random.uniform([]) > 0.5
    x = tf.cond(flip_ud, lambda: tf.image.flip_up_down(x), lambda: x)
    y = tf.cond(flip_ud, lambda: tf.image.flip_up_down(y), lambda: y)

    x = tf.image.random_brightness(x, max_delta=8.0)
    x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
    x = tf.clip_by_value(x, 0.0, 255.0)
    return x, y

groups = [
    make_group(empty_idx),
    make_group(sparse_idx),
    make_group(dense_idx),
]

# Each training epoch draws roughly 25% empty, 35% sparse, 40% dense images.
train_ds = tf.data.Dataset.sample_from_datasets(
    groups, weights=[0.25, 0.35, 0.40], seed=SEED
)
train_ds = train_ds.map(
    augment, num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

steps = math.ceil(len(X_train) / BATCH_SIZE)

print("Building model and loading Experiment 1 weights...", flush=True)
model = build_model()
model.load_weights("model_results/experiment_1/model.weights.h5")

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss=bce_dice_loss
)

callbacks = [
    ModelCheckpoint(
        OUT + "/models/best_model.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("Starting Experiment 4...", flush=True)
history = model.fit(
    train_ds,
    steps_per_epoch=steps,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

model.save_weights(OUT + "/models/final_best.weights.h5")
np.savez(OUT + "/training_history.npz", **history.history)

print("Experiment 4 training complete.", flush=True)
