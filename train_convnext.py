#!/usr/bin/env python3

import time
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, UpSampling2D, Concatenate, Conv2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

from train_model import (
    load_split,
    attention_gate,
    conv_block,
    urban_attention_block,
    dice_score_np,
    binary_iou_np,
)


# --------------------------------------------------
# LOSS FUNCTIONS
# --------------------------------------------------

def combined_loss(y_true, y_pred):
    smooth = 1e-6
    axes = [1, 2, 3]

    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = (
        tf.reduce_sum(y_true, axis=axes)
        + tf.reduce_sum(y_pred, axis=axes)
    )

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    dice_loss = 1.0 - tf.reduce_mean(dice)

    tp = intersection
    fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=axes)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=axes)

    tversky = (tp + smooth) / (
        tp + 0.3 * fp + 0.7 * fn + smooth
    )

    focal_tversky = tf.reduce_mean(
        tf.pow(1.0 - tversky, 0.75)
    )

    return 0.5 * bce + dice_loss + 0.5 * focal_tversky


def dice_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    axes = [1, 2, 3]

    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = (
        tf.reduce_sum(y_true, axis=axes)
        + tf.reduce_sum(y_pred, axis=axes)
    )

    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return tf.reduce_mean(dice)


# --------------------------------------------------
# BALANCED DATA GENERATOR
# --------------------------------------------------

class BalancedSequence(tf.keras.utils.Sequence):

    def __init__(self, X, Y, batch_size=4):
        super().__init__()

        self.X = X
        self.Y = Y
        self.batch_size = batch_size

        coverage = np.mean(Y, axis=(1, 2, 3))

        self.groups = np.where(
            coverage == 0,
            0,
            np.where(coverage < 0.05, 1, 2)
        )

        counts = np.bincount(self.groups, minlength=3)
        group_probabilities = np.zeros(3, dtype=np.float64)

        present = counts > 0
        group_probabilities[present] = 1.0 / present.sum()

        self.sample_probabilities = np.zeros(len(X), dtype=np.float64)

        for group in range(3):
            if counts[group] > 0:
                self.sample_probabilities[self.groups == group] = (
                    group_probabilities[group] / counts[group]
                )

        print("Training image groups:", counts, flush=True)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.random.choice(
            len(self.X),
            size=len(self.X),
            replace=True,
            p=self.sample_probabilities
        )

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        batch_x = []
        batch_y = []

        for i in batch_indices:
            img = self.X[i].copy()
            mask = self.Y[i].copy()

            k = np.random.randint(0, 4)
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)

            if np.random.random() < 0.5:
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)

            if np.random.random() < 0.5:
                img = np.flip(img, axis=0)
                mask = np.flip(mask, axis=0)

            if np.random.random() < 0.5:
                contrast = np.random.uniform(0.9, 1.1)
                brightness = np.random.uniform(-10.0, 10.0)
                img = np.clip(img * contrast + brightness, 0.0, 255.0)

            batch_x.append(np.ascontiguousarray(img))
            batch_y.append(np.ascontiguousarray(mask))

        return (
            np.asarray(batch_x, dtype=np.float32),
            np.asarray(batch_y, dtype=np.float32)
        )


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "model_results" / "experiment_6_convnext"
WEIGHTS_DIR = RESULTS / "models"

PRETRAINED_WEIGHTS = (
    Path.home() / ".keras/models/convnext_tiny_notop.h5"
)

STAGE1_WEIGHTS = WEIGHTS_DIR / "stage1_best.weights.h5"
STAGE2_WEIGHTS = WEIGHTS_DIR / "stage2_best.weights.h5"
FINAL_WEIGHTS = WEIGHTS_DIR / "best_model.weights.h5"

BATCH_SIZE = 4
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 35


# --------------------------------------------------
# GPU CONFIGURATION
# --------------------------------------------------

gpus = tf.config.list_physical_devices("GPU")

if not gpus:
    raise RuntimeError("No GPU detected. Submit this script through PBS.")

tf.config.set_visible_devices(gpus[0], "GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

print("TensorFlow:", tf.__version__, flush=True)
print("Using GPU:", gpus[0], flush=True)

if not PRETRAINED_WEIGHTS.exists():
    raise FileNotFoundError(
        f"ConvNeXt weights not found: {PRETRAINED_WEIGHTS}"
    )

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def build_convnext_unet():

    inputs = Input(shape=(128, 128, 3), name="image")

    encoder = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights=None,
        input_tensor=inputs
    )

    encoder.load_weights(PRETRAINED_WEIGHTS)

    # ConvNeXt feature maps
    s1 = encoder.get_layer(
        "convnext_tiny_stem"
    ).output                                      # 32 x 32 x 96

    s2 = encoder.get_layer(
        "convnext_tiny_downsampling_block_0"
    ).output                                      # 16 x 16 x 192

    s3 = encoder.get_layer(
        "convnext_tiny_downsampling_block_1"
    ).output                                      # 8 x 8 x 384

    bridge = urban_attention_block(
        encoder.output
    )                                             # 4 x 4 x 768

    # Decoder: 4 -> 8
    d1 = UpSampling2D()(bridge)
    a1 = attention_gate(s3, d1, 256)
    d1 = Concatenate()([d1, a1])
    d1 = conv_block(d1, 256)

    # Decoder: 8 -> 16
    d2 = UpSampling2D()(d1)
    a2 = attention_gate(s2, d2, 128)
    d2 = Concatenate()([d2, a2])
    d2 = conv_block(d2, 128)

    # Decoder: 16 -> 32
    d3 = UpSampling2D()(d2)
    a3 = attention_gate(s1, d3, 64)
    d3 = Concatenate()([d3, a3])
    d3 = conv_block(d3, 64)

    # Decoder: 32 -> 64
    d4 = UpSampling2D()(d3)
    d4 = conv_block(d4, 32)

    # Decoder: 64 -> 128
    d5 = UpSampling2D()(d4)
    d5 = conv_block(d5, 16)

    outputs = Conv2D(
        1, 1, activation="sigmoid", name="building_mask"
    )(d5)

    return Model(inputs, outputs, name="ConvNeXtTiny_Attention_UNet"), encoder


# --------------------------------------------------
# CALLBACKS AND COMPILATION
# --------------------------------------------------

def make_callbacks(checkpoint_path, patience):

    return [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_dice_metric",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_dice_metric",
            mode="max",
            patience=patience,
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


def compile_model(model, learning_rate):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=combined_loss,
        metrics=[dice_metric],
        jit_compile=False
    )


# --------------------------------------------------
# EAGER VALIDATION
# --------------------------------------------------

def validation_scores(model, X_val, Y_val):

    print("Running eager validation predictions...", flush=True)

    predictions = []

    for i in range(len(X_val)):
        pred = model(X_val[i:i + 1], training=False)
        predictions.append(pred.numpy())

        if (i + 1) % 20 == 0:
            print(f"Predicted {i + 1}/{len(X_val)}", flush=True)

    predictions = np.concatenate(predictions, axis=0)
    binary = (predictions > 0.5).astype(np.uint8)

    return (
        dice_score_np(Y_val, binary),
        binary_iou_np(Y_val, binary)
    )


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def main():

    print("\nLoading training and validation data...", flush=True)

    X_train, Y_train = load_split("train")
    X_val, Y_val = load_split("val")

    Y_train = (Y_train > 0.5).astype(np.float32)
    Y_val = (Y_val > 0.5).astype(np.float32)

    print("Train:", X_train.shape, Y_train.shape, flush=True)
    print("Validation:", X_val.shape, Y_val.shape, flush=True)

    train_sequence = BalancedSequence(
        X_train, Y_train, BATCH_SIZE
    )

    print("\nBuilding ConvNeXt-Tiny Attention U-Net...", flush=True)

    model, encoder = build_convnext_unet()

    print("Total parameters:", f"{model.count_params():,}", flush=True)

    # ---------------- STAGE 1 ----------------

    print("\n========== STAGE 1: FROZEN ENCODER ==========", flush=True)

    encoder.trainable = False
    compile_model(model, 1e-3)

    start = time.time()

    history1 = model.fit(
        train_sequence,
        validation_data=(X_val, Y_val),
        validation_batch_size=BATCH_SIZE,
        epochs=STAGE1_EPOCHS,
        callbacks=make_callbacks(STAGE1_WEIGHTS, 6),
        verbose=1
    )

    print(
        "Stage 1 duration:",
        round(time.time() - start, 1), "seconds",
        flush=True
    )

    if not STAGE1_WEIGHTS.exists():
        raise RuntimeError("Stage 1 did not produce a checkpoint.")

    model.load_weights(STAGE1_WEIGHTS)

    # ---------------- STAGE 2 ----------------

    print("\n========== STAGE 2: FINE-TUNING ==========", flush=True)

    encoder.trainable = True

    # Fine-tune deeper ConvNeXt stages only.
    for layer in encoder.layers:
        layer.trainable = (
            layer.name.startswith((
                "convnext_tiny_downsampling_block_2",
                "convnext_tiny_stage_2_",
                "convnext_tiny_stage_3_",
                "layer_normalization"
            ))
        )

    trainable_layers = sum(
        1 for layer in encoder.layers if layer.trainable
    )

    print("Trainable encoder layers:", trainable_layers, flush=True)

    compile_model(model, 1e-5)

    start = time.time()

    history2 = model.fit(
        train_sequence,
        validation_data=(X_val, Y_val),
        validation_batch_size=BATCH_SIZE,
        epochs=STAGE2_EPOCHS,
        callbacks=make_callbacks(STAGE2_WEIGHTS, 8),
        verbose=1
    )

    print(
        "Stage 2 duration:",
        round(time.time() - start, 1), "seconds",
        flush=True
    )

    if not STAGE2_WEIGHTS.exists():
        raise RuntimeError("Stage 2 did not produce a checkpoint.")

    # ---------------- CHECKPOINT COMPARISON ----------------

    print("\nComparing checkpoints on validation data...", flush=True)

    model.load_weights(STAGE1_WEIGHTS)
    dice1, iou1 = validation_scores(model, X_val, Y_val)

    model.load_weights(STAGE2_WEIGHTS)
    dice2, iou2 = validation_scores(model, X_val, Y_val)

    print(f"Stage 1 Dice: {dice1:.4f} | IoU: {iou1:.4f}", flush=True)
    print(f"Stage 2 Dice: {dice2:.4f} | IoU: {iou2:.4f}", flush=True)

    if dice2 >= dice1:
        shutil.copy2(STAGE2_WEIGHTS, FINAL_WEIGHTS)
        selected = "Stage 2 fine-tuned"
        best_dice, best_iou = dice2, iou2
    else:
        shutil.copy2(STAGE1_WEIGHTS, FINAL_WEIGHTS)
        selected = "Stage 1 frozen encoder"
        best_dice, best_iou = dice1, iou1

    np.savez(
        RESULTS / "training_histories.npz",
        stage1_loss=history1.history["loss"],
        stage1_val_loss=history1.history["val_loss"],
        stage1_val_dice=history1.history["val_dice_metric"],
        stage2_loss=history2.history["loss"],
        stage2_val_loss=history2.history["val_loss"],
        stage2_val_dice=history2.history["val_dice_metric"]
    )

    report = (
        "EXPERIMENT 6 — CONVNEXT-TINY ATTENTION U-NET\n"
        f"Selected checkpoint: {selected}\n"
        f"Stage 1 validation Dice: {dice1:.4f}\n"
        f"Stage 1 validation IoU: {iou1:.4f}\n"
        f"Stage 2 validation Dice: {dice2:.4f}\n"
        f"Stage 2 validation IoU: {iou2:.4f}\n"
        f"Best validation Dice: {best_dice:.4f}\n"
        f"Best validation IoU: {best_iou:.4f}\n"
        f"Selected weights: {FINAL_WEIGHTS}\n"
    )

    print("\n" + report, flush=True)

    with open(RESULTS / "validation_summary.txt", "w") as f:
        f.write(report)

    print("EXPERIMENT 6 TRAINING COMPLETED.", flush=True)
    print("Test set has NOT been used.", flush=True)


if __name__ == "__main__":
    main()
