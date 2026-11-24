import time
import tensorflow as tf

from train_model import (
    build_model,
    load_split,
    bce_dice_loss,
)

RESULTS = "model_results/experiment_2"
INITIAL_WEIGHTS = "model_results/experiment_1/model.weights.h5"

EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-5

print("Loading training and validation data...", flush=True)

X_train, Y_train = load_split("train")
X_val, Y_val = load_split("val")

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print("Building model...", flush=True)

model = build_model(encoder_weights="imagenet")

print("Loading Experiment 1 trained weights...", flush=True)
model.load_weights(INITIAL_WEIGHTS)

print("Compiling with lower learning rate...", flush=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=bce_dice_loss,
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{RESULTS}/models/best_model.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

print("Starting Experiment 2 fine-tuning...", flush=True)

start = time.time()

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

print(f"Training completed in {time.time()-start:.1f} seconds")

model.save_weights(
    f"{RESULTS}/models/final_best.weights.h5"
)

print("Experiment 2 checkpoint saved successfully.")
