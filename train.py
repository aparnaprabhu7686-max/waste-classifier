"""Train the waste classifier (MobileNetV2 transfer learning).

Usage:  python train.py
Expects images in dataset/<glass|metal|organic|paper|plastic>/
"""
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


def load_data():
    common = dict(
        directory=str(config.DATASET_DIR),
        validation_split=0.2,
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_names=config.CLASSES,
        label_mode="categorical",
    )
    train = keras.utils.image_dataset_from_directory(subset="training", **common)
    val = keras.utils.image_dataset_from_directory(subset="validation", **common)
    auto = tf.data.AUTOTUNE
    return train.prefetch(auto), val.prefetch(auto)


def build_model():
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ])
    base = keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=config.IMG_SIZE + (3,))
    x = augment(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(config.CLASSES), activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def main():
    train_ds, val_ds = load_data()
    model, base = build_model()

    cbs = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ]

    # Phase 1: train the classifier head
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_HEAD, callbacks=cbs)

    # Phase 2: fine-tune the top of the base network
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_FINE, callbacks=cbs)

    loss, acc = model.evaluate(val_ds)
    print(f"\nValidation accuracy: {acc:.2%}")

    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(config.MODEL_PATH)
    config.LABELS_PATH.write_text(json.dumps(config.CLASSES))
    print(f"Saved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
