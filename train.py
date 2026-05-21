import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is not installed. Install it first with: pip install tensorflow"
    ) from exc


# Fixed folder order so training and detection use the same label mapping.
# These names must match the class folder names inside the dataset directory.
CLASS_NAMES = [
    "stop",
    "speed breaker",
    "no entry",
    "hospital",
    "petrol pump",
    "overtaking p",
]

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 20
VALIDATION_SPLIT = 0.2
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "data"
MODEL_PATH = Path(__file__).resolve().parent / "traffic_model.h5"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder_path: Path) -> int:
    """Count image files inside a folder and its subfolders."""
    return sum(
        1
        for file_path in folder_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_dataset(dataset_dir: Path) -> None:
    """Check that the dataset folder exists and each class contains images."""
    if not dataset_dir.exists():
        raise SystemExit(
            f"Dataset folder not found: {dataset_dir}\n"
            "Create a 'data' folder beside train.py and place class folders inside it."
        )

    missing_classes = []
    empty_classes = []

    print("Checking dataset...")
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            missing_classes.append(class_name)
            continue

        image_count = count_images(class_dir)
        print(f"  {class_name}: {image_count} images")

        if image_count == 0:
            empty_classes.append(class_name)

    if missing_classes:
        raise SystemExit(
            "Missing class folders in dataset: " + ", ".join(missing_classes)
        )

    if empty_classes:
        raise SystemExit(
            "These class folders do not contain any supported images: "
            + ", ".join(empty_classes)
        )


def build_model(num_classes: int) -> tf.keras.Model:
    """Create a lightweigh CNN suitable for CPU training and inference."""
    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_generators(dataset_dir: Path):
    """Create training and validation generators with the same class order."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
    )

    train_generator = train_datagen.flow_from_directory(
        directory=str(dataset_dir),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=str(dataset_dir),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    if train_generator.samples == 0:
        raise SystemExit("No training images were found. Check the dataset folder.")

    if validation_generator.samples == 0:
        raise SystemExit(
            "No validation images were found. Add more images or reduce VALIDATION_SPLIT."
        )

    return train_generator, validation_generator


def main() -> None:
    tf.keras.utils.set_random_seed(42)

    validate_dataset(DATASET_DIR)
    train_generator, validation_generator = create_generators(DATASET_DIR)

    print("\nClass mapping used by the model:")
    for label, index in train_generator.class_indices.items():
        print(f"  {index}: {label}")

    model = build_model(num_classes=len(CLASS_NAMES))
    print("\nModel summary:")
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    print("\nStarting training...")
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    model.save(MODEL_PATH)

    print("\nTraining complete.")
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
