import random
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is not installed. Install it first with: pip install opencv-python"
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is not installed. Install it first with: pip install numpy"
    ) from exc


CLASS_NAMES = [
    "stop",
    "speed breaker",
    "no entry",
    "hospital",
    "petrol pump",
    "overtaking p",
]

IMAGE_SIZE = (64, 64)
TARGET_IMAGES_PER_CLASS = 180
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "data"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_files(folder_path: Path):
    """Return supported image files in a class folder."""
    return sorted(
        [
            file_path
            for file_path in folder_path.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def next_augmented_index(class_dir: Path) -> int:
    """Find the next number to use for augmented filenames."""
    max_index = 0
    for file_path in class_dir.iterdir():
        if not file_path.is_file():
            continue
        stem = file_path.stem
        if not stem.startswith("aug_"):
            continue
        suffix = stem.replace("aug_", "", 1)
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def load_source_images(class_dir: Path):
    """Load the original class images and ignore previously augmented files."""
    source_images = []
    for file_path in list_image_files(class_dir):
        if file_path.stem.startswith("aug_"):
            continue

        image = cv2.imread(str(file_path))
        if image is None:
            print(f"  Warning: could not read {file_path.name}, skipping it.")
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        source_images.append(image)

    return source_images


def random_augmentation(image: np.ndarray) -> np.ndarray:
    """Apply lightweight random transforms using OpenCV and NumPy only."""
    height, width = image.shape[:2]

    angle = random.uniform(-18, 18)
    scale = random.uniform(0.9, 1.1)
    shift_x = random.uniform(-0.12, 0.12) * width
    shift_y = random.uniform(-0.12, 0.12) * height

    center = (width / 2, height / 2)
    transform = cv2.getRotationMatrix2D(center, angle, scale)
    transform[0, 2] += shift_x
    transform[1, 2] += shift_y

    augmented = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if random.random() < 0.35:
        blur_kernel = random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (blur_kernel, blur_kernel), 0)

    if random.random() < 0.35:
        noise = np.random.normal(0, 8, augmented.shape).astype(np.float32)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(
            np.uint8
        )

    alpha = random.uniform(0.85, 1.2)
    beta = random.randint(-18, 18)
    return cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)


def augment_class(class_dir: Path) -> None:
    """Generate new images until the class reaches the target count."""
    current_images = list_image_files(class_dir)
    current_count = len(current_images)
    images_needed = max(0, TARGET_IMAGES_PER_CLASS - current_count)

    print(f"\nClass: {class_dir.name}")
    print(f"  Current images: {current_count}")

    if images_needed == 0:
        print("  Target already reached. Skipping.")
        return

    source_images = load_source_images(class_dir)
    if not source_images:
        print("  No source images found. Skipping.")
        return

    print(f"  Generating {images_needed} augmented images...")
    file_index = next_augmented_index(class_dir)

    for _ in range(images_needed):
        source_image = random.choice(source_images)
        augmented_image = random_augmentation(source_image)
        output_path = class_dir / f"aug_{file_index:03d}.jpg"
        cv2.imwrite(str(output_path), augmented_image)
        file_index += 1

    final_count = len(list_image_files(class_dir))
    print(f"  Final images: {final_count}")


def validate_dataset(dataset_dir: Path) -> None:
    """Make sure the dataset folder and expected class folders exist."""
    if not dataset_dir.exists():
        raise SystemExit(
            f"Dataset folder not found: {dataset_dir}\n"
            "Create a 'data' folder beside this script and place class folders inside it."
        )

    missing_classes = [
        class_name for class_name in CLASS_NAMES if not (dataset_dir / class_name).is_dir()
    ]

    if missing_classes:
        raise SystemExit(
            "Missing class folders in data/: " + ", ".join(missing_classes)
        )


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    validate_dataset(DATASET_DIR)

    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Target images per class: {TARGET_IMAGES_PER_CLASS}")

    for class_name in CLASS_NAMES:
        augment_class(DATASET_DIR / class_name)

    print("\nDataset augmentation complete.")


if __name__ == "__main__":
    main()
