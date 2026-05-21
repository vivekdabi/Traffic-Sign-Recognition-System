import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

try:
    from tensorflow.keras.models import load_model
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is not installed. Install it first with: pip install tensorflow"
    ) from exc


# The class order must match the folder order used in train.py.
CLASS_NAMES = [
    "stop",
    "speed breaker",
    "no entry",
    "hospital",
    "petrol pump",
    "overtaking p",
]

DISPLAY_NAMES = {
    "stop": "Stop",
    "speed breaker": "Speed Breaker",
    "no entry": "No Entry",
    "hospital": "Hospital",
    "petrol pump": "Petrol Pump",
    "overtaking p": "No Overtaking",
}

IMAGE_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.70
MODEL_PATH = Path(__file__).resolve().parent / "traffic_model.h5"


def format_label(label: str) -> str:
    """Convert folder-style names to readable text."""
    return DISPLAY_NAMES.get(label, label.replace("_", " ").title())


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Resize and normalize the webcam ROI for model prediction."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, IMAGE_SIZE)
    normalized_image = resized_image.astype("float32") / 255.0
    return np.expand_dims(normalized_image, axis=0)


def get_center_roi(frame: np.ndarray, roi_ratio: float = 0.55):
    """Use a center box so the user can place the traffic sign inside it."""
    height, width = frame.shape[:2]
    box_size = int(min(height, width) * roi_ratio)

    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def predict_sign(model, roi: np.ndarray):
    """Return the predicted class name and confidence score."""
    input_image = preprocess_image(roi)
    predictions = model.predict(input_image, verbose=0)[0]
    best_index = int(np.argmax(predictions))
    confidence = float(predictions[best_index])
    label = CLASS_NAMES[best_index]
    return label, confidence


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model file not found: {MODEL_PATH}\n"
            "Run train.py first to create traffic_model.h5."
        )

    model = load_model(MODEL_PATH)

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise SystemExit(
            "The saved model output does not match the number of class labels in detect.py."
        )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise SystemExit(
            "Could not open the webcam. Make sure a camera is connected and not being used by another app."
        )

    print("Webcam started. Place a traffic sign inside the center box.")
    print("Press 'q' to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read a frame from the webcam.")
                break

            roi, (x1, y1, x2, y2) = get_center_roi(frame)
            label, confidence = predict_sign(model, roi)

            if confidence >= CONFIDENCE_THRESHOLD:
                display_text = f"{format_label(label)}: {confidence * 100:.1f}%"
                box_color = (0, 255, 0)
            else:
                display_text = f"Uncertain: {format_label(label)} ({confidence * 100:.1f}%)"
                box_color = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                display_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Keep the sign inside the box | Press q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-Time Traffic Sign Recognition System  By Vivek Dabi IO76 ", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
