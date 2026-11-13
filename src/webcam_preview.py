"""Live webcam preview with MediaPipe Face Landmarker visualization."""

import argparse
import time
import urllib.request
import ssl
import certifi
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# Face Landmarker indices around the eyes and mouth for a clear, uncluttered overlay.
FEATURE_LANDMARKS = (
    33, 133, 160, 158, 144, 153, 362, 263, 385, 387, 373, 380,
    78, 308, 13, 14,
)
EYE_CONTOURS = ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))
MOUTH_OUTLINE = (78, 81, 13, 311, 308, 402, 14, 178)


def ensure_model() -> Path:
    """Download Google's task model once and keep it in the project models folder."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the MediaPipe Face Landmarker model from Google...")
    try:
        with urllib.request.urlopen(MODEL_URL, context=ssl.create_default_context(cafile=certifi.where()), timeout=60) as response:
            MODEL_PATH.write_bytes(response.read())
    except Exception as exc:
        MODEL_PATH.unlink(missing_ok=True)
        raise SystemExit(
            "Could not download the face landmark model. Check your internet "
            f"connection and try again. Details: {exc}"
        ) from exc
    return MODEL_PATH


def to_pixel_points(landmarks, width: int, height: int) -> np.ndarray:
    return np.array(
        [(round(point.x * width), round(point.y * height)) for point in landmarks],
        dtype=np.int32,
    )


def eye_aspect_ratio(points: np.ndarray, indices: tuple[int, ...]) -> float:
    eye = points[list(indices)].astype(float)
    vertical = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    return float(vertical / (2.0 * horizontal + 1e-6))


def mouth_aspect_ratio(points: np.ndarray) -> float:
    vertical = np.linalg.norm(points[13].astype(float) - points[14].astype(float))
    horizontal = np.linalg.norm(points[78].astype(float) - points[308].astype(float))
    return float(vertical / (horizontal + 1e-6))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show MediaPipe face landmarks from a live webcam."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Camera index to open (default: 1, the built-in MacBook camera on your setup).",
    )
    args = parser.parse_args()

    model_path = ensure_model()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Check camera permissions or try "
            "another index, for example: python src/webcam_preview.py --camera 0"
        )

    print("MediaPipe preview started. Press q or Escape in the window to quit.")
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the camera.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
                if result.face_landmarks:
                    points = to_pixel_points(result.face_landmarks[0], width, height)
                    left_ear = eye_aspect_ratio(points, (33, 160, 158, 133, 153, 144))
                    right_ear = eye_aspect_ratio(points, (362, 385, 387, 263, 373, 380))
                    ear = (left_ear + right_ear) / 2.0
                    mar = mouth_aspect_ratio(points)
                    cv2.putText(
                frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
                    for contour in EYE_CONTOURS:
                        cv2.polylines(
                            frame, [points[list(contour)]], True, (0, 255, 0), 2
                        )
                    cv2.polylines(
                        frame,
                        [points[list(MOUTH_OUTLINE)]],
                        True,
                        (255, 180, 0),
                        2,
                    )
                    for index in FEATURE_LANDMARKS:
                        cv2.circle(frame, tuple(points[index]), 2, (0, 200, 255), -1)
                    label = "Face detected"
                    color = (0, 220, 0)
                else:
                    label = "No face detected"
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    label,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    "Green: eyes   Orange: mouth   q/Esc: quit",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("DMS - MediaPipe Face Landmarker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
