
"""
APATE — face deepfake inference module.

Scope: face-manipulation classification only.
This does not detect general image edits or verify news articles.
"""

from typing import Any
from io import BytesIO

import cv2
import numpy as np
import timm
import torch
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download


VALID_PREDICTIONS = {
    "potentially_manipulated",
    "no_manipulation_detected",
    "inconclusive",
}

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MODEL_REPO = "mahedi420/deepfake-vit-detector"
MODEL_FILE = "model.pt"

# Treat low-confidence outputs as inconclusive.
# This is a demo threshold, not a scientifically calibrated cutoff.
MIN_CONFIDENCE = 0.65

_model = None


def _result(
    prediction: str,
    score: float | None,
    message: str,
    status: str = "success",
) -> dict[str, Any]:
    return {
        "status": status,
        "prediction": prediction,
        "score": score,
        "message": message,
    }


def _inconclusive(
    message: str,
    status: str = "success",
    score: float | None = None,
) -> dict[str, Any]:
    return _result("inconclusive", score, message, status)


def _load_image(image_bytes: bytes) -> np.ndarray:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("No image bytes were provided.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large. Maximum size is 15 MB.")

    with Image.open(BytesIO(image_bytes)) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions are too large.")

        return np.array(image.convert("RGB"))


def _get_model():
    """Load the model once and reuse it for subsequent requests."""
    global _model

    if _model is not None:
        return _model

    checkpoint_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=2,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    _model = model
    return _model


def _detect_faces(
    rgb_image: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Could not load the OpenCV face detector.")

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    original_height, original_width = gray.shape

    max_dimension = 1200
    scale = min(1.0, max_dimension / max(original_width, original_height))

    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (
                max(1, int(original_width * scale)),
                max(1, int(original_height * scale)),
            ),
        )

    detected = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60),
    )

    faces = [
        (
            int(x / scale),
            int(y / scale),
            int(w / scale),
            int(h / scale),
        )
        for x, y, w, h in detected
    ]

    return sorted(faces, key=lambda face: face[2] * face[3], reverse=True)


def _prepare_face(
    rgb_image: np.ndarray,
    face: tuple[int, int, int, int],
) -> torch.Tensor:
    """Crop a face with padding and apply ImageNet normalization."""
    image_height, image_width = rgb_image.shape[:2]
    x, y, width, height = face

    # Add some context around the detected face.
    pad_x = int(width * 0.20)
    pad_y = int(height * 0.20)

    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_width, x + width + pad_x)
    bottom = min(image_height, y + height + pad_y)

    if right <= left or bottom <= top:
        raise ValueError("The detected face crop is invalid.")

    face_crop = rgb_image[top:bottom, left:right]
    face_image = Image.fromarray(face_crop).resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(face_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std

    # HWC -> CHW, then add batch dimension.
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return tensor.unsqueeze(0)


def analyze_image(image_bytes: bytes) -> dict[str, Any]:
    """
    Return the APATE response contract:
    status, prediction, score, message.
    """
    try:
        rgb_image = _load_image(image_bytes)
    except (ValueError, UnidentifiedImageError, OSError) as exc:
        message = (
            str(exc)
            if isinstance(exc, ValueError)
            else "Unsupported or invalid image file."
        )
        return _inconclusive(message, status="error")

    try:
        faces = _detect_faces(rgb_image)
    except Exception:
        return _inconclusive(
            "Face detection failed. Please try another image.",
            status="error",
        )

    if not faces:
        return _inconclusive(
            "No face was detected. This model evaluates face deepfakes, "
            "not general image edits or AI-generated images."
        )

    try:
        model = _get_model()
        face_tensor = _prepare_face(rgb_image, faces[0])

        with torch.inference_mode():
            logits = model(face_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Model card's label mapping: 0 = real, 1 = fake.
        real_score = float(probabilities[0].item())
        fake_score = float(probabilities[1].item())
        confidence = max(real_score, fake_score)

        if confidence < MIN_CONFIDENCE:
            return _inconclusive(
                "The model output is uncertain. No reliable verdict "
                "was made. Results are limited to face deepfake detection.",
                score=round(confidence, 4),
            )

        if fake_score > real_score:
            return _result(
                "potentially_manipulated",
                round(fake_score, 4),
                "The face deepfake classifier flagged this image as "
                "potentially manipulated. This is an automated model "
                "output, not proof of manipulation.",
            )

        return _result(
            "no_manipulation_detected",
            round(real_score, 4),
            "The face deepfake classifier did not flag this face as fake. "
            "This does not prove the image is authentic or unedited.",
        )

    except Exception:
        return _inconclusive(
            "Model inference failed. Check the model checkpoint and "
            "dependencies, then try again.",
            status="error",
        )
