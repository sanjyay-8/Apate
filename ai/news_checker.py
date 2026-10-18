
"""
APATE — AI-powered news text screening.

Uses a pretrained text classifier. Predictions are not proof
that a news article or its claims are true or false.
"""

from typing import Any

_classifier = None


def _get_classifier():
    """Load the model once and reuse it."""
    global _classifier

    if _classifier is None:
        from transformers import pipeline

        _classifier = pipeline(
            "text-classification",
            model="Lennywinks/fake-news-detector-english",
        )

    return _classifier


def analyze_news(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {
            "status": "error",
            "prediction": "inconclusive",
            "score": None,
            "message": "Please provide article text.",
        }

    try:
        result = _get_classifier()(
            text[:4000],
            truncation=True,
            max_length=512,
        )[0]

        label = result["label"].strip().lower()
        score = round(float(result["score"]), 4)

        if label == "fake":
            prediction = "potentially_misleading"
            message = (
                "The AI classifier flagged this text as potentially misleading. "
                "Verify its claims using reliable, independent sources."
            )
        elif label == "real":
            prediction = "no_misleading_signals_detected"
            message = (
                "The AI classifier labelled this text as real. "
                "This does not verify the article or prove its claims are true."
            )
        else:
            prediction = "inconclusive"
            message = f"The model returned an unrecognized label: {label}."

        return {
            "status": "success",
            "prediction": prediction,
            "score": score,
            "message": message,
        }

    except Exception as exc:
        return {
            "status": "error",
            "prediction": "inconclusive",
            "score": None,
            "message": f"News analysis failed: {type(exc).__name__}: {exc}",
        }
