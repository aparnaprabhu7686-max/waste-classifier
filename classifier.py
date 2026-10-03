"""Shared inference helper used by predict.py and app.py."""
import json
import numpy as np
from PIL import Image

import config

_model = None
_labels = None


def _load():
    global _model, _labels
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run `python train.py` first (see README).")
        from tensorflow import keras
        _model = keras.models.load_model(config.MODEL_PATH)
        _labels = json.loads(config.LABELS_PATH.read_text())
    return _model, _labels


def classify(image_file, top_k=3):
    """image_file: path or file-like object. Returns a result dict."""
    model, labels = _load()
    img = Image.open(image_file).convert("RGB").resize(config.IMG_SIZE)
    arr = np.expand_dims(np.asarray(img, dtype="float32"), 0)  # preprocessing is inside the model
    probs = model.predict(arr, verbose=0)[0]
    order = np.argsort(probs)[::-1][:top_k]
    predictions = [{"label": labels[i], "confidence": float(probs[i])} for i in order]
    best = predictions[0]["label"]
    return {
        "label": best,
        "confidence": predictions[0]["confidence"],
        "predictions": predictions,
        **config.TIPS.get(best, {}),
    }
