"""Local image classification workbench for the bundled Keras model."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "best_model_fft_baseline.keras"))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
model = None
model_error = None


def load_model():
    global model, model_error
    if model is not None or model_error is not None:
        return
    if not MODEL_PATH.is_file():
        model_error = f"Model file not found: {MODEL_PATH.name}"
        return
    try:
        import tensorflow as tf
        # The notebook defines this preprocessing layer inline, so it is not
        # importable as a Python module when the saved model is loaded here.
        @tf.keras.utils.register_keras_serializable()
        class ResNetPreprocess(tf.keras.layers.Layer):
            def call(self, inputs):
                return tf.keras.applications.resnet50.preprocess_input(
                    tf.cast(inputs, tf.float32)
                )

        model = tf.keras.models.load_model(
            MODEL_PATH, compile=False, custom_objects={"ResNetPreprocess": ResNetPreprocess}
        )
    except Exception as exc:  # surface runtime/dependency problems in the UI
        model_error = f"Could not load model: {exc}"


@app.get("/")
def index():
    load_model()
    return render_template("index.html", model_ready=model is not None,
                           model_name=MODEL_PATH.name, model_error=model_error)


@app.get("/api/status")
def status():
    load_model()
    return jsonify({"ready": model is not None, "model": MODEL_PATH.name,
                    "error": model_error, "input_size": 200,
                    "labels": {"0": "Authentic", "1": "Manipulated"}})


@app.post("/api/predict")
def predict():
    load_model()
    if model is None:
        return jsonify({"error": model_error or "Model is unavailable."}), 503
    files = request.files.getlist("images")
    if not files or all(not f.filename for f in files):
        return jsonify({"error": "Choose at least one image."}), 400
    threshold = request.form.get("threshold", "0.5")
    try:
        threshold = float(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Threshold must be between 0 and 1."}), 400
    results = []
    for upload in files[:20]:
        if not upload.filename:
            continue
        try:
            image = Image.open(io.BytesIO(upload.read())).convert("RGB")
            width, height = image.size
            # Matches the notebook's deterministic center crop / pad pipeline.
            w, h = image.size
            if w >= 200 and h >= 200:
                left, top = (w - 200) // 2, (h - 200) // 2
                image = image.crop((left, top, left + 200, top + 200))
            else:
                canvas = Image.new("RGB", (200, 200))
                left, top = (200 - w) // 2, (200 - h) // 2
                canvas.paste(image, (left, top))
                image = canvas
            array = np.asarray(image, dtype=np.uint8)[None, ...]
            started = time.perf_counter()
            score = float(np.asarray(model.predict(array, verbose=0)).reshape(-1)[0])
            elapsed = (time.perf_counter() - started) * 1000
            results.append({"filename": upload.filename, "width": width, "height": height,
                            "score": score, "label": "Manipulated" if score >= threshold else "Authentic",
                            "confidence": score if score >= threshold else 1 - score,
                            "latency_ms": round(elapsed, 1)})
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            results.append({"filename": upload.filename, "error": f"Invalid image: {exc}"})
    return jsonify({"results": results, "threshold": threshold})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Upload is too large. The limit is 20 MB per request."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
