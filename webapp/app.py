"""Flask web app + JSON API for the food classifier.

    python -m webapp.app                       # uses models/best.pt
    MODEL_PATH=runs/efficientnet_b0/best.pt python -m webapp.app

Endpoints
    GET  /                 web UI
    POST /api/predict      multipart field "file" -> prediction JSON (top-k, Grad-CAM, reject flag)
    GET  /api/health       liveness + model loaded?
    GET  /api/model-info   architecture, size, held-out test metrics, calibration
    GET  /api/classes      list of supported dishes
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from foodai.data import load_rgb
from foodai.predict import Predictor
from foodai.utils import count_parameters, load_json

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _normalise(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def create_app(model_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads
    Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs

    model_path = Path(model_path or os.environ.get("MODEL_PATH", ROOT / "models" / "best.pt"))
    nutrition_path = ROOT / "data" / "nutrition.json"
    nutrition = {_normalise(k): v for k, v in load_json(nutrition_path).items() if not k.startswith("_")} \
        if nutrition_path.exists() else {}

    predictor: Predictor | None = None
    load_error: str | None = None
    gate = None
    if os.environ.get("FOOD_GATE", "1") != "0":  # set FOOD_GATE=0 to disable the 'is this food?' check
        try:
            from foodai.foodgate import FoodGate
            gate = FoodGate(threshold=float(os.environ.get("FOOD_GATE_THRESHOLD", 0.5)))
        except Exception as exc:  # transformers missing or CLIP could not be downloaded
            print(f"[warning] food gate disabled, non-food images will get a dish name: {exc}")
    if model_path.exists():
        try:
            predictor = Predictor(str(model_path), device=os.environ.get("DEVICE", "cpu"), gate=gate)
        except Exception as exc:  # corrupt / incompatible checkpoint
            load_error = f"Could not load model: {exc}"
    else:
        load_error = f"Model file not found: {model_path}. Train one first (see README) or set MODEL_PATH."
    lock = threading.Lock()  # Grad-CAM registers temporary hooks, so serialise inference

    @app.get("/")
    def index():
        return render_template("index.html", ready=predictor is not None, error=load_error)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok" if predictor else "model_unavailable", model_loaded=predictor is not None,
                       detail=load_error)

    @app.get("/api/classes")
    def classes():
        if not predictor:
            return jsonify(error=load_error), 503
        return jsonify(classes=predictor.classes)

    @app.get("/api/model-info")
    def model_info():
        if not predictor:
            return jsonify(error=load_error), 503
        meta = predictor.meta
        return jsonify(
            arch=predictor.arch, num_classes=len(predictor.classes), img_size=meta["img_size"],
            params=count_parameters(predictor.model), temperature=predictor.temperature,
            reject_threshold=predictor.threshold, test_metrics=meta.get("test_metrics", {}),
        )

    @app.post("/api/predict")
    def predict():
        if not predictor:
            return jsonify(error=load_error), 503
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify(error="No file uploaded. Send an image in the multipart field 'file'."), 400
        if Path(file.filename).suffix.lower() not in ALLOWED_EXT:
            return jsonify(error="Unsupported file type. Please upload a JPG, PNG, WEBP or BMP image."), 400
        try:
            image = load_rgb(file.stream)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify(error="That file could not be read as an image."), 400
        with lock:
            result = predictor.predict(image, topk=3, explain=True)
        info = nutrition.get(_normalise(result["label"])) if result["is_food"] else None
        if info:
            result["nutrition"] = {"serving": info["serving"], "kcal_min": info["kcal"][0],
                                   "kcal_max": info["kcal"][1], "note": "rough estimate"}
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Image is too large (limit 8 MB)."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
