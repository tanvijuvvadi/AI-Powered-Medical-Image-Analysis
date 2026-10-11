"""Minimal Flask API for the 'doctor-assist' demo.  Research use only."""
from flask import Flask, jsonify, request

from src.predict import get_model, predict_gray
from src.preprocessing import decode_gray

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB upload limit
ALLOWED = {".jpg", ".jpeg", ".png"}
DISCLAIMER = "Educational demo. Not a medical device. Not for clinical use."


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="Send a file in form field 'image'"), 400
    if not any(file.filename.lower().endswith(e) for e in ALLOWED):
        return jsonify(error="Only .jpg/.jpeg/.png supported"), 400
    try:
        gray = decode_gray(file.read())
    except ValueError as e:
        return jsonify(error=str(e)), 400
    label, prob = predict_gray(gray, get_model())
    return jsonify(prediction=label, probability_pneumonia=round(prob, 4), disclaimer=DISCLAIMER)


def run():
    app.run(host="127.0.0.1", port=5000, debug=False)
