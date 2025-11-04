

from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import os
from datetime import datetime
import threading

app = Flask(__name__)
model = YOLO(r"./runs/detect/weed_train/weights/best.pt")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Server running"}), 200

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model.predict(img, conf=0.25)
    annotated = results[0].plot()

    # Show annotated image in real-time
    cv2.imshow("🌿 Weed Detection Live", annotated)
    cv2.waitKey(1)

    # Check for weed class (class 0)
    weed_detected = any(
        int(box.cls[0].item()) == 0 for box in results[0].boxes
    )

    return jsonify({"weed_detected": weed_detected})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
