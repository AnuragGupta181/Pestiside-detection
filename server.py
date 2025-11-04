
from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import os
from datetime import datetime
import threading

app = Flask(__name__)

# Load YOLO model
MODEL_PATH = r"./runs/detect/weed_train/weights/best.pt"
model = YOLO(MODEL_PATH)

SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)

latest_frame = None
lock = threading.Lock()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Server running"}), 200

@app.route("/detect", methods=["POST"])
def detect():
    global latest_frame
    try:
        img = None

        if 'image' in request.files:
            img_bytes = request.files['image'].read()
        else:
            img_bytes = request.data

        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, img)

        results = model(img, verbose=False)
        annotated_frame = results[0].plot()

        detected_classes = [results[0].names[int(c)] for c in results[0].boxes.cls]
        weed_detected = any("weed" in c.lower() for c in detected_classes)

        with lock:
            latest_frame = annotated_frame.copy()

        print(f"[INFO] Detection: {detected_classes}")

        return jsonify({
            "weed_detected": weed_detected,
            "detections": detected_classes,
            "saved_image": filename
        }), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


def show_live_feed():
    global latest_frame
    while True:
        if latest_frame is not None:
            with lock:
                frame = latest_frame.copy()
            cv2.imshow("🌿 Weed Detection Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27: 
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=show_live_feed, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)