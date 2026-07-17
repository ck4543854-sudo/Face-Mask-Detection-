"""
=========================================================
 Face Mask Detection - FLASK WEB APP (Backend)
=========================================================
Browser-based frontend: Live webcam stream + Image upload,
dono se mask detection karta hai.

Usage:
    cd webapp
    python app.py

Phir browser me kholo: http://127.0.0.1:5000
=========================================================
"""

import base64
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
import time

import cv2
import numpy as np
import os
from flask import Flask, Response, jsonify, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

FACE_DETECTOR_DIR = os.path.join(PROJECT_ROOT, "face_detector")
MODEL_PATH = os.path.join(PROJECT_ROOT, "mask_detector.h5")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.5

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "webapp", "templates"),
             static_folder=os.path.join(BASE_DIR, "webapp", "static"))

# ---------------------------------------------------------
# Models load karo (server start hote hi ek baar)
# ---------------------------------------------------------
print("[INFO] loading face detector model...")
prototxtPath = os.path.join(FACE_DETECTOR_DIR, "deploy.prototxt")
weightsPath = os.path.join(FACE_DETECTOR_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNetPath = os.path.join(BASE_DIR, "mask_detector.h5")
maskNet = load_model(maskNetPath)
print("[INFO] models loaded. Server ready!")


def detect_and_predict_mask(frame):
    """Frame me faces detect karke mask/no-mask predict karta hai."""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32, verbose=0)

    return locs, preds


def annotate_frame(frame, locs, preds):
    """Bounding boxes + labels frame par draw karta hai. Returns detection summary list."""
    summary = []
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        conf = float(max(mask, withoutMask)) * 100
        color = (61, 220, 132) if label == "Mask" else (75, 75, 255)  # BGR

        label_text = f"{label}: {conf:.1f}%"
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.rectangle(frame, (startX, startY - 28), (startX + len(label_text) * 10, startY), color, -1)
        cv2.putText(frame, label_text, (startX + 4, startY - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2)

        summary.append({"label": label, "confidence": round(conf, 1)})

    return frame, summary


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


def generate_frames():
    """MJPEG generator - live webcam stream ke liye."""
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)

    if not vs.isOpened():
        print("[ERROR] Webcam open nahi ho paya.")
        return

    try:
        while True:
            success, frame = vs.read()
            if not success:
                break

            frame = cv2.resize(frame, (640, 480))
            locs, preds = detect_and_predict_mask(frame)
            frame, _ = annotate_frame(frame, locs, preds)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        vs.release()


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "Koi image file nahi mili."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Koi file select nahi ki."}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Image read nahi ho payi. Sahi image file try karo."}), 400

    locs, preds = detect_and_predict_mask(frame)
    frame, summary = annotate_frame(frame, locs, preds)

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return jsonify({"error": "Result image encode nahi ho payi."}), 500

    b64_image = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "image": f"data:image/jpeg;base64,{b64_image}",
        "faces_found": len(summary),
        "detections": summary,
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
