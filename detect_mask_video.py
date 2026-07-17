"""
=========================================================
 Face Mask Detection - REAL-TIME VIDEO / WEBCAM DETECTION
=========================================================
Webcam se live video capture karke har frame me face detect
karta hai aur predict karta hai ki Mask hai ya No Mask.

Usage:
    python detect_mask_video.py
    python detect_mask_video.py --face face_detector --model mask_detector.model --confidence 0.5

Quit karne ke liye 'q' press karo.
=========================================================
"""

import argparse
import time
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
import imutils
from imutils.video import VideoStream

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str, default="face_detector",
                     help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                     help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                     help="minimum probability to filter weak face detections")
    return ap.parse_args()


def detect_and_predict_mask(frame, faceNet, maskNet, confidence_threshold):
    """
    Frame me se faces detect karta hai aur unhe mask model ko
    predict ke liye bhejta hai.
    Returns: (face_locations, mask_predictions)
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # bounding box frame ke andar hi rahe
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
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


def main():
    args = parse_args()

    # Face detector load karo (Caffe DNN model)
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.join(args.face, "deploy.prototxt")
    weightsPath = os.path.join(args.face, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
        print("[ERROR] Face detector files nahi mile! 'face_detector' folder check karo.")
        return

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # Mask classifier load karo
    print("[INFO] loading face mask detector model...")
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file '{args.model}' nahi mila!")
        print("Pehle train_mask_detector.py chalao ya pretrained model daalo.")
        return

    maskNet = load_model(args.model)

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    prev_time = time.time()

    while True:
        frame = vs.read()
        if frame is None:
            print("[ERROR] Webcam se frame nahi mil raha. Camera check karo.")
            break

        frame = imutils.resize(frame, width=700)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, args.confidence)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Face Mask Detector - Press 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    main()
