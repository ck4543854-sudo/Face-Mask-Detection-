"""
=========================================================
 Face Mask Detection - SINGLE IMAGE DETECTION
=========================================================
Ek single image file par mask detection run karta hai
aur result ko save/display karta hai.

Usage:
    python detect_mask_image.py --image path/to/photo.jpg
=========================================================
"""

import argparse
import os

import cv2
import numpy as np
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                     help="path to input image")
    ap.add_argument("-f", "--face", type=str, default="face_detector",
                     help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                     help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                     help="minimum probability to filter weak detections")
    ap.add_argument("-o", "--output", type=str, default="output.jpg",
                     help="path to save output annotated image")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image '{args.image}' nahi mili!")
        return

    print("[INFO] loading face detector model...")
    prototxtPath = os.path.join(args.face, "deploy.prototxt")
    weightsPath = os.path.join(args.face, "res10_300x300_ssd_iter_140000.caffemodel")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args.model)

    image = cv2.imread(args.image)
    orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    face_count = 0

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args.confidence:
            face_count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = maskNet.predict(face)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

            cv2.putText(image, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

            print(f"[RESULT] Face {face_count}: {label_text}")

    if face_count == 0:
        print("[INFO] Koi face detect nahi hui image me.")

    cv2.imwrite(args.output, image)
    print(f"[INFO] Annotated image saved to '{args.output}'")


if __name__ == "__main__":
    main()
