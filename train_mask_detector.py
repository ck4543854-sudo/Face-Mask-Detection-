"""
=========================================================
 Face Mask Detection - MODEL TRAINING SCRIPT
=========================================================
Yeh script MobileNetV2 (transfer learning) use karke
face mask classifier (Mask / No Mask) train karta hai.

Dataset structure required:
    dataset/
        with_mask/     -> mask wali images
        without_mask/  -> bina mask wali images

Dataset Kaggle se download karo:
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
(ya "Face Mask Detection Dataset" search karo)

Usage:
    python train_mask_detector.py --dataset dataset --model mask_detector.model
=========================================================
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # GUI ke bina bhi graph save ho jaye
import matplotlib.pyplot as plt

import numpy as np
from imutils import paths

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, default="dataset",
                     help="path to input dataset (with_mask / without_mask folders)")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                     help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.keras",
                     help="path to output trained model (.h5 / .model)")
    ap.add_argument("-e", "--epochs", type=int, default=20,
                     help="number of training epochs")
    ap.add_argument("-b", "--batch_size", type=int, default=32,
                     help="training batch size")
    return ap.parse_args()


def load_dataset(dataset_path):
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset_path))
    data = []
    labels = []

    for imagePath in imagePaths:
        # folder name hi label hai: with_mask / without_mask
        label = imagePath.split(os.path.sep)[-2]

        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels


def build_model():
    """MobileNetV2 base (ImageNet pretrained) + custom classification head"""
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                             input_tensor=Input(shape=(224, 224, 3)))

    # base model ka top hata kar apna head banate hain
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    # base layers freeze karo taaki sirf head train ho (transfer learning)
    for layer in baseModel.layers:
        layer.trainable = False

    return model


def main():
    args = parse_args()

    INIT_LR = 1e-4
    EPOCHS = args.epochs
    BS = args.batch_size

    if not os.path.isdir(args.dataset):
        print(f"[ERROR] Dataset folder '{args.dataset}' nahi mila!")
        print("Pehle Kaggle se dataset download karke dataset/with_mask aur")
        print("dataset/without_mask folders me daalo.")
        return

    data, labels = load_dataset(args.dataset)

    if len(data) == 0:
        print("[ERROR] Dataset khaali hai! Kripya images add karo.")
        return

    # labels ko one-hot encode karo
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # train/test split (80/20)
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42)

    # data augmentation (overfitting kam karne ke liye)
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    print("[INFO] compiling model...")
    model = build_model()
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(testY.argmax(axis=1), predIdxs,
                                 target_names=lb.classes_))

    print(f"[INFO] saving mask detector model to '{args.model}'...")
    model.save(args.model)

    # accuracy/loss graph plot karke save karo
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args.plot)
    print(f"[INFO] training plot saved to '{args.plot}'")


if __name__ == "__main__":
    main()
