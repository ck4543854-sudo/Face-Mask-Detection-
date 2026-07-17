# 😷 Face Mask Detection System (Deep Learning + OpenCV)

Real-time face mask detector jo webcam feed se detect karta hai ki
person ne mask pehna hai ya nahi, using **MobileNetV2 (Transfer Learning)**
aur **OpenCV DNN Face Detector**.

---

## 📁 Project Structure

```
face_mask_detector/
├── dataset/
│   ├── with_mask/          <- mask wali training images yaha daalo
│   └── without_mask/       <- bina mask wali training images yaha daalo
├── face_detector/
│   ├── deploy.prototxt                              ✅ included (pretrained)
│   └── res10_300x300_ssd_iter_140000.caffemodel     ✅ included (pretrained)
├── mask_detector.model      ✅ included (pretrained, ready to use!)
├── train_mask_detector.py   <- naya model train karne ke liye
├── detect_mask_video.py     <- webcam se real-time detection
├── detect_mask_image.py     <- single image par detection
├── requirements.txt
└── README.md
```

> ✅ **Face detector aur pretrained mask_detector.model already included hai** —
> tum bina training ke seedha real-time detection run kar sakte ho!

---

## 🚀 Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Real-time webcam detection chalao (pretrained model ke saath)
```bash
python detect_mask_video.py
```
- `q` press karo quit karne ke liye
- Green box = Mask ✅ | Red box = No Mask ❌

### 3. Single image par test karo
```bash
python detect_mask_image.py --image path/to/photo.jpg
```
Output `output.jpg` me save hoga.

---

## 🏋️ Apna Model Train Karna (Optional)

Agar apna khud ka model train karna hai (better accuracy / apne dataset ke saath):

### Step 1: Dataset download karo
Kaggle se "Face Mask Detection Dataset" download karo:
👉 https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

Extract karke is structure me daalo:
```
dataset/
├── with_mask/       (mask images yaha)
└── without_mask/    (bina mask images yaha)
```

### Step 2: Training chalao
```bash
python train_mask_detector.py --dataset dataset --epochs 20 --model mask_detector.model
```

Training complete hone ke baad:
- `mask_detector.model` — trained model save hoga
- `plot.png` — accuracy/loss graph save hoga

### Step 3: Naye model se detect karo
```bash
python detect_mask_video.py --model mask_detector.model
```

---

## ⚙️ Command Line Arguments

### `train_mask_detector.py`
| Argument | Default | Description |
|---|---|---|
| `--dataset` | `dataset` | Dataset folder path |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--model` | `mask_detector.model` | Output model path |
| `--plot` | `plot.png` | Output graph path |

### `detect_mask_video.py`
| Argument | Default | Description |
|---|---|---|
| `--face` | `face_detector` | Face detector folder |
| `--model` | `mask_detector.model` | Mask classifier model |
| `--confidence` | `0.5` | Minimum detection confidence |

---

## 🧠 How It Works

```
Webcam Frame
     │
     ▼
Face Detection (OpenCV DNN - ResNet SSD)
     │
     ▼
Extract Face ROI → Resize (224x224) → Preprocess
     │
     ▼
MobileNetV2 Classifier → [Mask, No Mask]
     │
     ▼
Draw Bounding Box + Label + Confidence %
```

---

## 🛠️ Tech Stack
- **Python 3.8+**
- **TensorFlow / Keras** — MobileNetV2 transfer learning
- **OpenCV** — face detection (DNN module) + video capture
- **imutils** — video stream helper
- **NumPy, Matplotlib, scikit-learn**

---

## 📊 Model Architecture

```
MobileNetV2 (ImageNet pretrained, base frozen)
        │
AveragePooling2D (7x7)
        │
Flatten
        │
Dense(128, relu)
        │
Dropout(0.5)
        │
Dense(2, softmax)  →  [Mask, No Mask]
```

---

## 🔧 Troubleshooting

**Webcam nahi khul raha?**
- Check karo koi aur app webcam use to nahi kar raha
- `VideoStream(src=0)` me `src` value change karke try karo (0, 1, 2...)

**"No module named cv2" error?**
```bash
pip install opencv-python
```

**Low FPS / slow detection?**
- Frame resize width kam karo (`imutils.resize(frame, width=500)`)
- GPU version install karo: `pip install tensorflow-gpu`

**Model accuracy kam hai?**
- Zyada aur diverse dataset use karo
- Epochs badhao (`--epochs 30`)
- Data augmentation already included hai training script me

---

## 📌 Future Enhancements
- Multiple mask types classification (surgical, N95, cloth)
- IoT integration (auto door lock / alert siren)
- Mobile deployment (TensorFlow Lite)
- Attendance system integration

---

## 📄 Credits
Face detector model: OpenCV pretrained ResNet-10 SSD (Caffe)
Mask classifier: MobileNetV2 transfer learning approach
