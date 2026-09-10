# 🤖 Multi-Project Data Science & AI Portfolio

Welcome to my portfolio repository containing **11 end-to-end** Data Science, Machine Learning, and Computer Vision web applications powered by **Python, OpenCV, Scikit-Learn, PyTorch, Ultralytics YOLO, and Streamlit**.

---

## 📋 Table of Contents

**Original 3 Projects (root folder):**
1. [Flight Price Prediction Agent](#-flight-price-prediction-agent--data-science-web-app)
2. [PAN Card Tampering & Fraud Detection App](#-pan-card-tampering--fraud-detection-app)
3. [Dog Breed Prediction App](#-dog-breed-prediction-app)

**Course Projects (8 subfolders):**
4. [Image Watermarking App](./01_image_watermark)
5. [Traffic Sign Classification](./02_traffic_sign)
6. [Text Extraction (OCR) App](./03_text_extraction)
7. [Plant Disease Prediction](./04_plant_disease)
8. [Vehicle Detection & Counting](./05_vehicle_detection)
9. [Face Swapping App](./06_face_swap)
10. [Data Science & ML Compiler App](./07_ds_compiler_app)
11. [Multilingual Indian Languages TTS App](./08_tts_indian_languages)

---

## ✈️ 1. Flight Price Prediction Agent & Data Science Web App

An end-to-end Data Science and Machine Learning web application that estimates flight ticket prices in real-time, provides market analytics, and delivers intelligent booking recommendations.
- **Model:** Random Forest Regressor (Scikit-Learn)
- **Command:** `streamlit run flight_app.py --server.port 2011 --server.address 127.0.0.1`

---

## 🔍 2. PAN Card Tampering & Fraud Detection App

A Computer Vision and Image Processing web application designed to verify authenticity and detect structural tampering, modifications, or fraud in PAN card documents using SSIM and OpenCV.
- **Command:** `streamlit run pan_app.py --server.port 2012 --server.address 127.0.0.1`

---

## 🐶 3. Dog Breed Prediction App

A deep learning and computer vision web application that classifies dog breeds from uploaded images, provides confidence scores, temperament analysis, and care statistics.
- **Model:** ResNet-18 (pre-trained via `timm`, fine-tuned)
- **Command:** `streamlit run dog_app.py --server.port 2013 --server.address 127.0.0.1`

---

## 🖼️ 4. Image Watermarking App — [`01_image_watermark/`](./01_image_watermark)

Upload an image and apply a **text watermark** (font size, opacity, position, rotation, color) or a **logo watermark** (scale, position, opacity), preview side-by-side, and download the result.
- **Tech:** Pillow (PIL) — non-destructive RGBA compositing
- **Command:** `streamlit run 01_image_watermark/app.py --server.port 2001`

---

## 🚦 5. Traffic Sign Classification — [`02_traffic_sign/`](./02_traffic_sign)

Upload a traffic-sign photo and get the **top-3 predicted classes** with confidence bars + German sign meaning. Trained on the official GTSRB dataset (12 classes, 99.5% holdout accuracy).
- **Model:** Custom small CNN (PyTorch), trained on CPU
- **Dataset:** GTSRB via HuggingFace (`tanganke/gtsrb`)
- **Command:** `streamlit run 02_traffic_sign/app.py --server.port 2002`

---

## 🔤 6. Text Extraction (OCR) App — [`03_text_extraction/`](./03_text_extraction)

Upload an image and extract all text via **OCR** with preprocessing toggles (grayscale / threshold / upscale), detected-region overlays, metric cards, and .txt download.
- **Engine:** EasyOCR (deep-learning OCR); Windows-OCR fallback
- **Command:** `streamlit run 03_text_extraction/app.py --server.port 2003`

---

## 🌱 7. Plant Disease Prediction — [`04_plant_disease/`](./04_plant_disease)

Upload a leaf photo and get the **top-3 predicted diseases** with confidence, plus treatment guidance. Trained on PlantVillage (12 disease classes, 5,940 images).
- **Model:** LeafNet — custom CNN (PyTorch), trained on CPU
- **Dataset:** PlantVillage via HuggingFace (`BrandonFors/Plant-Diseases-PlantVillage-Dataset`)
- **Command:** `streamlit run 04_plant_disease/app.py --server.port 2004`

---

## 🚗 8. Vehicle Detection & Counting — [`05_vehicle_detection/`](./05_vehicle_detection)

Upload an image or video; the app draws bounding boxes on vehicles (car, bus, truck, motorcycle, bicycle) and shows per-class + total counts. Videos get an annotated output download.
- **Model:** YOLOv8-nano (Ultralytics), pre-trained on COCO
- **Command:** `streamlit run 05_vehicle_detection/app.py --server.port 2005`

---

## 🔄 9. Face Swapping App — [`06_face_swap/`](./06_face_swap)

Upload two photos (a source face + a target person); the app detects both faces, swaps the source face onto the target, and lets you download the result.
- **Tech:** MediaPipe face landmarks + OpenCV affine warp & blending
- **Command:** `streamlit run 06_face_swap/app.py --server.port 2006`

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Vishuu77/Machine-Learning-Data-Science-Projects-.git
   cd Machine-Learning-Data-Science-Projects-
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (Each subfolder also has its own README with app-specific notes.)

3. **Note on datasets:** large raw datasets are excluded from this repo (see `.gitignore`). Run each project's `train_model.py` to re-download and retrain, or use the pre-trained `model.pt` artifacts already included.

---

## 📄 License
Open-source under the MIT License.
