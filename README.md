# 🤖 Multi-Project Data Science & AI Portfolio

Welcome to my portfolio repository containing end-to-end Data Science, Machine Learning, and Computer Vision web applications powered by **Python, OpenCV, Scikit-Learn, and Streamlit**.

---

## 📋 Table of Contents
1. [Flight Price Prediction Agent](#-flight-price-prediction-agent--data-science-web-app)
2. [PAN Card Tampering & Fraud Detection App](#-pan-card-tampering--fraud-detection-app)

---

## ✈️ 1. Flight Price Prediction Agent & Data Science Web App

An end-to-end Data Science and Machine Learning web application that estimates flight ticket prices in real-time, provides market analytics, and delivers intelligent booking recommendations.

### 🌟 Features
- **🎫 Real-Time Price Predictor UI**: Interactive inputs for Airline, Source, Destination, Total Stops, Journey Date, Departure Time, and Duration with confidence intervals.
- **📊 Market Analytics & EDA**: Visualizes average pricing trends across different airlines and stop counts.
- **🤖 Autonomous Agent Insights**: Booking window optimization and duration-vs-price correlation analysis.
- **⚙️ Model Diagnostics**: Displays Random Forest Regressor hyperparameters and feature importances.

### 🚀 Running the Flight App (Port 2001)
```bash
streamlit run flight_app.py --server.port 2001 --server.address 127.0.0.1
```
🌐 **URL:** `http://127.0.0.1:2001`

---

## 🔍 2. PAN Card Tampering & Fraud Detection App

A Computer Vision and Image Processing web application designed to verify authenticity and detect structural tampering, modifications, or fraud in PAN card documents.

### 🌟 Features
- **🛡️ SSIM & OpenCV Engine**: Compares structural similarity, fonts, and alignment between reference (original) and target PAN cards.
- **🟥 Visual Difference Mapping**: Automatically highlights altered or tampered regions with red bounding boxes.
- **📊 Similarity Scoring**: Outputs real-time similarity percentages and determines whether the card is **Authentic ✅** or **Tampered ❌**.

### 🚀 Running the PAN App (Port 2002)
```bash
streamlit run pan_app.py --server.port 2002 --server.address 127.0.0.1
```
🌐 **URL:** `http://127.0.0.1:2002`

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

---

## 📄 License
Open-source under the MIT License.
