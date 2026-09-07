# ✈️ Flight Price Prediction Agent & Data Science Web App

An end-to-end Data Science and Machine Learning web application powered by **Python, Pandas, Scikit-Learn, and Streamlit**. This project features an autonomous predictive agent that estimates flight ticket prices in real-time, provides market analytics, and delivers intelligent booking recommendations.

---

## 🌟 Features

1. **🎫 Real-Time Price Predictor UI**:
   - Interactive inputs for Airline, Source, Destination, Total Stops, Journey Date, Departure Time, and Duration.
   - Real-time price estimation with upper/lower bounds confidence intervals.
   - Autonomous AI agent recommendation engine (*Great Deal*, *Fair Price*, or *Premium Warning*).

2. **📊 Market Analytics & EDA**:
   - Visualizes average pricing trends across different airlines and stop counts.
   - Exploratory data analysis (EDA) charts and dataset inspection viewer.

3. **🤖 Autonomous Agent Insights**:
   - Booking window optimization and duration-vs-price correlation analysis.

4. **⚙️ Model Diagnostics**:
   - Displays Random Forest Regressor hyperparameters, test set performance ($R^2$ Score: `0.7832`), and feature importances.

---

## 📂 Project Structure

```text
Flight-Price-Prediction-Agent/
│
├── Data_Train.xlsx                     # Historical training dataset
├── Test_set.xlsx                       # Test dataset
├── flight_app.py                       # Main Streamlit UI application (runs on port 2001)
├── train_model.py                      # ML model training script (Random Forest)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # Project documentation
```

---

## 🚀 Getting Started & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Flight-Price-Prediction-Agent.git
cd Flight-Price-Prediction-Agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional - pre-trained artifacts included)
```bash
python train_model.py
```

### 4. Run the Streamlit App (Port 2001)
```bash
streamlit run flight_app.py --server.port 2001 --server.address 127.0.0.1
```
Open your browser at **`http://127.0.0.1:2001`**.

---

## 🧠 Machine Learning Core
- **Algorithm**: Random Forest Regressor ($n\_estimators = 100$)
- **Preprocessing**: Label Encoding for categorical features, time feature extraction (day, month, hour, minute), and custom duration parsing.
- **Performance**: Test Set $R^2$ Score = `0.7832`

---

## 📄 License
This project is open-source under the MIT License.
