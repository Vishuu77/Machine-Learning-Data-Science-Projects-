import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

print("📥 Loading training data...")
df = pd.read_excel('C:/Users/Admin/flight_project/Data_Train.xlsx')

print(f"📊 Dataset shape: {df.shape}")
print("🧹 Cleaning and feature engineering...")

# Drop missing values
df.dropna(inplace=True)

# Feature Engineering
# Date of Journey
df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.day
df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.month

# Dep Time
df['Dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_min'] = pd.to_datetime(df['Dep_Time']).dt.minute

# Duration parsing (e.g., '2h 50m' -> minutes)
def parse_duration(duration_str):
    parts = duration_str.split()
    hours = 0
    mins = 0
    for part in parts:
        if 'h' in part:
            hours = int(part.replace('h', ''))
        elif 'm' in part:
            mins = int(part.replace('m', ''))
    return hours * 60 + mins

df['Duration_mins'] = df['Duration'].apply(parse_duration)

# Total Stops mapping
stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
df['Total_Stops_Encoded'] = df['Total_Stops'].map(stops_mapping).fillna(0)

# Categorical Label Encoding
le_airline = LabelEncoder()
le_source = LabelEncoder()
le_dest = LabelEncoder()

df['Airline_Encoded'] = le_airline.fit_transform(df['Airline'])
df['Source_Encoded'] = le_source.fit_transform(df['Source'])
df['Destination_Encoded'] = le_dest.fit_transform(df['Destination'])

features = [
    'Airline_Encoded', 'Source_Encoded', 'Destination_Encoded',
    'Total_Stops_Encoded', 'Journey_Day', 'Journey_Month',
    'Dep_hour', 'Dep_min', 'Duration_mins'
]

X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"✅ Model R^2 Score on Test Set: {score:.4f}")

# Save model and encoders
os.makedirs('C:/Users/Admin/flight_project/model', exist_ok=True)
with open('C:/Users/Admin/flight_project/model/flight_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('C:/Users/Admin/flight_project/model/le_airline.pkl', 'wb') as f:
    pickle.dump(le_airline, f)
with open('C:/Users/Admin/flight_project/model/le_source.pkl', 'wb') as f:
    pickle.dump(le_source, f)
with open('C:/Users/Admin/flight_project/model/le_dest.pkl', 'wb') as f:
    pickle.dump(le_dest, f)

print("💾 Model and encoders saved successfully!")
