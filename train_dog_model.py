import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Starting Dog Breed ML Model Training...")

# Define Dog Breeds
breeds = [
    "Labrador Retriever",
    "Golden Retriever",
    "German Shepherd",
    "Siberian Husky",
    "Bulldog",
    "Beagle",
    "Poodle"
]

X = []
y = []

# Simulate robust training feature vectors representing different dog visual characteristics
# Features: [Red Mean, Green Mean, Blue Mean, Hue Mean, Saturation Mean, Value Mean, Edge Density, Aspect Ratio, Texture Variance]
np.random.seed(42)

breed_profiles = {
    "Labrador Retriever": {"r": (120, 150), "g": (100, 130), "b": (80, 110), "s": (50, 90), "edge": (0.08, 0.18)},
    "Golden Retriever":   {"r": (140, 170), "g": (115, 145), "b": (85, 115), "s": (70, 110), "edge": (0.15, 0.28)},
    "German Shepherd":    {"r": (70, 105),  "g": (75, 110),  "b": (65, 95),   "s": (40, 80),  "edge": (0.20, 0.35)},
    "Siberian Husky":     {"r": (150, 185), "g": (145, 180), "b": (140, 175), "s": (10, 45),  "edge": (0.12, 0.25)},
    "Bulldog":            {"r": (110, 140), "g": (95, 125),  "b": (80, 110),  "s": (30, 70),  "edge": (0.22, 0.38)},
    "Beagle":             {"r": (100, 130), "g": (85, 115),  "b": (70, 100),  "s": (50, 90),  "edge": (0.14, 0.26)},
    "Poodle":             {"r": (125, 160), "g": (110, 145), "b": (95, 130),  "s": (20, 60),  "edge": (0.25, 0.42)}
}

for breed, profile in breed_profiles.items():
    for _ in range(300): # 300 samples per breed for robust training
        r = np.random.uniform(profile["r"][0], profile["r"][1])
        g = np.random.uniform(profile["g"][0], profile["g"][1])
        b = np.random.uniform(profile["b"][0], profile["b"][1])
        h = np.random.uniform(10, 170)
        s = np.random.uniform(profile["s"][0], profile["s"][1])
        v = np.random.uniform(80, 220)
        edge = np.random.uniform(profile["edge"][0], profile["edge"][1])
        aspect = np.random.uniform(0.8, 1.3)
        texture_var = np.random.uniform(10, 50)
        
        feature_vector = [r, g, b, h, s, v, edge, aspect, texture_var]
        X.append(feature_vector)
        y.append(breed)

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Training Complete! Validation Accuracy: {acc * 100:.2f}%")

# Save model
os.makedirs(r"C:/Users/Admin/flight_project/model", exist_ok=True)
model_path = r"C:/Users/Admin/flight_project/model/dog_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print(f"Model successfully saved to {model_path}")
