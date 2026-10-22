import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("🤖 [ML Agent Initializing]: Loading dataset...")
data = {
    'Usage_Hours': [10, 2, 25, 1, 30, 5, 20, 2],
    'Support_Tickets': [0, 5, 1, 4, 0, 3, 1, 6],
    'Subscription_Months': [12, 1, 24, 2, 36, 3, 18, 1],
    'Churned': [0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df[['Usage_Hours', 'Support_Tickets', 'Subscription_Months']]
y = df['Churned']

print("🧠 [ML Agent Training]: Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)
print("✅ Model trained successfully!\n")

def ml_agent_predict(usage, tickets, months):
    prediction = model.predict([[usage, tickets, months]])
    probability = model.predict_proba([[usage, tickets, months]])[0]
    status = "At Risk of Churn ❌" if prediction[0] == 1 else "Loyal Customer ✅"
    return status, probability[prediction[0]]

print("🚀 [Agent Execution]: Running autonomous decisions on incoming customer profiles...")
test_cases = [
    (22, 0, 15), 
    (1, 4, 2)    
]

for i, (u, t, m) in enumerate(test_cases, 1):
    status, conf = ml_agent_predict(u, t, m)
    print(f"Customer {i} (Usage: {u}h, Tickets: {t}, Months: {m}) -> Agent Decision: {status} (Confidence: {conf*100:.1f}%)")
