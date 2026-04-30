import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("data/processed/dataset.csv", header=None)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained & saved!")