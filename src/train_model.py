import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Correct path
df = pd.read_csv("cleaned_data/tb_cleaned.csv")

# Optional: focus on one country
df = df[df["Country"] == "India"]

X = df[["Year","Population"]]
y = df["TB_Cases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/tb_model.pkl")

print("Model trained and saved")