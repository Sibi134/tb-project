import joblib
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model path
model_path = os.path.join(BASE_DIR, "models", "tb_model.pkl")

# Load trained model
model = joblib.load(model_path)


def predict_tb(year, population):
    prediction = model.predict([[year, population]])
    return int(prediction[0])