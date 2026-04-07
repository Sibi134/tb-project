import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load raw datasets
beds = pd.read_csv(os.path.join(BASE_DIR, "cleaned_data", "hospital_beds.csv"), skiprows=4)
doctors = pd.read_csv(os.path.join(BASE_DIR, "cleaned_data", "doctors.csv"), skiprows=4)

# ---------------------------
# Process hospital beds
# ---------------------------
beds = beds[["Country Name", "2020"]]
beds.columns = ["Country", "Beds_per_1000"]
beds = beds.dropna()

# ---------------------------
# Process doctors
# ---------------------------
doctors = doctors[["Country Name", "2020"]]
doctors.columns = ["Country", "Doctors_per_1000"]
doctors = doctors.dropna()

# ---------------------------
# Save cleaned datasets
# ---------------------------
beds.to_csv(os.path.join(BASE_DIR, "cleaned_data", "beds_cleaned.csv"), index=False)
doctors.to_csv(os.path.join(BASE_DIR, "cleaned_data", "doctors_cleaned.csv"), index=False)

print("✅ Health datasets cleaned and saved")