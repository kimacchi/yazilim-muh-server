import joblib
import pandas as pd

# Load the saved pipeline
joblib_file = "trained_pipeline.pkl"
loaded_pipeline = joblib.load(joblib_file)

print(f"Model loaded from {joblib_file}")

# New data input
new_data = pd.DataFrame({
    'BHK': [2],
    'Size': [800],
    'Floor': ['1 out of 3'],
    'Area Type': ['Super Area'],
    'Area Locality': ['Phool Bagan, Kankurgachi'],
    'City': ['Kolkata'],
    'Furnishing Status': ['Semi-Furnished'],
    'Tenant Preferred': ['Bachelors/Family'],
    'Bathroom': [1],
    'Point of Contact': ['Contact Owner']
})

# Ensure new data has the same column order as the training data


# Make a prediction using the loaded pipeline
prediction = loaded_pipeline.predict(new_data)
print(f"Predicted value: {prediction[0]}")