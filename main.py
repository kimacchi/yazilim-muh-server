import joblib
import pandas as pd

from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load the saved pipeline
joblib_file = "trained_pipeline.pkl"
loaded_pipeline = joblib.load(joblib_file)

print(f"Model loaded from {joblib_file}")

# # New data input
# new_data = pd.DataFrame({
#     "BHK": [2],
#     "Size": [800],
#     "Floor": ["1 out of 0"],
#     "Area Type": ["Super Area"],
#     "Area Locality": ["Phool Bagan, Kankurgachi"],
#     "City": ["Kolkata"],
#     "Furnishing Status": ["Semi-Furnished"],
#     "Tenant Preferred": ["Bachelors/Family"],
#     "Bathroom": [1],
#     "Point of Contact": ["Contact Owner"]
# })

# # Ensure new data has the same column order as the training data


# # Make a prediction using the loaded pipeline
# prediction = loaded_pipeline.predict(new_data)
# print(f"Predicted value: {prediction[0]}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict-value")
def predict_value(data: dict) -> Union[dict, str]:
    # Convert the data into a DataFrame
    print(data)
    new_data = pd.DataFrame(data)
    
    # Make a prediction using the loaded pipeline
    prediction = loaded_pipeline.predict(new_data)
    
    return {"prediction": prediction[0]}


