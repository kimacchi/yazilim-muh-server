from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.simplefilter(action='ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.svm import SVR
import sklearn.metrics as metrics
import xgboost as xgb
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the dataset
dataset = pd.read_csv(r"./dataset/House_Rent_Dataset.csv")
dataset=dataset[np.abs(stats.zscore(dataset["Rent"])) < 3]

dataset=dataset.drop(columns="Posted On")
# Splitting the data into features (X) and target (y)
x=dataset.drop(columns="Rent")
y=dataset["Rent"]

# Identify categorical and numerical columns
categorical_columns = x.select_dtypes(include=['object']).columns
numerical_columns = x.select_dtypes(exclude=['object']).columns



# Preprocessor: handle categorical and numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ])

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Print the score of the model
score = pipeline.score(X_test, y_test)
print(f"Model Score: {score}")

# Fit the pipeline (from the previous example)
pipeline.fit(X_train, y_train)

# Save the pipeline to a file
joblib_file = "trained_pipeline.pkl"
joblib.dump(pipeline, joblib_file)

print(f"Model saved to {joblib_file}")


# New data input
# new_data = pd.DataFrame({
#     'BHK': [2],
#     'Size': [800],
#     'Floor': ['Ground out of 2'],
#     'Area Type': ['Super Area'],
#     'Area Locality': ['Bandel'],
#     'City': ['Kolkata'],
#     'Furnishing Status': ['Unfurnished'],
#     'Tenant Preferred': ['Bachelors/Family'],
#     'Bathroom': [2],
#     'Point of Contact': ['Contact Owner']
# })

# # Make a prediction
# prediction = pipeline.predict(new_data)
# print(f"Predicted value: {prediction[0]}")
