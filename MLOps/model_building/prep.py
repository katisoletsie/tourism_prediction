import numpy as np 
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/katisoletsie/TourismPackagePrediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

df.drop(columns=["CustomerID"], inplace=True)   # complete the code: drop the customer identifier column, it is not a predictive feature

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"  # complete the code to set the name of the column to predict (whether customer purchased the package), 1 if the customer purchased the package, else 0

numeric_features = [
    'Age',
    'DurationOfPitch',
    'MonthlyIncome',
    'NumberOfFollowups', 
    'NumberOfPersonVisiting', 
    'NumberOfFollowups', 
    'NumberOfTrips',
    'NumberOfChildrenVisiting'         
]

categorical_features = [
    'TypeofContact', 
    'CityTier', 
    'Occupation', 
    'Gender', 
    'ProductPitched', 
    'PreferredPropertyStar', 
    'MaritalStatus',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar', 
    'Designation'    
]

# Define predictor matrix (X) using selected numeric and categorical features
X = df[numeric_features + categorical_features]

# Define target variable
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
# complete the code: which variable should stay balanced across the splits?
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="katisoletsie/TourismPackagePrediction",
        repo_type="dataset",
    )
