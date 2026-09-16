# for data manipulation
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")   # complete the code to set the MLflow tracking URI
mlflow.set_experiment("TourismPackagePrediction")     # complete the code to set the MLflow experiment name (same as the dev experimentation cell)

api = HfApi()

Xtrain_path = "hf://datasets/katisoletsie/TourismPackagePrediction/Xtrain.csv"
Xtest_path = "hf://datasets/katisoletsie/TourismPackagePrediction/Xtest.csv"
ytrain_path = "hf://datasets/katisoletsie/TourismPackagePrediction/ytrain.csv"
ytest_path = "hf://datasets/katisoletsie/TourismPackagePrediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# complete the code to list all numerical feature names (same as in prep.py)   
numeric_features = [
    'Age',
    'DurationOfPitch',
    'MonthlyIncome',
    'NumberOfFollowups', 
    'NumberOfPersonVisiting',  
    'NumberOfTrips',
    'NumberOfChildrenVisiting',                  
]
# complete the code to list all categorical feature names (same as in prep.py)
categorical_features = [
    'TypeofContact', 
    'CityTier', 
    'PreferredPropertyStar',
    'Gender', 
    'ProductPitched', 
    'PitchSatisfactionScore',
    'MaritalStatus',
    'Passport',
    'Occupation',   
    'OwnCar', 
    'Designation'    
] 

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
# Fill in suitable values for each parameter based on your understanding of XGBoost tuning.
param_grid = {
    'xgbclassifier__n_estimators': [50,75,100,125,150],        # Number of boosting trees. More trees can improve performance but increase training time.
    'xgbclassifier__max_depth': [2,3,4,5],           # Maximum depth of each tree. Higher values increase model complexity and risk of overfitting.
    'xgbclassifier__colsample_bytree': [0.4,0.6,0.8],    # Fraction of features sampled when building each tree.
    'xgbclassifier__colsample_bylevel': [0.4,0.6,0.8],   # Fraction of features sampled at each tree level.
    'xgbclassifier__learning_rate': [0.01,0.1,0.2],       # Step size used during boosting. Smaller values may improve generalization but require more trees.
    'xgbclassifier__reg_lambda': [0.4,0.6,0.8],          # L2 regularization strength. Higher values help reduce overfitting.
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)   # complete the code to build the model pipeline by chaining preprocessor and xgb_model

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search as a nested run,
    # so all experiments can be compared side by side in the MLflow UI
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Set classification threshold
    classification_threshold = 0.45    # Choose a classification threshold between 0 and 1. Lower thresholds typically increase recall and decrease precision and vice versa. Experiment with different values to find the best trade-off.

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally

    model_path = "tourism_package_prediction_model.joblib"   # Specify the local file path (inside tourism_project/deployment/) where the trained model should be saved.
    joblib.dump(best_model, model_path)  # complete the code to save the model

        # and log it as an MLflow artifact for traceability
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved at {model_path}")

    # Upload to Hugging Face
    repo_id = "katisoletsie/TourismPackagePrediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="tourism_package_prediction_model.joblib",
        path_in_repo="tourism_package_prediction_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )    
