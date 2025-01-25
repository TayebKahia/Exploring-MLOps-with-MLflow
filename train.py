import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Parse parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_estimators", type=int, default=100, help="Number of trees in the forest"
)
parser.add_argument(
    "--max_depth", type=int, default=5, help="Maximum depth of the tree"
)
args = parser.parse_args()

# Load the California Housing dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Train the model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("mse", mse)

    # Log the trained model
    mlflow.sklearn.log_model(model, "california_prediction")

    # Print metrics for verification
    print(f"Mean Squared Error: {mse}")

# Instructions to view the MLflow UI
print("Run 'mlflow ui' to view the experiment logs at http://localhost:5000.")
