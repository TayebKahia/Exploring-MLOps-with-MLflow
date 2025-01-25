model_name = "XGB-Smote"
run_id = "64c9651ae0de42ba898917002ffb2ccb/model"
model_uri = f"runs:/{run_id}/model_name"

print("MLflow is running")

with mlflow.start_run(run_id=run_id):
    mlflow.register_model(model_uri=model_uri, name=model_name)
