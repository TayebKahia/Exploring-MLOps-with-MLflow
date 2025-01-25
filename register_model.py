import mlflow

run_id = "64c9651ae0de42ba898917002ffb2ccb"
# model_name = "CaliforniaHousingModel"
# model_uri = f"runs:/{run_id}/model_name"

# with mlflow.start_run(run_id=run_id):
#     mlflow.register_model(model_uri=model_uri, name=model_name)


from mlflow.tracking import MlflowClient

model_uri = f"runs:/{run_id}/model"
model_name = "CaliforniaHousingModel"

client = MlflowClient()
# Transition the model to the "Staging" stage
client.transition_model_version_stage(
    name=model_name,
    version=2,  # Replace with your registered model's version
    stage="Staging",
)
print(f"Model {model_name} version 2 transitioned to 'Staging'.")
