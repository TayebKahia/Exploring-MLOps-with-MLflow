import mlflow
import pandas as pd

logged_model = "runs:/64c9651ae0de42ba898917002ffb2ccb/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = {
    "MedInc": [8.3252],
    "HouseAge": [41],
    "AveRooms": [6.9841],
    "AveBedrms": [1.0238],
    "Population": [322],
    "AveOccup": [2.5556],
    "Latitude": [37.88],
    "Longitude": [-122.23],
}

input_df = pd.DataFrame(data)
predictions = loaded_model.predict(input_df)

print("Predictions:", predictions)
