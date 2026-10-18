import pandas as pd
import mlflow
from mlflow import MlflowClient

# Point to the same tracking database the UI and registration used.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_name = "wine_quality"

# 1. Add a "production" alias to version 1 of the registered model.
#    This is the modern replacement for the deprecated "Production" stage.
client = MlflowClient()
client.set_registered_model_alias(name=model_name, alias="production", version=1)
print(f"Alias 'production' now points to version 1 of '{model_name}'.")

# 2. Load the model back by its alias, not by a hardcoded version number.
#    This is how production code retrieves "whatever is currently production".
model_uri = f"models:/{model_name}@production"
model = mlflow.pyfunc.load_model(model_uri)
print(f"Loaded model from: {model_uri}")

# 3. Prepare a batch of data to predict on.
#    We reuse the processed data and drop the target columns, exactly as in training,
#    then take a small batch of rows to simulate new incoming data.
data = pd.read_csv("data/processed_wine.csv")
features = data.drop(columns=["quality", "high_quality"])
batch = features.head(10)

# 4. Run batch inference: predict on all 10 rows at once.
predictions = model.predict(batch)

# 5. Show the results alongside the true labels so we can eyeball them.
results = batch.copy()
results["predicted_high_quality"] = predictions
results["actual_high_quality"] = data["high_quality"].head(10).values
print("\nBatch inference results (first 10 rows):")
print(results[["predicted_high_quality", "actual_high_quality"]])