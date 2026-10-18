import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Point to the same tracking database the UI uses.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Load the processed data and rebuild the same train/test split.
data = pd.read_csv("data/processed_wine.csv")
X = data.drop(columns=["quality", "high_quality"])
y = data["high_quality"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. The name under which the model will be registered.
model_name = "wine_quality"

# 3. Train the chosen model (XGBoost) and register it in one run.
xgb_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}

with mlflow.start_run(run_name="xgboost_registered"):
    model = XGBClassifier(**xgb_params, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate so the registered run carries its metrics too.
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_params(xgb_params)
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)

    # Log the model AND register it in the Model Registry in one call,
    # by passing registered_model_name. This creates (or versions) the
    # registered model called "wine_quality".
    mlflow.xgboost.log_model(
        model,
        name="model",
        registered_model_name=model_name,
    )

    print(f"Registered '{model_name}'. Metrics: AUC={auc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}")