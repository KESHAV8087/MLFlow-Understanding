import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import mlflow.xgboost
# 1. Load the processed data we created in the preprocessing step.
data = pd.read_csv("data/processed_wine.csv")

# 2. Separate features (X) from target (y).
#    Drop both 'quality' and 'high_quality' from the inputs:
#    'high_quality' is the target, and 'quality' would leak the answer.
X = data.drop(columns=["quality", "high_quality"])
y = data["high_quality"]

# 3. Split into training and test sets, holding back 20% for honest evaluation.
#    'stratify=y' keeps the same proportion of high-quality wines in both sets,
#    which matters because the classes are imbalanced.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. A helper function that evaluates a model and logs everything to MLflow.
def train_and_log(model, model_name, params):
    with mlflow.start_run(run_name=model_name):
        # Train the model.
        model.fit(X_train, y_train)

        # Predict class labels (0 or 1) and class probabilities.
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        # Compute three metrics suited to imbalanced classification.
        auc = roc_auc_score(y_test, proba)
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log the hyperparameters we chose.
        mlflow.log_params(params)

        # Log the three metrics.
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        # Log the trained model itself.
                # Use the logger that matches the model type.
        if isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

        print(f"{model_name}: AUC={auc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}")

# 5. Baseline model: Random Forest (follows the original lab).
rf_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
rf_model = RandomForestClassifier(**rf_params)
train_and_log(rf_model, "random_forest", rf_params)

# 6. Second model: XGBoost (your modification).
xgb_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}
xgb_model = XGBClassifier(**xgb_params, eval_metric="logloss")
train_and_log(xgb_model, "xgboost", xgb_params)

print("\nBoth runs complete. Launch the MLflow UI to compare them.")