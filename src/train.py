import os
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_config, load_data, validate_data


def preprocess_data(data: pd.DataFrame, target_column: str):
    """
    Limpia el dataset y crea la variable objetivo binaria.

    target = 1 si quality >= 6
    target = 0 si quality < 6
    """
    data = data.copy()

    data = data.drop_duplicates()
    data = data.dropna()

    data["target"] = (data[target_column] >= 6).astype(int)

    X = data.drop(columns=[target_column, "target"])
    y = data["target"]

    return X, y


def train_model(X_train, y_train, config: dict):
    """Entrena un modelo Random Forest dentro de un pipeline."""
    model_config = config["model"]

    model = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        random_state=model_config["random_state"],
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test) -> dict:
    """EvalÃºa el modelo entrenado."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    return metrics


def main():
    config = load_config()

    data_path = config["data"]["path"]
    separator = config["data"]["separator"]
    target_column = config["data"]["target_column"]

    data = load_data(data_path, separator)
    validate_data(data, target_column)

    X, y = preprocess_data(data, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    os.makedirs(config["outputs"]["model_dir"], exist_ok=True)

    with mlflow.start_run(run_name="random_forest_wine_quality"):
        model = train_model(X_train, y_train, config)
        metrics = evaluate_model(model, X_test, y_test)

        input_example = X_test.head(5)
        signature = infer_signature(X_test, model.predict(X_test))

        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_param("max_depth", config["model"]["max_depth"])
        mlflow.log_param("test_size", config["training"]["test_size"])

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=config["outputs"]["model_name"],
            signature=signature,
            input_example=input_example,
        )

        local_model_path = os.path.join(
            config["outputs"]["model_dir"],
            config["outputs"]["model_name"],
        )

        mlflow.sklearn.save_model(
            sk_model=model,
            path=local_model_path,
            signature=signature,
            input_example=input_example,
        )

        print("Entrenamiento finalizado correctamente.")
        print("MÃ©tricas del modelo:")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
