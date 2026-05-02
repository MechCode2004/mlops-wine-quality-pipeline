import yaml
import pandas as pd


def load_config(config_path: str = "config.yaml") -> dict:
    """Carga el archivo de configuraciÃ³n YAML."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_data(data_path: str, separator: str = ";") -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV."""
    data = pd.read_csv(data_path, sep=separator)
    return data


def validate_data(data: pd.DataFrame, target_column: str) -> None:
    """Valida que el dataset tenga datos y la columna objetivo."""
    if data.empty:
        raise ValueError("El dataset estÃ¡ vacÃ­o.")

    if target_column not in data.columns:
        error_message = (
            f"La columna objetivo '{target_column}' no existe en el dataset."
        )
        raise ValueError(error_message)

    if data[target_column].isnull().any():
        raise ValueError("La columna objetivo contiene valores nulos.")
