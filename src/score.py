"""Script de scoring para prediccion de churn en Azure ML."""

import argparse
import sys
from pathlib import Path
from typing import List

import mlflow.sklearn
import pandas as pd

CATEGORICAL_COLUMNS: List[str] = [
    "contract_type",
    "internet_service",
    "payment_method",
]


def score(
    input_path: str,
    model_name: str,
    output_path: str,
) -> bool:
    """Ejecuta inferencia usando el modelo registrado."""
    try:
        print(f"Cargando modelo '{model_name}'...")
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        print("Modelo cargado exitosamente.")

        print(f"Leyendo datos de entrada: {input_path}")
        input_df = pd.read_csv(input_path)
        print(f"  Registros leidos: {len(input_df)}")

        original_df = input_df.copy()

        print("Aplicando preprocesamiento...")
        features_df = pd.get_dummies(
            input_df,
            columns=CATEGORICAL_COLUMNS,
        )
        print(f"  Features despues de encoding: " f"{len(features_df.columns)}")

        print("Ejecutando prediccion...")
        probabilities = model.predict_proba(features_df)[:, 1]
        original_df["churn_probability"] = probabilities
        print(f"  Predicciones generadas: " f"{len(probabilities)}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        original_df.to_csv(output_file, index=False)

        print("\n" + "=" * 60)
        print("INFERENCIA COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        print(f"  Registros procesados: {len(original_df)}")
        print(f"  Archivo de salida: {output_path}")
        print("=" * 60)
        return True

    except Exception as exc:  # pylint: disable=broad-except
        print(f"\nError durante la inferencia: {exc}")
        return False


def main() -> None:
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description="Scoring de prediccion de churn")
    parser.add_argument(
        "--input-path",
        required=True,
        help="Ruta al archivo CSV de entrada",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Nombre del modelo registrado en MLflow",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Ruta al archivo CSV de salida",
    )
    args = parser.parse_args()

    success = score(args.input_path, args.model_name, args.output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
