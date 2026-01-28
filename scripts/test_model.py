"""Prueba de modelo de prediccion de churn con datos de ejemplo."""

import argparse
import sys
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Prueba un modelo de prediccion de churn."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/model.pkl",
        help="Ruta al modelo (default: artifacts/model.pkl)",
    )
    return parser.parse_args()


def create_sample_records() -> pd.DataFrame:
    """Crea registros de ejemplo para prueba del modelo."""
    records = [
        {
            "tenure_months": 5,
            "monthly_charges": 90.0,
            "total_charges": 450.0,
            "contract_type": "month-to-month",
            "internet_service": "fiber_optic",
            "payment_method": "electronic_check",
            "num_support_tickets": 7,
        },
        {
            "tenure_months": 48,
            "monthly_charges": 55.0,
            "total_charges": 2640.0,
            "contract_type": "two-year",
            "internet_service": "dsl",
            "payment_method": "bank_transfer",
            "num_support_tickets": 1,
        },
        {
            "tenure_months": 12,
            "monthly_charges": 75.0,
            "total_charges": 900.0,
            "contract_type": "one-year",
            "internet_service": "fiber_optic",
            "payment_method": "credit_card",
            "num_support_tickets": 3,
        },
        {
            "tenure_months": 2,
            "monthly_charges": 110.0,
            "total_charges": 220.0,
            "contract_type": "month-to-month",
            "internet_service": "fiber_optic",
            "payment_method": "mailed_check",
            "num_support_tickets": 9,
        },
        {
            "tenure_months": 60,
            "monthly_charges": 30.0,
            "total_charges": 1800.0,
            "contract_type": "two-year",
            "internet_service": "no",
            "payment_method": "bank_transfer",
            "num_support_tickets": 0,
        },
    ]
    return pd.DataFrame(records)


def encode_features(data: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categoricas con get_dummies."""
    encoded: pd.DataFrame = pd.get_dummies(
        data,
        columns=[
            "contract_type",
            "internet_service",
            "payment_method",
        ],
        dtype=int,
    )
    return encoded


def align_features(
    data: pd.DataFrame,
    model_features: List[str],
) -> pd.DataFrame:
    """Alinea las columnas con las features del modelo."""
    for col in model_features:
        if col not in data.columns:
            data[col] = 0
    data = data[model_features]
    return data


def main() -> None:
    """Funcion principal de prueba del modelo."""
    args = parse_args()

    print("=" * 60)
    print("  PRUEBA DE MODELO DE PREDICCION DE CHURN")
    print("=" * 60)

    print(f"\nCargando modelo desde: {args.model_path}")
    model: RandomForestClassifier = joblib.load(
        args.model_path
    )
    print("  Modelo cargado exitosamente.")

    model_features: List[str] = list(
        model.feature_names_in_
    )
    print(
        f"  Tipo de modelo: "
        f"{type(model).__name__}"
    )
    print(f"  Numero de features: {len(model_features)}")

    print("\nCreando registros de ejemplo...")
    sample_data = create_sample_records()
    print(f"  Registros creados: {len(sample_data)}")

    encoded_data = encode_features(sample_data)
    aligned_data = align_features(
        encoded_data, model_features
    )

    print("\nEjecutando predicciones...")
    probabilities: np.ndarray = model.predict_proba(
        aligned_data
    )
    predictions: np.ndarray = model.predict(aligned_data)

    print("\n" + "-" * 60)
    print("  RESULTADOS DE PREDICCION")
    print("-" * 60)
    header = (
        f"{'#':<4}"
        f"{'Contrato':<18}"
        f"{'Internet':<14}"
        f"{'P(Churn)':<12}"
        f"{'Prediccion':<12}"
    )
    print(header)
    print("-" * 60)

    for idx in range(len(sample_data)):
        row = sample_data.iloc[idx]
        prob_churn = probabilities[idx][1]
        pred_label = (
            "CHURN" if predictions[idx] == 1 else "NO CHURN"
        )
        line = (
            f"{idx + 1:<4}"
            f"{row['contract_type']:<18}"
            f"{row['internet_service']:<14}"
            f"{prob_churn:<12.4f}"
            f"{pred_label:<12}"
        )
        print(line)

    print("-" * 60)
    print("\nPrueba completada exitosamente.")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as exc:
        print(f"Error durante la prueba: {exc}")
        sys.exit(1)
