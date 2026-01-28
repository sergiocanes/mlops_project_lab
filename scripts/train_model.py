"""Entrenamiento de modelo de prediccion de churn con datos sinteticos."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de prediccion de churn."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Numero de muestras (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directorio de salida (default: artifacts)",
    )
    return parser.parse_args()


def generate_synthetic_data(
    n_samples: int, seed: int
) -> pd.DataFrame:
    """Genera datos sinteticos de churn de clientes."""
    rng = np.random.default_rng(seed)

    customer_ids: List[str] = [
        f"CUST-{i:05d}" for i in range(1, n_samples + 1)
    ]
    tenure_months: np.ndarray = rng.integers(
        1, 73, size=n_samples
    )
    monthly_charges: np.ndarray = rng.uniform(
        20.0, 120.0, size=n_samples
    )
    noise: np.ndarray = rng.normal(0, 50, size=n_samples)
    total_charges: np.ndarray = (
        tenure_months * monthly_charges + noise
    )
    total_charges = np.maximum(total_charges, 0.0)

    contract_options = ["month-to-month", "one-year", "two-year"]
    contract_type: np.ndarray = rng.choice(
        contract_options, size=n_samples
    )

    internet_options = ["dsl", "fiber_optic", "no"]
    internet_service: np.ndarray = rng.choice(
        internet_options, size=n_samples
    )

    payment_options = [
        "electronic_check",
        "mailed_check",
        "bank_transfer",
        "credit_card",
    ]
    payment_method: np.ndarray = rng.choice(
        payment_options, size=n_samples
    )

    num_support_tickets: np.ndarray = rng.integers(
        0, 11, size=n_samples
    )

    # Logica realista de churn
    churn_prob = np.full(n_samples, 0.15)
    churn_prob = np.where(
        contract_type == "month-to-month", 0.40, churn_prob
    )
    churn_prob = np.where(
        payment_method == "electronic_check",
        np.maximum(churn_prob, 0.35),
        churn_prob,
    )
    high_tickets = num_support_tickets > 5
    fiber = internet_service == "fiber_optic"
    churn_prob = np.where(
        fiber & high_tickets,
        np.minimum(churn_prob + 0.20, 1.0),
        churn_prob,
    )
    long_tenure = tenure_months > 36
    churn_prob = np.where(
        long_tenure, churn_prob * 0.5, churn_prob
    )

    churned: np.ndarray = rng.binomial(1, churn_prob)

    data = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "num_support_tickets": num_support_tickets,
            "churned": churned,
        }
    )
    return data


def prepare_features(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara features y target para entrenamiento."""
    target: pd.Series = data["churned"]
    features_df: pd.DataFrame = data.drop(
        columns=["customer_id", "churned"]
    )
    features_encoded: pd.DataFrame = pd.get_dummies(
        features_df,
        columns=[
            "contract_type",
            "internet_service",
            "payment_method",
        ],
        dtype=int,
    )
    return features_encoded, target


def train_and_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Entrena el modelo y calcula metricas."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=seed,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
    )
    model.fit(x_train, y_train)

    y_pred: np.ndarray = model.predict(x_test)

    metrics: Dict[str, float] = {
        "accuracy": round(
            float(accuracy_score(y_test, y_pred)), 4
        ),
        "precision": round(
            float(
                precision_score(y_test, y_pred, zero_division=0)
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(y_test, y_pred, zero_division=0)
            ),
            4,
        ),
        "f1_score": round(
            float(f1_score(y_test, y_pred, zero_division=0)), 4
        ),
    }
    return model, metrics


def save_artifacts(
    model: RandomForestClassifier,
    metadata: Dict[str, Any],
    output_dir: str,
) -> None:
    """Guarda el modelo y metadatos en disco."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"  Modelo guardado en: {model_path}")

    metadata_path = os.path.join(
        output_dir, "model_metadata.json"
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadatos guardados en: {metadata_path}")


def main() -> None:
    """Funcion principal de entrenamiento."""
    args = parse_args()

    print("=" * 60)
    print("  ENTRENAMIENTO DE MODELO DE PREDICCION DE CHURN")
    print("=" * 60)

    print(
        f"\nGenerando {args.n_samples} muestras sinteticas "
        f"(semilla={args.seed})..."
    )
    data = generate_synthetic_data(args.n_samples, args.seed)
    print(f"  Datos generados: {data.shape[0]} filas")
    print(
        f"  Tasa de churn: "
        f"{data['churned'].mean():.2%}"
    )

    print("\nPreparando features...")
    features, target = prepare_features(data)
    feature_list: List[str] = list(features.columns)
    print(f"  Numero de features: {len(feature_list)}")

    print("\nEntrenando modelo RandomForest...")
    model, metrics = train_and_evaluate(
        features, target, args.seed
    )

    print("\nMetricas de evaluacion:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")

    metadata: Dict[str, Any] = {
        "model_name": "churn_prediction_rf",
        "model_type": "RandomForestClassifier",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "num_samples": args.n_samples,
        "features": feature_list,
        "metrics": metrics,
        "random_seed": args.seed,
    }

    print(f"\nGuardando artefactos en '{args.output_dir}'...")
    save_artifacts(model, metadata, args.output_dir)

    print("\nEntrenamiento completado exitosamente.")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as exc:
        print(f"Error durante el entrenamiento: {exc}")
        sys.exit(1)
