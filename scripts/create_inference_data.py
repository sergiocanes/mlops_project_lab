"""Generacion de datos de inferencia para prediccion de churn."""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Genera datos de inferencia "
            "para prediccion de churn."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Semilla aleatoria (default: 123)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Numero de muestras (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directorio de salida (default: data)",
    )
    return parser.parse_args()


def generate_inference_data(
    n_samples: int, seed: int
) -> pd.DataFrame:
    """Genera datos de inferencia sin columna de churn."""
    rng = np.random.default_rng(seed)

    customer_ids: List[str] = [
        f"CUST-{i:05d}"
        for i in range(10001, 10001 + n_samples)
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
        }
    )
    return data


def main() -> None:
    """Funcion principal de generacion de datos."""
    args = parse_args()

    print("=" * 60)
    print("  GENERACION DE DATOS DE INFERENCIA")
    print("=" * 60)

    print(
        f"\nGenerando {args.n_samples} registros "
        f"(semilla={args.seed})..."
    )
    data = generate_inference_data(args.n_samples, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, "inference_input.csv"
    )
    data.to_csv(output_path, index=False)

    print(f"  Registros generados: {len(data)}")
    print(f"  Archivo guardado en: {output_path}")

    print("\nPrimeras 5 filas de ejemplo:")
    print(data.head().to_string(index=False))

    print("\nGeneracion de datos completada exitosamente.")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as exc:
        print(
            f"Error durante la generacion de datos: {exc}"
        )
        sys.exit(1)
