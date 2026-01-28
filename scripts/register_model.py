"""Registrar modelo en Azure ML usando MLflow."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Carga configuracion desde archivo JSON."""
    path = Path(config_path)
    if not path.exists():
        print("Error: No se encontro el archivo " f"de configuracion: {config_path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)
    return config


def register_model(  # pylint: disable=too-many-locals
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    model_path: str,
    model_name: str,
) -> bool:
    """Registra el modelo en Azure ML con MLflow."""
    try:
        # pylint: disable=import-outside-toplevel
        import joblib
        import mlflow
        import mlflow.sklearn
        import numpy as np
        import pandas as pd
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        from mlflow.models import infer_signature

        # pylint: enable=import-outside-toplevel

        print("Autenticando con Azure...")
        credential = DefaultAzureCredential()

        print("Conectando al workspace de Azure ML...")
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

        print("Configurando MLflow tracking URI...")
        mlflow.set_tracking_uri(ml_client.tracking_uri)

        model_file = Path(model_path)
        if not model_file.exists():
            print("Error: No se encontro el modelo " f"en: {model_path}")
            return False

        metadata_path = model_file.parent / "model_metadata.json"
        if not metadata_path.exists():
            print("Error: No se encontro " f"model_metadata.json en: {metadata_path}")
            return False

        with open(metadata_path, encoding="utf-8") as f:
            metadata: Dict[str, Any] = json.load(f)

        print(f"Cargando modelo desde: {model_path}")
        model = joblib.load(model_file)

        print("Creando datos de ejemplo para la firma...")
        num_features = 5
        sample_data = pd.DataFrame(
            np.random.rand(num_features, num_features),
            columns=[f"feature_{i}" for i in range(num_features)],
        )
        signature = infer_signature(sample_data, model.predict(sample_data))

        print("Registrando modelo con MLflow...")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "model_type": metadata.get("model_type", "unknown"),
                    "random_seed": metadata.get("random_seed", 0),
                    "num_samples": metadata.get("num_samples", 0),
                }
            )

            mlflow.log_metrics(
                {
                    "accuracy": float(metadata.get("accuracy", 0.0)),
                    "precision": float(metadata.get("precision", 0.0)),
                    "recall": float(metadata.get("recall", 0.0)),
                    "f1_score": float(metadata.get("f1_score", 0.0)),
                }
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name,
            )

        print("\n" + "=" * 60)
        print("MODELO REGISTRADO EXITOSAMENTE")
        print("=" * 60)
        print(f"  Modelo URI: {model_info.model_uri}")
        print(f"  Nombre: {model_name}")
        print("=" * 60)
        return True

    except ImportError:
        print("Error: Paquetes requeridos no instalados.")
        print(
            "Ejecuta: pip install azure-ai-ml "
            "azure-identity mlflow joblib numpy "
            "pandas scikit-learn"
        )
        return False
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\nError al registrar el modelo: {exc}")
        return False


def main() -> None:
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description="Registrar modelo en Azure ML")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Ruta al archivo de configuracion",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/model.pkl",
        help="Ruta al archivo del modelo",
    )
    parser.add_argument(
        "--model-name",
        default="churn-prediction-model",
        help="Nombre para registrar el modelo",
    )
    parser.add_argument(
        "--subscription-id",
        help="Azure Subscription ID",
    )
    parser.add_argument(
        "--resource-group",
        help="Azure Resource Group",
    )
    parser.add_argument(
        "--workspace-name",
        help="Azure ML Workspace Name",
    )
    args = parser.parse_args()

    if args.subscription_id and args.resource_group and args.workspace_name:
        sub_id = args.subscription_id
        rg_name = args.resource_group
        ws_name = args.workspace_name
    else:
        config = load_config(args.config)
        sub_id = str(config.get("subscription_id", ""))
        rg_name = str(config.get("resource_group", ""))
        ws_name = str(config.get("workspace_name", ""))

    if not all([sub_id, rg_name, ws_name]):
        print("Error: Faltan parametros de configuracion.")
        print(
            "Usa --config config.json o "
            "--subscription-id/--resource-group/"
            "--workspace-name"
        )
        sys.exit(1)

    success = register_model(
        sub_id,
        rg_name,
        ws_name,
        args.model_path,
        args.model_name,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
