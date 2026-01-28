"""Subir datos a Azure ML como Data Asset."""

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


def upload_data(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    file_path: str,
    asset_name: str,
) -> bool:
    """Sube un archivo como Data Asset a Azure ML."""
    try:
        # pylint: disable=import-outside-toplevel
        from azure.ai.ml import MLClient
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Data
        from azure.identity import DefaultAzureCredential

        # pylint: enable=import-outside-toplevel

        data_file = Path(file_path)
        if not data_file.exists():
            print("Error: No se encontro el archivo: " f"{file_path}")
            return False

        print("Autenticando con Azure...")
        credential = DefaultAzureCredential()

        print("Conectando al workspace de Azure ML...")
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

        print(f"Subiendo archivo: {file_path}")
        data_asset = Data(
            name=asset_name,
            path=file_path,
            type=AssetTypes.URI_FILE,
            description=("Data asset subido desde " f"{data_file.name}"),
        )

        created_asset = ml_client.data.create_or_update(data_asset)

        print("\n" + "=" * 60)
        print("DATOS SUBIDOS EXITOSAMENTE")
        print("=" * 60)
        print(f"  Nombre: {created_asset.name}")
        print(f"  Version: {created_asset.version}")
        print(f"  URI: {created_asset.path}")
        print("=" * 60)
        return True

    except ImportError:
        print("Error: Paquetes de Azure no instalados.")
        print("Ejecuta: pip install azure-ai-ml azure-identity")
        return False
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\nError al subir los datos: {exc}")
        return False


def main() -> None:
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description="Subir datos a Azure ML")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Ruta al archivo de configuracion",
    )
    parser.add_argument(
        "--file-path",
        default="data/inference_input.csv",
        help="Ruta al archivo de datos",
    )
    parser.add_argument(
        "--asset-name",
        default="churn-inference-data",
        help="Nombre del Data Asset",
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

    success = upload_data(
        sub_id,
        rg_name,
        ws_name,
        args.file_path,
        args.asset_name,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
