"""Verificar conexion con Azure ML workspace."""

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


def verify_connection(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
) -> bool:
    """Verifica la conexion al workspace de Azure ML."""
    try:
        # pylint: disable=import-outside-toplevel
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

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

        workspace = ml_client.workspaces.get(workspace_name)
        print("\n" + "=" * 60)
        print("CONEXION EXITOSA")
        print("=" * 60)
        print(f"  Workspace: {workspace.name}")
        print(f"  Location:  {workspace.location}")
        print(f"  Resource Group: {resource_group}")
        print(f"  Subscription:   {subscription_id[:8]}...")
        print("=" * 60)
        return True

    except ImportError:
        print("Error: Paquetes de Azure no instalados.")
        print("Ejecuta: pip install azure-ai-ml azure-identity")
        return False
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\nError de conexion: {exc}")
        print("\nVerifica que:")
        print("  1. Ejecutaste 'az login' en tu terminal")
        print("  2. Los datos en config.json son correctos")
        print("  3. Tienes permisos en el workspace")
        return False


def main() -> None:
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Verificar conexion con Azure ML workspace"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Ruta al archivo de configuracion",
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

    success = verify_connection(sub_id, rg_name, ws_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
