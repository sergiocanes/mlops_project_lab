"""Ejecutar pipeline de inferencia en Azure ML."""

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


def ensure_compute(ml_client: Any, compute_name: str) -> None:
    """Crea o verifica el cluster de computo."""
    # pylint: disable=import-outside-toplevel
    from azure.ai.ml.entities import AmlCompute

    # pylint: enable=import-outside-toplevel

    try:
        ml_client.compute.get(compute_name)
        print("Cluster de computo " f"'{compute_name}' encontrado.")
    except Exception:  # pylint: disable=broad-except
        print("Creando cluster de computo " f"'{compute_name}'...")
        compute = AmlCompute(
            name=compute_name,
            size="Standard_DS2_v2",
            min_instances=0,
            max_instances=1,
        )
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"Cluster '{compute_name}' creado exitosamente.")


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
def run_pipeline(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    model_name: str,
    data_asset_name: str,
    compute_name: str,
) -> bool:
    """Ejecuta el pipeline de inferencia en Azure ML."""
    try:
        # pylint: disable=import-outside-toplevel
        from azure.ai.ml import Input, MLClient, command
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Environment
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

        ensure_compute(ml_client, compute_name)

        print("Creando entorno desde conda.yml...")
        env = Environment(
            name="churn-inference-env",
            conda_file="conda.yml",
            image=("mcr.microsoft.com/azureml/" "openmpi4.1.0-ubuntu20.04:latest"),
        )

        print("Obteniendo ultima version del data asset " f"'{data_asset_name}'...")
        data_asset = ml_client.data.get(name=data_asset_name, label="latest")
        print(f"  Data asset version: {data_asset.version}")

        score_command = (
            "python score.py "
            "--input-path ${{inputs.input_data}} "
            f"--model-name {model_name} "
            "--output-path ${{outputs.predictions}}"
        )

        data_path = f"azureml:{data_asset.name}" f":{data_asset.version}"

        print("Creando job de inferencia...")
        inference_job = command(
            code="./src",
            command=score_command,
            inputs={
                "input_data": Input(
                    type=AssetTypes.URI_FILE,
                    path=data_path,
                ),
            },
            outputs={
                "predictions": {
                    "type": "uri_file",
                },
            },
            environment=env,
            compute=compute_name,
            display_name="churn-inference-job",
            description=("Job de inferencia para " "prediccion de churn"),
        )

        print("Enviando job a Azure ML...")
        returned_job = ml_client.jobs.create_or_update(inference_job)
        job_name = returned_job.name

        print(f"Job enviado: {job_name}")
        print("Esperando finalizacion del job...")
        ml_client.jobs.stream(job_name)

        completed_job = ml_client.jobs.get(job_name)

        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "inference_output.csv"

        print("Descargando resultados...")
        ml_client.jobs.download(
            job_name,
            output_name="predictions",
            download_path=str(output_dir),
        )

        print("\n" + "=" * 60)
        print("PIPELINE EJECUTADO EXITOSAMENTE")
        print("=" * 60)
        print(f"  Job URL: {completed_job.studio_url}")
        print(f"  Estado: {completed_job.status}")
        print(f"  Resultados: {output_path}")
        print("=" * 60)
        return True

    except ImportError:
        print("Error: Paquetes de Azure no instalados.")
        print("Ejecuta: pip install azure-ai-ml azure-identity")
        return False
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\nError al ejecutar el pipeline: {exc}")
        return False


def main() -> None:
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description=("Ejecutar pipeline de inferencia en Azure ML")
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Ruta al archivo de configuracion",
    )
    parser.add_argument(
        "--model-name",
        default="churn-prediction-model",
        help="Nombre del modelo registrado",
    )
    parser.add_argument(
        "--data-asset-name",
        default="churn-inference-data",
        help="Nombre del Data Asset de entrada",
    )
    parser.add_argument(
        "--compute-name",
        default="cpu-cluster",
        help="Nombre del cluster de computo",
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

    success = run_pipeline(
        sub_id,
        rg_name,
        ws_name,
        args.model_name,
        args.data_asset_name,
        args.compute_name,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
