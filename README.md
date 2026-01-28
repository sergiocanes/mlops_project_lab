# Lab: GitOps Workflow para Azure ML

Tutorial completo para implementar un flujo GitOps con CI/CD usando GitHub Actions, entrenar un modelo de Machine Learning localmente, registrarlo en Azure ML con MLflow, y ejecutar un pipeline de inferencia en la nube.

---

## Tabla de Contenido

- [Prerequisitos](#prerequisitos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Parte 1: Crear Repositorio](#parte-1-crear-repositorio)
- [Parte 2: GitOps Workflow (CI/CD)](#parte-2-gitops-workflow-cicd)
- [Parte 3: Entrenar Modelo Local](#parte-3-entrenar-modelo-local)
- [Parte 4: Configurar Azure ML](#parte-4-configurar-azure-ml)
- [Parte 5: Registrar Modelo en Azure ML](#parte-5-registrar-modelo-en-azure-ml)
- [Parte 6: Pipeline de Inferencia](#parte-6-pipeline-de-inferencia)
- [Bonus: Automatizacion con GitHub Actions](#bonus-automatizacion-con-github-actions)
- [Apendice A: Crear Workspace de Azure ML](#apendice-a-crear-workspace-de-azure-ml)
- [Apendice B: Troubleshooting](#apendice-b-troubleshooting)

---

## Prerequisitos

Antes de comenzar, asegurate de tener instalado lo siguiente:

| Herramienta | Version Minima | Verificacion |
|-------------|---------------|--------------|
| Python | 3.10+ | `python --version` |
| Git | 2.0+ | `git --version` |
| GitHub Account | - | Accede a [github.com](https://github.com) |
| Azure CLI | 2.50+ | `az --version` |
| pip | 21.0+ | `pip --version` |

### Verificar instalacion

Ejecuta estos comandos en tu terminal para confirmar que todo esta listo:

```bash
python --version
# Esperado: Python 3.10.x o superior

git --version
# Esperado: git version 2.x.x

az --version
# Esperado: azure-cli 2.50.0 o superior

pip --version
# Esperado: pip 21.x o superior
```

> **Nota**: Si no tienes Python 3.10+, descargalo desde [python.org](https://www.python.org/downloads/). Si no tienes Azure CLI, consulta las instrucciones de instalacion en [la documentacion oficial](https://learn.microsoft.com/es-es/cli/azure/install-azure-cli).

---

## Estructura del Proyecto

```
labs/gitops-workflow/
├── .github/
│   └── workflows/
│       ├── ci.yml                         # Workflow de Integracion Continua
│       ├── cd.yml                         # Workflow de Entrega Continua
│       ├── deploy_model.yml               # Workflow bonus de despliegue
│       ├── reusable_lint.yml              # Validacion de codigo Python
│       ├── reusable_semantic_version.yml  # Versionado semantico
│       ├── reusable_create_tag.yml        # Creacion de tags Git
│       ├── reusable_create_release.yml    # Creacion de releases
│       └── reusable_pr_comment.yml        # Comentarios en PRs
├── scripts/
│   ├── __init__.py
│   ├── train_model.py                     # Entrenamiento del modelo
│   ├── test_model.py                      # Prueba del modelo
│   ├── create_inference_data.py           # Generacion de datos de inferencia
│   ├── verify_azure_connection.py         # Verificar conexion Azure
│   ├── register_model.py                  # Registrar modelo en Azure ML
│   ├── upload_data.py                     # Subir datos a Azure ML
│   └── run_inference_pipeline.py          # Ejecutar pipeline de inferencia
├── src/
│   └── score.py                           # Script de scoring (Azure ML)
├── artifacts/                             # Modelos entrenados (generado)
│   └── .gitkeep
├── data/                                  # Datos CSV (generado)
│   └── .gitkeep
├── conda.yml                              # Entorno para Azure ML compute
├── requirements.txt                       # Dependencias Python
├── pyproject.toml                         # Configuracion de linters
├── .gitignore                             # Archivos ignorados
└── README.md                              # Este tutorial
```

---

## Parte 1: Crear Repositorio

En esta seccion vas a crear tu propio repositorio en GitHub con el contenido del lab.

### Paso 1.1: Crear repositorio en GitHub

1. Ve a [github.com/new](https://github.com/new)
2. Nombre del repositorio: `mlops-gitops-lab` (o el nombre que prefieras)
3. Visibilidad: **Public** (necesario para GitHub Actions gratuito)
4. **NO** inicialices con README, .gitignore ni licencia
5. Haz clic en **Create repository**

### Paso 1.2: Clonar y copiar contenido

```bash
# Clonar tu repositorio vacio
git clone https://github.com/TU_USUARIO/mlops-gitops-lab.git
cd mlops-gitops-lab

# Copiar el contenido del lab a tu repositorio
# (ajusta la ruta segun donde tengas el template)
cp -r /ruta/al/template_repo/labs/gitops-workflow/* .
cp -r /ruta/al/template_repo/labs/gitops-workflow/.* . 2>/dev/null
```

> **Importante**: Asegurate de copiar los archivos ocultos (`.github/`, `.gitignore`). El comando con `.*` se encarga de eso.

### Paso 1.3: Primer commit y push

```bash
# Verificar que los archivos estan correctos
ls -la
ls .github/workflows/

# Agregar todos los archivos
git add .

# Crear el primer commit con conventional commit
git commit -m "feat: initial lab setup with CI/CD workflows and ML scripts"

# Subir a GitHub
git push origin main
```

### Checkpoint: Verificar repositorio

Abre tu repositorio en GitHub y verifica que:
- [ ] La carpeta `.github/workflows/` contiene 8 archivos `.yml`
- [ ] La carpeta `scripts/` contiene los archivos Python
- [ ] Existe `requirements.txt` y `pyproject.toml`
- [ ] El commit aparece con el mensaje `feat: initial lab setup...`

---

## Parte 2: GitOps Workflow (CI/CD)

En esta seccion vas a experimentar con el flujo completo de GitOps: crear una rama, hacer cambios, abrir un Pull Request que dispare CI, y al hacer merge, disparar CD para crear una version y release.

### Paso 2.1: Crear rama feature

```bash
# Asegurate de estar en main y actualizado
git checkout main
git pull origin main

# Crear una nueva rama
git checkout -b feature/update-model-params
```

### Paso 2.2: Hacer un cambio en el codigo

Abre el archivo `scripts/train_model.py` y modifica el numero de estimadores del modelo. Busca la linea:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=seed,
)
```

Y cambiala a:

```python
model = RandomForestClassifier(
    n_estimators=200,
    random_state=seed,
)
```

### Paso 2.3: Commit con conventional commits

```bash
# Verificar los cambios
git diff

# Agregar y hacer commit
git add scripts/train_model.py
git commit -m "feat: increase model estimators to 200 for better accuracy"

# Subir la rama
git push origin feature/update-model-params
```

> **Sobre Conventional Commits**: Los mensajes de commit siguen un formato especifico que permite generar versiones automaticamente:
> - `feat:` - Nueva funcionalidad (incrementa version MINOR: 0.1.0 -> 0.2.0)
> - `feat!:` - Cambio que rompe compatibilidad (incrementa MAJOR: 0.1.0 -> 1.0.0)
> - `fix:` - Correccion de bug (incrementa PATCH: 0.1.0 -> 0.1.1)

### Paso 2.4: Crear Pull Request

1. Ve a tu repositorio en GitHub
2. Veras un banner amarillo sugiriendo crear un Pull Request - haz clic en **Compare & pull request**
3. Titulo del PR: `feat: increase model estimators to 200`
4. En la descripcion, escribe una breve explicacion del cambio
5. Haz clic en **Create pull request**

### Paso 2.5: Verificar CI

Una vez creado el PR, el workflow de CI se ejecuta automaticamente:

1. Ve a la pestana **Actions** de tu repositorio
2. Veras un workflow llamado **CI Pipeline** en ejecucion
3. Espera a que termine (1-2 minutos)
4. El CI ejecuta las siguientes validaciones:
   - **Black**: Formato de codigo
   - **Flake8**: Errores de estilo
   - **isort**: Orden de imports
   - **Pylint**: Analisis estatico
   - **MyPy**: Verificacion de tipos

5. Regresa al PR - veras un comentario automatico con los resultados:

```
## CI Pipeline Results

| Check | Status |
|-------|--------|
| Lint  | ✅ Passed |

**Overall**: ✅ Ready to merge
```

> **Si el CI falla**, consulta la seccion [Troubleshooting](#apendice-b-troubleshooting) para resolver errores comunes de lint.

### Paso 2.6: Merge del PR

1. Una vez que el CI pase (check verde), haz clic en **Merge pull request**
2. Selecciona **Create a merge commit**
3. Haz clic en **Confirm merge**

### Paso 2.7: Verificar CD

Despues del merge, el workflow de CD se ejecuta automaticamente:

1. Ve a la pestana **Actions**
2. Veras un workflow llamado **CD Pipeline** en ejecucion
3. Espera a que termine
4. El CD ejecuta en secuencia:
   - **Semantic Version**: Analiza los commits y genera version (ej: `v0.1.0`)
   - **Create Tag**: Crea un tag Git con la version
   - **Create Release**: Publica un release en GitHub con los archivos del proyecto

5. Ve a la seccion **Releases** (en la barra lateral derecha del repositorio)
6. Veras el release creado automaticamente con la version generada

### Checkpoint: Verificar flujo GitOps

- [ ] El CI se ejecuto y paso en el PR
- [ ] Aparecio un comentario automatico en el PR con los resultados
- [ ] El CD se ejecuto despues del merge
- [ ] Existe un tag Git con la version (ej: `v0.1.0`)
- [ ] Existe un release en GitHub con los archivos del proyecto

---

## Parte 3: Entrenar Modelo Local

En esta seccion vas a entrenar un modelo de prediccion de churn (abandono de clientes) usando datos sinteticos generados automaticamente.

### Paso 3.1: Instalar dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 3.2: Entrenar el modelo

```bash
python scripts/train_model.py
```

Salida esperada:

```
============================================================
  ENTRENAMIENTO DE MODELO DE PREDICCION DE CHURN
============================================================

Generando 10000 muestras sinteticas (semilla=42)...
  Datos generados: 10000 filas
  Tasa de churn: ~25%

Preparando features...
  Numero de features: 14

Entrenando modelo RandomForest...

Metricas de evaluacion:
  accuracy: 0.8xxx
  precision: 0.7xxx
  recall: 0.5xxx
  f1_score: 0.6xxx

Guardando artefactos en 'artifacts'...
  Modelo guardado en: artifacts/model.pkl
  Metadatos guardados en: artifacts/model_metadata.json

Entrenamiento completado exitosamente.
```

> **Que hace este script?**
> 1. Genera 10,000 registros sinteticos de clientes con datos como tenure, monthly_charges, contract_type, etc.
> 2. La probabilidad de churn es realista: clientes con contrato mes-a-mes (~40%), pago con cheque electronico (~35%), y alta cantidad de tickets de soporte tienen mayor probabilidad de churn.
> 3. Entrena un RandomForestClassifier con 100 arboles (o 200 si hiciste el cambio de la Parte 2).
> 4. Guarda el modelo como `artifacts/model.pkl` y las metricas como `artifacts/model_metadata.json`.

### Paso 3.3: Probar el modelo

```bash
python scripts/test_model.py
```

Salida esperada:

```
============================================================
  PRUEBA DE MODELO DE PREDICCION DE CHURN
============================================================

Cargando modelo desde: artifacts/model.pkl
  Modelo cargado exitosamente.
  Tipo de modelo: RandomForestClassifier
  Numero de features: 14

Creando registros de ejemplo...
  Registros creados: 5

Ejecutando predicciones...

------------------------------------------------------------
  RESULTADOS DE PREDICCION
------------------------------------------------------------
#   Contrato          Internet      P(Churn)    Prediccion
------------------------------------------------------------
1   month-to-month    fiber_optic   0.8xxx      CHURN
2   two-year          dsl           0.0xxx      NO CHURN
3   one-year          fiber_optic   0.2xxx      NO CHURN
4   month-to-month    fiber_optic   0.9xxx      CHURN
5   two-year          no            0.0xxx      NO CHURN
------------------------------------------------------------

Prueba completada exitosamente.
```

> **Interpretacion**: El modelo asigna alta probabilidad de churn a clientes con contrato mes-a-mes y servicio de fibra optica (registros 1 y 4), y baja probabilidad a clientes con contratos largos (registros 2 y 5). Esto es consistente con la logica de generacion de datos.

### Checkpoint: Verificar entrenamiento

- [ ] El archivo `artifacts/model.pkl` existe
- [ ] El archivo `artifacts/model_metadata.json` existe y contiene metricas
- [ ] `test_model.py` ejecuta predicciones correctamente
- [ ] Los clientes con contrato mes-a-mes tienen mayor probabilidad de churn

---

## Parte 4: Configurar Azure ML

En esta seccion vas a configurar la conexion con tu workspace de Azure ML. Si aun no tienes un workspace, consulta el [Apendice A](#apendice-a-crear-workspace-de-azure-ml).

### Paso 4.1: Autenticarse con Azure

```bash
az login
```

Se abrira tu navegador para que te autentiques. Despues de iniciar sesion, veras en la terminal la lista de suscripciones disponibles.

> **Si tienes multiples suscripciones**, selecciona la correcta:
> ```bash
> az account set --subscription "NOMBRE_O_ID_DE_TU_SUSCRIPCION"
> ```

### Paso 4.2: Crear archivo de configuracion

Crea un archivo `config.json` en la raiz del proyecto con los datos de tu workspace:

```json
{
    "subscription_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "resource_group": "nombre-de-tu-resource-group",
    "workspace_name": "nombre-de-tu-workspace"
}
```

> **Donde encontrar estos datos?**
> - **subscription_id**: Ejecuta `az account show --query id -o tsv`
> - **resource_group**: El nombre del resource group que creaste en Azure
> - **workspace_name**: El nombre del workspace de Azure ML

> **Seguridad**: El archivo `config.json` esta incluido en `.gitignore`, por lo que **no se subira** a GitHub. Nunca subas credenciales a tu repositorio.

### Paso 4.3: Verificar conexion

```bash
python scripts/verify_azure_connection.py
```

Salida esperada (si la conexion es exitosa):

```
Autenticando con Azure...
Conectando al workspace de Azure ML...

============================================================
CONEXION EXITOSA
============================================================
  Workspace: mi-workspace
  Location:  eastus
  Resource Group: mi-resource-group
  Subscription:   xxxxxxxx...
============================================================
```

Si ves este mensaje, tu configuracion esta correcta y puedes continuar con las siguientes secciones.

> **Si la conexion falla**, verifica:
> 1. Que ejecutaste `az login` y te autenticaste correctamente
> 2. Que los datos en `config.json` son correctos
> 3. Que tienes permisos de acceso al workspace (rol Contributor o superior)

### Checkpoint: Verificar conexion Azure

- [ ] `az login` funciono correctamente
- [ ] `config.json` tiene los datos correctos de tu workspace
- [ ] `verify_azure_connection.py` muestra "CONEXION EXITOSA"

---

## Parte 5: Registrar Modelo en Azure ML

En esta seccion vas a registrar el modelo entrenado en Azure ML usando MLflow, para que quede disponible en la nube para inferencia.

### Paso 5.1: Registrar el modelo

Asegurate de que `artifacts/model.pkl` y `artifacts/model_metadata.json` existen (de la Parte 3).

```bash
python scripts/register_model.py
```

Salida esperada:

```
Autenticando con Azure...
Conectando al workspace de Azure ML...
Configurando MLflow tracking URI...
Cargando modelo desde: artifacts/model.pkl
Creando datos de ejemplo para la firma...
Registrando modelo con MLflow...

============================================================
MODELO REGISTRADO EXITOSAMENTE
============================================================
  Modelo URI: runs:/xxxxxxxx/model
  Nombre: churn-prediction-model
============================================================
```

> **Que hace este script?**
> 1. Se conecta a tu workspace de Azure ML usando `DefaultAzureCredential` (tu sesion de `az login`)
> 2. Configura MLflow para que apunte al tracking server de tu workspace
> 3. Crea un run de MLflow donde registra:
>    - **Parametros**: tipo de modelo, semilla aleatoria, numero de muestras
>    - **Metricas**: accuracy, precision, recall, f1-score
>    - **Modelo**: el archivo `.pkl` con su signature (esquema de entrada/salida)
> 4. El modelo queda registrado como `churn-prediction-model` en el Model Registry de Azure ML

### Paso 5.2: Verificar en Azure ML Studio

1. Abre [Azure ML Studio](https://ml.azure.com)
2. Selecciona tu workspace
3. En el menu lateral, ve a **Assets** > **Models**
4. Veras `churn-prediction-model` listado con:
   - **Version**: 1 (se incrementa con cada registro)
   - **Framework**: scikit-learn
   - **Metricas**: accuracy, precision, recall, f1_score

5. Haz clic en el modelo para ver sus detalles:
   - Pestana **Details**: Informacion general y metadatos
   - Pestana **Artifacts**: Archivos del modelo (model.pkl, conda.yaml, etc.)

### Checkpoint: Verificar registro de modelo

- [ ] El script `register_model.py` se ejecuto sin errores
- [ ] El modelo aparece en Azure ML Studio > Models
- [ ] La version del modelo es 1 (o el numero que corresponda)
- [ ] Las metricas estan registradas correctamente

---

## Parte 6: Pipeline de Inferencia

En esta seccion vas a ejecutar un pipeline de inferencia completo en Azure ML: generar datos de entrada, subirlos a la nube, ejecutar el scoring, y descargar los resultados.

### Paso 6.1: Generar datos de inferencia

```bash
python scripts/create_inference_data.py
```

Salida esperada:

```
============================================================
  GENERACION DE DATOS DE INFERENCIA
============================================================

Generando 500 registros (semilla=123)...
  Registros generados: 500
  Archivo guardado en: data/inference_input.csv

Primeras 5 filas de ejemplo:
 customer_id  tenure_months  monthly_charges  ...
  CUST-10001             45            78.32  ...
  CUST-10002             12            55.10  ...
  ...

Generacion de datos completada exitosamente.
```

> **Nota**: Los datos generados tienen las mismas features que los datos de entrenamiento (tenure, monthly_charges, contract_type, etc.) pero **sin la columna `churned`** - esa es la que el modelo va a predecir.

### Paso 6.2: Subir datos a Azure ML

```bash
python scripts/upload_data.py
```

Salida esperada:

```
Autenticando con Azure...
Conectando al workspace de Azure ML...
Subiendo archivo: data/inference_input.csv

============================================================
DATOS SUBIDOS EXITOSAMENTE
============================================================
  Nombre: churn-inference-data
  Version: 1
  URI: azureml://...
============================================================
```

> **Verificacion**: Puedes ver los datos en Azure ML Studio > **Assets** > **Data**. El asset `churn-inference-data` deberia aparecer listado.

### Paso 6.3: Ejecutar pipeline de inferencia

```bash
python scripts/run_inference_pipeline.py
```

Salida esperada:

```
Autenticando con Azure...
Conectando al workspace de Azure ML...
Cluster de computo 'cpu-cluster' encontrado.
Creando entorno desde conda.yml...
Obteniendo ultima version del data asset 'churn-inference-data'...
  Data asset version: 1
Creando job de inferencia...
Enviando job a Azure ML...
Job enviado: churn-inference-job-xxxxx
Esperando finalizacion del job...
[... progreso del job ...]
Descargando resultados...

============================================================
PIPELINE EJECUTADO EXITOSAMENTE
============================================================
  Job URL: https://ml.azure.com/runs/...
  Estado: Completed
  Resultados: data/inference_output.csv
============================================================
```

> **Que sucede en la nube?**
> 1. Se crea (o reutiliza) un cluster de computo `cpu-cluster` (Standard_DS2_v2, 0-1 nodos)
> 2. Se crea un entorno de ejecucion basado en `conda.yml` con las dependencias necesarias
> 3. Se ejecuta `src/score.py` que:
>    - Carga el modelo `churn-prediction-model` desde el Model Registry de MLflow
>    - Lee el CSV de entrada
>    - Aplica el mismo preprocesamiento que durante el entrenamiento (`get_dummies`)
>    - Calcula probabilidades con `predict_proba`
>    - Guarda el resultado como CSV
> 4. El CSV de salida se descarga a `data/inference_output.csv`

> **Importante**: La primera ejecucion puede demorar varios minutos ya que Azure ML necesita:
> - Iniciar el cluster de computo (si tiene 0 nodos activos)
> - Construir la imagen Docker con el entorno conda
> Ejecuciones posteriores seran mas rapidas.

### Paso 6.4: Verificar resultados

Abre el archivo `data/inference_output.csv` y verifica que contiene:
- Todas las columnas originales (customer_id, tenure_months, monthly_charges, etc.)
- Una columna adicional `churn_probability` con valores entre 0 y 1

```bash
# Ver las primeras lineas del resultado
head -5 data/inference_output.csv
```

Los clientes con alta probabilidad de churn (cercana a 1.0) son los que el modelo predice que abandonaran el servicio.

### Checkpoint: Verificar pipeline de inferencia

- [ ] `create_inference_data.py` genero `data/inference_input.csv` con 500 registros
- [ ] `upload_data.py` subio los datos como Data Asset en Azure ML
- [ ] `run_inference_pipeline.py` ejecuto el job exitosamente
- [ ] `data/inference_output.csv` contiene la columna `churn_probability`
- [ ] El job es visible en Azure ML Studio > **Jobs**

---

## Bonus: Automatizacion con GitHub Actions

Esta seccion es **opcional y avanzada**. Muestra como automatizar el registro del modelo usando un workflow de GitHub Actions que se dispara manualmente.

### Concepto: Service Principal

Para que GitHub Actions pueda conectarse a Azure ML, necesitas una identidad de aplicacion llamada **Service Principal**. Es como un "usuario de servicio" que tiene permisos para acceder a tu workspace sin necesidad de login interactivo.

### Paso B.1: Crear Service Principal

```bash
az ad sp create-for-rbac \
    --name "mlops-lab-sp" \
    --role contributor \
    --scopes /subscriptions/TU_SUBSCRIPTION_ID/resourceGroups/TU_RESOURCE_GROUP \
    --sdk-auth
```

La salida sera similar a:

```json
{
    "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    ...
}
```

> **Guarda estos valores** - los necesitaras en el siguiente paso.

### Paso B.2: Configurar GitHub Secrets

1. Ve a tu repositorio en GitHub
2. Settings > Secrets and variables > Actions
3. Haz clic en **New repository secret** para cada uno de los siguientes:

| Nombre del Secret | Valor |
|-------------------|-------|
| `AZURE_TENANT_ID` | `tenantId` del Service Principal |
| `AZURE_CLIENT_ID` | `clientId` del Service Principal |
| `AZURE_CLIENT_SECRET` | `clientSecret` del Service Principal |
| `AZURE_SUBSCRIPTION_ID` | Tu Subscription ID de Azure |
| `AZURE_RESOURCE_GROUP` | Nombre de tu Resource Group |
| `AZURE_ML_WORKSPACE_NAME` | Nombre de tu workspace de Azure ML |

### Paso B.3: Ejecutar workflow de despliegue

1. Ve a la pestana **Actions** de tu repositorio
2. En el menu lateral izquierdo, selecciona **Deploy Model to Azure ML**
3. Haz clic en **Run workflow**
4. Opcionalmente, cambia el nombre del modelo (por defecto: `churn-prediction-model`)
5. Haz clic en **Run workflow** (boton verde)

El workflow ejecuta automaticamente:
1. Checkout del codigo
2. Instalacion de Python 3.10 y dependencias
3. Entrenamiento del modelo (`train_model.py`)
4. Registro del modelo en Azure ML (`register_model.py`) usando las credenciales del Service Principal

6. Una vez completado, veras un **Job Summary** con los detalles del despliegue

### Checkpoint: Verificar automatizacion

- [ ] Los 6 GitHub Secrets estan configurados
- [ ] El workflow `Deploy Model to Azure ML` se ejecuto exitosamente
- [ ] El modelo aparece registrado en Azure ML Studio con una nueva version

---

## Apendice A: Crear Workspace de Azure ML

Si aun no tienes un workspace de Azure ML, sigue estos pasos para crear uno desde cero.

### A.1: Crear Resource Group

```bash
# Crear un Resource Group (elige la region mas cercana)
az group create \
    --name mlops-lab-rg \
    --location eastus
```

### A.2: Crear Workspace de Azure ML

```bash
# Crear el workspace de Azure ML
az ml workspace create \
    --name mlops-lab-workspace \
    --resource-group mlops-lab-rg \
    --location eastus
```

> **Nota**: La creacion del workspace puede demorar unos minutos. Azure ML crea automaticamente recursos adicionales (Storage Account, Key Vault, Application Insights, Container Registry).

### A.3: Verificar workspace

```bash
# Listar workspaces para confirmar
az ml workspace list --resource-group mlops-lab-rg --output table
```

Deberias ver algo como:

```
Name                  Resource Group    Location
--------------------  ----------------  ----------
mlops-lab-workspace   mlops-lab-rg      eastus
```

### A.4: Obtener datos de configuracion

```bash
# Subscription ID
az account show --query id -o tsv

# Resource Group (ya lo sabes, es: mlops-lab-rg)

# Workspace Name (ya lo sabes, es: mlops-lab-workspace)
```

Con estos datos, crea tu `config.json` como se indica en la [Parte 4](#paso-42-crear-archivo-de-configuracion).

### A.5: Crear Service Principal (para la seccion Bonus)

```bash
# Obtener tu Subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# Crear Service Principal con permisos en el Resource Group
az ad sp create-for-rbac \
    --name "mlops-lab-sp" \
    --role contributor \
    --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/mlops-lab-rg \
    --sdk-auth
```

Guarda la salida JSON completa - contiene `clientId`, `clientSecret`, `tenantId`, y `subscriptionId` necesarios para configurar los GitHub Secrets.

### A.6: Verificar acceso del Service Principal (opcional)

```bash
# Iniciar sesion como Service Principal para verificar acceso
az login --service-principal \
    --username CLIENT_ID \
    --password CLIENT_SECRET \
    --tenant TENANT_ID

# Verificar acceso al workspace
az ml workspace show \
    --name mlops-lab-workspace \
    --resource-group mlops-lab-rg

# Volver a tu sesion normal
az login
```

---

## Apendice B: Troubleshooting

### Errores de CI (Lint)

#### Black: "would reformat"

```
error: would reformat scripts/train_model.py
```

**Solucion**: Ejecuta Black para formatear automaticamente:

```bash
black scripts/
```

#### isort: "would sort imports differently"

```
ERROR: scripts/train_model.py Imports are incorrectly sorted
```

**Solucion**: Ejecuta isort para ordenar los imports:

```bash
isort scripts/
```

#### Flake8: "E501 line too long"

```
scripts/train_model.py:42:89: E501 line too long (95 > 88 characters)
```

**Solucion**: Divide la linea larga en varias lineas. El limite configurado es de 88 caracteres.

#### Pylint: warnings comunes

```
C0114: Missing module docstring
```

**Solucion**: Agrega un docstring al inicio del archivo:

```python
"""Descripcion del modulo."""
```

#### MyPy: "Cannot find implementation or library stub"

```
error: Cannot find implementation or library stub for "sklearn"
```

**Solucion**: Este error deberia estar suprimido por la configuracion `ignore_missing_imports = true` en `pyproject.toml`. Si aparece, verifica que tu `pyproject.toml` contiene esa configuracion.

### Errores de conexion Azure

#### "DefaultAzureCredential failed"

```
Error de conexion: DefaultAzureCredential failed to retrieve a token
```

**Solucion**:

1. Verifica que ejecutaste `az login`:
   ```bash
   az login
   ```

2. Verifica que tu sesion no expiro:
   ```bash
   az account show
   ```

3. Si la sesion expiro, vuelve a autenticarte:
   ```bash
   az login
   ```

#### "Subscription not found"

```
Error de conexion: (SubscriptionNotFound)
```

**Solucion**: Verifica que el `subscription_id` en `config.json` es correcto:

```bash
az account list --output table
```

#### "Resource group not found"

**Solucion**: Verifica que el resource group existe:

```bash
az group list --output table
```

### Errores de registro de modelo

#### "Workspace not found"

**Solucion**: Verifica que el nombre del workspace en `config.json` es correcto:

```bash
az ml workspace list --resource-group TU_RESOURCE_GROUP --output table
```

#### "model.pkl not found"

```
Error: No se encontro el modelo en: artifacts/model.pkl
```

**Solucion**: Ejecuta primero el entrenamiento:

```bash
python scripts/train_model.py
```

#### "MLflow tracking URI error"

**Solucion**: Verifica que el workspace tiene MLflow habilitado (todos los workspaces modernos de Azure ML lo tienen por defecto). Si el error persiste, intenta:

```bash
pip install --upgrade mlflow azureml-mlflow
```

### Errores de pipeline de inferencia

#### "Compute not found" o timeout al crear cluster

**Solucion**: El cluster `cpu-cluster` se crea automaticamente. Si falla:

1. Verifica que tu suscripcion tiene cuota para VMs `Standard_DS2_v2`:
   ```bash
   az ml compute list --resource-group TU_RG --workspace-name TU_WS --output table
   ```

2. Si hay problemas de cuota, puedes cambiar el tamano de VM editando `scripts/run_inference_pipeline.py`.

#### "Environment build failed"

**Solucion**: La imagen Docker del entorno puede fallar si hay conflictos de dependencias. Verifica que `conda.yml` tiene versiones compatibles:

```yaml
dependencies:
  - python=3.10
  - pip
  - pip:
      - scikit-learn==1.3.2
      - pandas==2.1.4
      - mlflow==2.9.2
      - azureml-mlflow==1.57.0
```

#### "Data asset not found"

```
Error: Data asset 'churn-inference-data' not found
```

**Solucion**: Asegurate de haber subido los datos primero:

```bash
python scripts/create_inference_data.py
python scripts/upload_data.py
```

#### "Model not found in registry"

**Solucion**: Asegurate de haber registrado el modelo primero (Parte 5):

```bash
python scripts/register_model.py
```

---

**Fin del tutorial.** Si completaste todas las partes, has implementado un flujo MLOps completo: desde CI/CD con GitOps hasta inferencia en la nube con Azure ML.
