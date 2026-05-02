# MLOps Wine Quality Pipeline

## 1. Descripción del proyecto

Este proyecto desarrolla un pipeline reproducible de machine learning para clasificar la calidad del vino a partir de variables fisicoquímicas. El flujo implementado integra las etapas principales de un proyecto de MLOps: carga de datos, validación, preprocesamiento, entrenamiento, evaluación, registro del modelo con MLflow y automatización mediante GitHub Actions.

El pipeline fue diseñado para ejecutarse tanto de forma local como en la nube, permitiendo que el modelo pueda entrenarse y validarse automáticamente cada vez que se realicen cambios en el repositorio.

## 2. Objetivo general

Desarrollar un pipeline reproducible de machine learning que permita entrenar, evaluar y registrar un modelo, y que esté completamente automatizado mediante CI/CD usando GitHub Actions.

## 3. Dataset utilizado

Para el desarrollo del proyecto se utilizó el dataset público Wine Quality Red Wine Dataset, disponible en UCI Machine Learning Repository.

El archivo utilizado es:

```text
data/winequality-red.csv
```

Este dataset contiene información fisicoquímica de muestras de vino tinto, incluyendo variables como:

```text
fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
quality
```

La variable objetivo original es `quality`, que representa la calidad del vino en una escala numérica.

Para este proyecto, el problema fue transformado en una tarea de clasificación binaria:

```text
quality >= 6  →  1, vino de buena calidad
quality < 6   →  0, vino de calidad regular
```

No se utilizaron datasets provenientes de `sklearn.datasets`, cumpliendo con el requisito de emplear una fuente de datos externa y de libre acceso.

## 4. Estructura del proyecto

La estructura general del repositorio es la siguiente:

```text
mlops-wine-quality-pipeline/
│
├── .github/
│   └── workflows/
│       └── ml.yml
│
├── data/
│   └── winequality-red.csv
│
├── models/
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── utils.py
│
├── tests/
│   └── test_pipeline.py
│
├── config.yaml
├── requirements.txt
├── Makefile
├── README.md
└── .gitignore
```

## 5. Descripción del pipeline de machine learning

El pipeline implementado realiza las siguientes etapas:

1. Carga del dataset desde un archivo CSV.
2. Validación básica del dataset.
3. Limpieza de datos.
4. Eliminación de valores nulos.
5. Eliminación de registros duplicados.
6. Transformación de la variable objetivo en una variable binaria.
7. División de los datos en entrenamiento y prueba.
8. Entrenamiento de un modelo de clasificación.
9. Evaluación del modelo mediante métricas.
10. Registro de parámetros, métricas, firma del modelo y ejemplo de entrada en MLflow.
11. Almacenamiento del modelo entrenado como artefacto.
12. Automatización del flujo mediante GitHub Actions.

## 6. Modelo utilizado

Se utilizó un modelo `RandomForestClassifier` de la librería `scikit-learn`.

El modelo se integró dentro de un pipeline que incluye escalamiento de variables mediante `StandardScaler` y posteriormente el entrenamiento del clasificador.

```text
StandardScaler + RandomForestClassifier
```

Los hiperparámetros principales se encuentran definidos en el archivo `config.yaml`, lo cual permite modificar la configuración del experimento sin alterar directamente el código fuente.

Ejemplo de configuración del modelo:

```yaml
model:
  type: "RandomForestClassifier"
  n_estimators: 100
  max_depth: 8
  random_state: 42
```

## 7. Archivo de configuración

El archivo `config.yaml` centraliza las rutas, parámetros del modelo, configuración de entrenamiento y configuración de MLflow.

Este archivo permite controlar:

```text
Ruta del dataset
Separador del archivo CSV
Columna objetivo
Hiperparámetros del modelo
Tamaño del conjunto de prueba
Semilla aleatoria
Ruta de tracking de MLflow
Nombre del experimento
Ruta de salida del modelo
```

Esto hace que el proyecto sea más organizado, mantenible y reproducible.

## 8. Métricas de evaluación

El modelo fue evaluado mediante métricas de clasificación. Las métricas registradas fueron:

```text
accuracy
f1_score
precision
recall
```

En una ejecución local del pipeline se obtuvieron los siguientes resultados:

```text
accuracy: 0.7684
f1_score: 0.7774
precision: 0.7914
recall: 0.7639
```

Estas métricas permiten analizar el desempeño general del modelo y su capacidad para clasificar correctamente los vinos de buena calidad frente a los de calidad regular.

## 9. Tracking con MLflow

El proyecto utiliza MLflow para registrar y organizar los experimentos de machine learning.

Durante la ejecución del pipeline se registran:

```text
Parámetros del modelo
Métricas de evaluación
Firma del modelo
Ejemplo de entrada
Modelo entrenado como artefacto
```

El tracking local de MLflow se configura en:

```text
file:./mlruns
```

Para visualizar los experimentos localmente, se puede ejecutar:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Luego se abre en el navegador:

```text
http://127.0.0.1:5000
```

En la interfaz de MLflow se puede observar el experimento `wine_quality_classification`, el run `random_forest_wine_quality`, las métricas, los parámetros y el modelo registrado.

## 10. Instalación local

Para ejecutar el proyecto localmente, primero se debe clonar el repositorio:

```bash
git clone https://github.com/MechCode2004/mlops-wine-quality-pipeline.git
```

Ingresar a la carpeta del proyecto:

```bash
cd mlops-wine-quality-pipeline
```

Crear un entorno virtual:

```bash
python -m venv venv
```

Activar el entorno virtual en Windows:

```bash
venv\Scripts\activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

También se puede usar el Makefile:

```bash
make install
```

## 11. Ejecución del pipeline

Para ejecutar el entrenamiento completo del modelo:

```bash
python src/train.py
```

O usando el Makefile:

```bash
make train
```

Al finalizar la ejecución, el sistema muestra las métricas del modelo en consola y registra el experimento en MLflow.

## 12. Ejecución de pruebas

El proyecto incluye pruebas básicas para validar el funcionamiento del pipeline.

Para ejecutar las pruebas:

```bash
pytest tests
```

O usando el Makefile:

```bash
make test
```

Las pruebas validan aspectos como:

```text
La existencia de la columna objetivo
La correcta transformación de la variable objetivo
La generación de una variable binaria
La coherencia entre variables predictoras y variable objetivo
```

## 13. Validación de estilo

Para validar el estilo del código se utiliza `flake8`.

Ejecutar:

```bash
flake8 src tests
```

O usando el Makefile:

```bash
make lint
```

Esta validación permite mantener un código más limpio, legible y consistente.

## 14. Automatización con GitHub Actions

El proyecto cuenta con un workflow de CI/CD definido en:

```text
.github/workflows/ml.yml
```

El workflow se ejecuta automáticamente cuando hay cambios en la rama `main`, cuando se abre un pull request o cuando se ejecuta manualmente desde GitHub Actions.

El flujo automatizado realiza los siguientes pasos:

```text
Clonar el repositorio
Configurar Python
Instalar dependencias
Ejecutar lint
Ejecutar pruebas
Entrenar el modelo
Guardar el modelo como artefacto
Guardar los experimentos de MLflow como artefacto
```

Esto permite que el pipeline sea reproducible en la nube y que cada cambio en el proyecto sea validado automáticamente.

## 15. Makefile

El proyecto incluye un archivo `Makefile` para facilitar la ejecución de tareas comunes.

Las tareas disponibles son:

```bash
make install
```

Instala las dependencias del proyecto.

```bash
make lint
```

Ejecuta la validación de estilo con `flake8`.

```bash
make test
```

Ejecuta las pruebas con `pytest`.

```bash
make train
```

Ejecuta el pipeline completo de entrenamiento.

## 16. Artefactos generados

Durante la ejecución del workflow en GitHub Actions se generan dos artefactos principales:

```text
wine-quality-model
mlflow-runs
```

El artefacto `wine-quality-model` contiene el modelo entrenado.

El artefacto `mlflow-runs` contiene los registros generados por MLflow durante la ejecución del pipeline.

Estos artefactos pueden descargarse directamente desde la ejecución del workflow en GitHub Actions.

## 17. Evidencias del proyecto

Las evidencias del proyecto incluyen:

```text
Captura de pantalla de MLflow mostrando el experimento registrado
Captura de pantalla de GitHub Actions ejecutado correctamente
Artefactos descargados desde GitHub Actions
URL del repositorio público
Archivo ZIP del proyecto
```

En MLflow se evidencia el registro de:

```text
Experimento: wine_quality_classification
Run: random_forest_wine_quality
Métricas: accuracy, f1_score, precision, recall
Parámetros: model_type, n_estimators, max_depth, test_size
Modelo: wine_quality_model
```

En GitHub Actions se evidencia la ejecución exitosa del workflow `ML Pipeline CI/CD`, incluyendo instalación, lint, pruebas, entrenamiento y carga de artefactos.

## 18. Repositorio público

URL del repositorio:

```text
https://github.com/MechCode2004/mlops-wine-quality-pipeline
```

## 19. Requisitos cumplidos

El proyecto cumple con los siguientes requisitos:

```text
Dataset de libre acceso externo a sklearn.datasets
Código organizado en carpeta src/
Script principal src/train.py
Uso de config.yaml para rutas e hiperparámetros
Preprocesamiento básico de datos
División entre entrenamiento y prueba
Entrenamiento de un modelo de clasificación
Cálculo de más de dos métricas de evaluación
Tracking con MLflow
Registro de parámetros, métricas, firma y ejemplo de entrada
Modelo guardado como artefacto
Makefile con tareas automáticas
Pruebas básicas con pytest
Validación de estilo con flake8
Workflow de CI/CD con GitHub Actions
Artefactos generados desde GitHub Actions
README con instrucciones claras
Repositorio público en GitHub
```

## 20. Conclusión

Este proyecto demuestra la implementación de un flujo básico de MLOps aplicado a un problema de clasificación de machine learning. La solución permite entrenar, evaluar y registrar un modelo de forma reproducible, integrando herramientas como scikit-learn, MLflow, pytest, flake8, Makefile y GitHub Actions.

La automatización mediante CI/CD permite validar el código, ejecutar pruebas, entrenar el modelo y guardar los artefactos de manera automática. Esto fortalece la trazabilidad, reproducibilidad y mantenimiento del proyecto, aspectos fundamentales en el desarrollo moderno de soluciones de machine learning.
