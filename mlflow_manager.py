# Contenido del NUEVO archivo: mlflow_manager.py

import mlflow
import mlflow.tensorflow
from model_training import run_experiment


def run_mlflow_experiment(params, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Executes a complete training experiment and tracks it using MLflow.

    This function orchestrates a single experiment run. It starts an MLflow run,
    logs the specified hyperparameters, calls the main training function,
    and then logs the resulting performance metrics and the trained model as an
    artifact to the MLflow tracking server. This approach ensures that each

    experiment is reproducible and its results are centrally managed.


    Args:
        params (dict): A dictionary of hyperparameters and settings for the run.
                       This dictionary will be logged directly to MLflow.
                       Example: {'model_type': 'LSTM', 'epochs': 50, 'lr': 0.001}
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        X_val (np.ndarray): Validation feature data.
        y_val (np.ndarray): Validation target data.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test target data.

    Returns:
        The trained model object returned by the run_experiment function.
    """

    with mlflow.start_run(run_name=params['model_type']) as run:
        print(f"\n--- Iniciando run de MLflow para: {params['model_type']} ---")

        mlflow.log_params(params)

        model, history, final_loss, final_accuracy = run_experiment(
            model_name=params['model_type'],
            params=params,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )

        print(f"Registrando métricas finales en MLflow...")
        mlflow.log_metric("final_test_accuracy", final_accuracy)
        mlflow.log_metric("final_test_loss", final_loss)

        # 4. Registrar el modelo como un artefacto explícitamente
        print("Registrando el modelo en MLflow...")
        mlflow.tensorflow.log_model(model=model, name="model")
        # El `artifact_path` es el nombre de la carpeta donde se guardará dentro del run.

        print(f"--- Run de MLflow para {params['model_type']} finalizado. ---")

        return model