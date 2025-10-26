# Contenido del NUEVO archivo: mlflow_manager.py

import mlflow
import mlflow.tensorflow
from model_training import run_experiment


def run_mlflow_experiment(params, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Ejecuta un experimento de entrenamiento completo utilizando MLflow para el tracking.
    """
    # Habilitamos el autologging de TensorFlow
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name=params['model_type']) as run:
        print(f"\n--- Iniciando run de MLflow para: {params['model_type']} ---")

        # 1. Registrar los parámetros
        mlflow.log_params(params)

        # 2. Llamar a la función de entrenamiento para que haga el trabajo pesado
        model, history, final_loss, final_accuracy = run_experiment(
            model_name=params['model_type'],
            params=params,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )

        # 3. Registrar las métricas finales manualmente (además del autologging)
        print(f"Registrando métricas finales en MLflow...")
        mlflow.log_metric("final_test_accuracy", final_accuracy)
        mlflow.log_metric("final_test_loss", final_loss)

        print(f"--- Run de MLflow para {params['model_type']} finalizado. ---")

        return model