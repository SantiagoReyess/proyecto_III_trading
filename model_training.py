# model_training.py

import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow


def prepare_data_for_model(df, lookback_period, val_size=0.15, test_size=0.15):
    """
    Prepara los datos (que ya vienen escalados) dividiéndolos cronológicamente
    en entrenamiento, validación y prueba.
    """
    # 1. Separar features (X) y target (y)
    features = df.drop(columns=['Date', 'Price', 'Open', 'High', 'Low', 'CVol', 'Signal'])
    target = df['Signal']

    # 2. Crear secuencias
    X_sequences, y_sequences = [], []
    for i in range(len(features) - lookback_period):
        X_sequences.append(features.iloc[i:i + lookback_period].values)
        y_sequences.append(target.iloc[i + lookback_period])
    X_sequences, y_sequences = np.array(X_sequences), np.array(y_sequences)

    # 3. Dividir cronológicamente
    test_split_index = int(len(X_sequences) * (1 - test_size))
    val_split_index = int(len(X_sequences) * (1 - test_size - val_size))

    X_train = X_sequences[:val_split_index]
    y_train = y_sequences[:val_split_index]

    X_val = X_sequences[val_split_index:test_split_index]
    y_val = y_sequences[val_split_index:test_split_index]

    X_test = X_sequences[test_split_index:]
    y_test = y_sequences[test_split_index:]

    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de X_val:   {X_val.shape}")
    print(f"Forma de X_test:  {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# (Las funciones de creación de modelos se quedan igual)
def create_dnn_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),  # Aplanamos la secuencia para la DNN
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_model(input_shape):
    # (Esta función se queda igual)
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_lstm_model(input_shape):
    # (Esta función se queda igual)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_creator(model_name):
    # Modificado para aceptar la nueva DNN
    if model_name.lower() == 'dnn':
        return create_dnn_model
    elif model_name.lower() == 'cnn':
        return create_cnn_model
    elif model_name.lower() == 'lstm':
        return create_lstm_model
    else:
        raise ValueError("Nombre de modelo no reconocido.")


def run_experiment(params, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Ejecuta un único experimento de entrenamiento con MLflow.
    Usa conjuntos de validación y prueba separados.
    """
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name=params['model_type']) as run:
        mlflow.log_params(params)
        model_name = params['model_type']
        print(f"\n--- Ejecutando experimento para: {model_name} ---")
        print(f"Parámetros: {params}")

        # Para CNN/LSTM, la forma es (timesteps, features)
        # Para la nueva DNN, la capa Flatten se encarga de la forma
        input_shape = (X_train.shape[1], X_train.shape[2])

        model_creator = get_model_creator(model_name)
        model = model_creator(input_shape)

        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(enumerate(weights))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Usar el conjunto de VALIDACIÓN para el callback y la validación durante el entrenamiento
        history = model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 100),
            batch_size=params.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            class_weight=class_weights_dict,
            verbose=2
        )

        # Usar el conjunto de PRUEBA solo para la evaluación final
        print(f"Evaluando el modelo final en el conjunto de prueba (datos nunca vistos)...")
        final_loss, final_accuracy = model.evaluate(X_test, y_test, verbose=0)

        mlflow.log_metric("final_test_accuracy", final_accuracy)
        mlflow.log_metric("final_test_loss", final_loss)

        print(f"--- Experimento para {model_name} finalizado. Test Accuracy: {final_accuracy:.4f} ---")