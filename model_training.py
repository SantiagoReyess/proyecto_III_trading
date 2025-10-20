# model_training_mlflow.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow

def prepare_data_for_model(df, lookback_period, test_size=0.2):
    X = df.drop(columns=['Signal', 'Date', 'Price', 'Open', 'High', 'Low', 'CVol']) # Modificar
    y = df['Signal']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_sequences, y_sequences = [], []
    for i in range(len(X_scaled) - lookback_period):
        X_sequences.append(X_scaled[i:i + lookback_period])
        y_sequences.append(y.iloc[i + lookback_period])
    X_sequences, y_sequences = np.array(X_sequences), np.array(y_sequences)
    split_index = int(len(X_sequences) * (1 - test_size))
    X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
    y_train, y_test = y_sequences[:split_index], y_sequences[split_index:]
    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def create_dnn_model(input_shape):
    # (Esta función se queda igual)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape[1],)),
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
    """Devuelve la función que crea el modelo solicitado."""
    if model_name.lower() == 'dnn':
        return create_dnn_model
    elif model_name.lower() == 'cnn':
        return create_cnn_model
    elif model_name.lower() == 'lstm':
        return create_lstm_model
    else:
        raise ValueError("Nombre de modelo no reconocido.")



