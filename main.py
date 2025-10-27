from data_download import data_download
from technical_indicators import get_signals
from data_labeling import label
from model_training import prepare_data_for_model, run_experiment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtesting import backtesting
from mlflow_manager import run_mlflow_experiment

def main():

    ## Descargar los datos
    data = data_download("PriceHistory.xlsx")

    ## Obtener los indicadores y escalar los datos
    data = get_signals(data)

    ## Label the dataframe (0 = sell, 1 = hold, 2 = buy)
    data = label(data, alpha=0.02)

    ## Prepare the data for the model
    lookback_period = 25
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data_for_model(df=data, lookback_period=lookback_period)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    y_val = y_val.astype(np.int32)

    params = {"epochs": 100, "batch_size": 10}
    model_dnn, history_dnn, final_loss_dnn, final_accuracy_dnn = run_experiment(model_name="dnn", params=params,
                                                                X_train=X_train, y_train=y_train,
                                                                X_test=X_test, y_test=y_test,
                                                                X_val=X_val, y_val=y_val)

    model_cnn, history_cnn, final_loss_cnn, final_accuracy_cnn = run_experiment(model_name="cnn", params=params,
                                                                X_train=X_train, y_train=y_train,
                                                                X_test=X_test, y_test=y_test,
                                                                X_val=X_val, y_val=y_val)

    print("\n--- Iniciando Backtesting Comparativo ---")

    # Preparamos el dataframe de prueba que usaremos para ambos modelos
    test_df_base = data.iloc[-len(X_test):].copy()

    # --- Backtest para el modelo DNN ---
    print("\n1. Ejecutando backtest para DNN...")
    predictions_dnn_prob = model_dnn.predict(X_test)
    predictions_dnn = np.argmax(predictions_dnn_prob, axis=1)

    test_df_dnn = test_df_base.copy()
    test_df_dnn['buy_signal'] = (predictions_dnn == 2)
    test_df_dnn['sell_signal'] = (predictions_dnn == 0)

    stop_loss = 0.05
    take_profit = 0.10
    n_shares = 100

    portfolio_historic_dnn = backtesting(dataframe=test_df_dnn,
                                         stop_loss=stop_loss,
                                         take_profit=take_profit,
                                         n_shares=n_shares)

    # --- Backtest para el modelo CNN ---
    print("2. Ejecutando backtest para CNN...")
    predictions_cnn_prob = model_cnn.predict(X_test)
    predictions_cnn = np.argmax(predictions_cnn_prob, axis=1)

    test_df_cnn = test_df_base.copy()
    test_df_cnn['buy_signal'] = (predictions_cnn == 2)
    test_df_cnn['sell_signal'] = (predictions_cnn == 0)

    portfolio_historic_cnn = backtesting(dataframe=test_df_cnn,
                                         stop_loss=stop_loss,
                                         take_profit=take_profit,
                                         n_shares=n_shares)

    # --- Resultados numéricos ---
    initial_capital = portfolio_historic_dnn[0]  # Es el mismo para ambos
    final_capital_dnn = portfolio_historic_dnn[-1]
    returns_dnn = (final_capital_dnn / initial_capital - 1) * 100

    final_capital_cnn = portfolio_historic_cnn[-1]
    returns_cnn = (final_capital_cnn / initial_capital - 1) * 100

    print("\n--- Resultados Finales del Backtesting ---")
    print(f"Capital Inicial: ${initial_capital:,.2f}")
    print("-" * 35)
    print(f"Estrategia DNN - Capital Final: ${final_capital_dnn:,.2f} | Rendimiento: {returns_dnn:.2f}%")
    print(f"Estrategia CNN - Capital Final: ${final_capital_cnn:,.2f} | Rendimiento: {returns_cnn:.2f}%")
    print("-" * 35)

    # --- Gráfica comparativa ---
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_historic_dnn, label='Estrategia DNN', color='blue')
    plt.plot(portfolio_historic_cnn, label='Estrategia CNN', color='red', linestyle='--')

    plt.title("Comparación de Estrategias: Evolución del Capital", fontsize=16)
    plt.xlabel("Periodos de Tiempo (Días)", fontsize=12)
    plt.ylabel("Valor del Portafolio ($)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()