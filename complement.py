import mlflow.pyfunc
from data_download import data_download
from technical_indicators import get_signals
import numpy as np
from model_training import prepare_data_for_model, run_experiment
from data_labeling import label
from backtesting import backtesting
import matplotlib.pyplot as plt
import pandas as pd
from metrics import calculate_metrics

# import best models
# Use your local Windows path — note the raw string (r"") to avoid backslash issues
dnn_path = r".\mlartifacts\0\models\m-f0c602d22af64df6810d04a7104ffd57\artifacts"
best_dnn_model = mlflow.pyfunc.load_model(dnn_path)

cnn_path = r".\mlartifacts\0\models\m-a48c4dd9bb8a4a91828285bb67a9d23d\artifacts"
best_cnn_model = mlflow.pyfunc.load_model(cnn_path)

# evaluate backtesting
data = data_download("PriceHistory.xlsx")
data = get_signals(data)

data = label(data, alpha=0.05)
print(data)

lookback_period = 50
X_train, y_train, X_test, y_test, X_val, y_val = prepare_data_for_model(df=data, lookback_period=lookback_period)

train_df_base = data.iloc[:len(X_train)].copy()
val_df_base = data.iloc[len(X_train):len(X_train)+len(X_val)].copy()
test_df_base = data.iloc[-len(X_test):].copy()

## predictions for dnn
dnn_train_pred = best_dnn_model.predict(X_train)
dnn_train_pred = np.argmax(dnn_train_pred, axis=1)

dnn_val_pred = best_dnn_model.predict(X_val)
dnn_val_pred = np.argmax(dnn_val_pred, axis=1)

dnn_test_pred = best_dnn_model.predict(X_test)
dnn_test_pred = np.argmax(dnn_test_pred, axis=1)

train_df_dnn = train_df_base.copy()
train_df_dnn['buy_signal'] = (dnn_train_pred == 2)
train_df_dnn['sell_signal'] = (dnn_train_pred == 0)

val_df_dnn = val_df_base.copy()
val_df_dnn['buy_signal'] = (dnn_val_pred == 2)
val_df_dnn['sell_signal'] = (dnn_val_pred == 0)

test_df_dnn = test_df_base.copy()
test_df_dnn['buy_signal'] = (dnn_test_pred == 2)
test_df_dnn['sell_signal'] = (dnn_test_pred == 0)

## predictions for cnn
cnn_train_pred= best_cnn_model.predict(X_train)
cnn_train_pred = np.argmax(cnn_train_pred, axis=1)

cnn_val_pred= best_cnn_model.predict(X_val)
cnn_val_pred = np.argmax(cnn_val_pred, axis=1)

cnn_test_pred= best_cnn_model.predict(X_test)
cnn_test_pred = np.argmax(cnn_test_pred, axis=1)

train_df_cnn = train_df_base.copy()
train_df_cnn['buy_signal'] = (cnn_train_pred == 2)
train_df_cnn['sell_signal'] = (cnn_train_pred == 0)

val_df_cnn = val_df_base.copy()
val_df_cnn['buy_signal'] = (cnn_val_pred == 2)
val_df_cnn['sell_signal'] = (cnn_val_pred == 0)

test_df_cnn = test_df_base.copy()
test_df_cnn['buy_signal'] = (cnn_test_pred == 2)
test_df_cnn['sell_signal'] = (cnn_test_pred == 0)

## Backtesting
stop_loss = 0.10
take_profit = 0.10
n_shares = 1200

train_historic_dnn = backtesting(dataframe=train_df_dnn,
                                         stop_loss=stop_loss,
                                         take_profit=take_profit,
                                         n_shares=n_shares)
val_historic_dnn = backtesting(dataframe=val_df_dnn,
                                         stop_loss=stop_loss,
                                         take_profit=take_profit,
                                         n_shares=n_shares)
test_historic_dnn = backtesting(dataframe=test_df_dnn,
                                         stop_loss=stop_loss,
                                         take_profit=take_profit,
                                         n_shares=n_shares)


train_historic_cnn = backtesting(dataframe=train_df_cnn,
                                         stop_loss=stop_loss,
                                         take_profit=take_profit,
                                         n_shares=n_shares)
val_historic_cnn = backtesting(dataframe=val_df_cnn,
                                         stop_loss=stop_loss,
                                         take_profit=take_profit,
                                         n_shares=n_shares)
test_historic_cnn = backtesting(dataframe=test_df_cnn,
                                         stop_loss=stop_loss,
                                         take_profit=take_profit,
                                         n_shares=n_shares)


import matplotlib.pyplot as plt
import pandas as pd

# --- Unir los periodos de DNN ---
dnn_full = pd.concat([
    pd.Series(train_historic_dnn),
    pd.Series(val_historic_dnn),
    pd.Series(test_historic_dnn)
]).reset_index(drop=True)

# --- Unir los periodos de CNN ---
cnn_full = pd.concat([
    pd.Series(train_historic_cnn),
    pd.Series(val_historic_cnn),
    pd.Series(test_historic_cnn)
]).reset_index(drop=True)

# --- Calcular posiciones de separación ---
train_end = len(train_historic_dnn)
val_end = train_end + len(val_historic_dnn)
test_end = val_end + len(test_historic_cnn)  # final total (por si se necesita)

# --- Graficar ---
plt.figure(figsize=(12, 6))
plt.plot(dnn_full, label='DNN Portfolio', color='darkblue', linewidth=1.8, alpha=0.5)
plt.plot(cnn_full, label='CNN Portfolio', color='darkred', linewidth=1.8, alpha=0.5)

# --- Líneas verticales para separar periodos ---
plt.axvline(x=train_end, color='gray', linestyle='--', alpha=0.8, label='End of Train')
plt.axvline(x=val_end, color='gray', linestyle=':', alpha=0.8, label='End of Validation')

# --- (Opcional) Sombras ligeras para distinguir periodos ---
plt.axvspan(0, train_end, color='blue', alpha=0.05)
plt.axvspan(train_end, val_end, color='orange', alpha=0.05)
plt.axvspan(val_end, test_end, color='green', alpha=0.05)

# --- Estilo general ---
plt.title('Portfolio Value Over Time (Train → Validation → Test)', fontsize=13)
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# print metrics
print(calculate_metrics(test_historic_dnn))
print(calculate_metrics(test_historic_cnn))

# Graph train test validation portfolio values