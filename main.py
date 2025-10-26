from data_download import data_download
from technical_indicators import get_signals
from split import  split_ttv
from data_labeling import label
from model_training import prepare_data_for_model, run_experiment

import numpy as np

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

    params = {"epochs": 20, "batch_size": 10}
    model_dnn, history_dnn, final_loss_dnn, final_accuracy_dnn = run_experiment(model_name="dnn", params=params,
                                                                X_train=X_train, y_train=y_train,
                                                                X_test=X_test, y_test=y_test,
                                                                X_val=X_val, y_val=y_val)

    model_cnn, history_cnn, final_loss_cnn, final_accuracy_cnn = run_experiment(model_name="cnn", params=params,
                                                                X_train=X_train, y_train=y_train,
                                                                X_test=X_test, y_test=y_test,
                                                                X_val=X_val, y_val=y_val)







    return print(final_accuracy_dnn, final_accuracy_cnn, data['signal'].value_counts(normalize=True))

if __name__ == "__main__":
    main()