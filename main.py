from data_download import data_download
from technical_indicators import get_signals
from split import  split_ttv
from data_labeling import label
from model_training import prepare_data_for_model, create_dnn_model, create_cnn_model, create_lstm_model, get_model_creator, run_experiment


def main():

    ## Descargar los datos
    data = data_download("PriceHistory.xlsx")

    ## Obtener los indicadores y escalar los datos
    data = get_signals(data)

    ## Label the dataframe (0 = sell, 1 = hold, 2 = buy)
    data = label(data, alpha=0.05)

    ## Prepare the data for the model
    lookback_period = 5
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data_for_model(data=data, lookback_period=lookback_period)




    return print(X_train)

if __name__ == "__main__":
    main()