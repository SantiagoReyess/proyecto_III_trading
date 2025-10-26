from data_download import data_download
from technical_indicators import get_signals
from split import  split_ttv
from data_labeling import label


def main():

    ## Descargar los datos
    data = data_download("PriceHistory.xlsx")

    ## Obtener los indicadores y escalar los datos
    data = get_signals(data)

    ## Label the dataframe (0 = sell, 1 = hold, 2 = buy)
    data = label(data, alpha=0.05)

    ## Prepare the data for the model


    return print(data["signal"])

if __name__ == "__main__":
    main()