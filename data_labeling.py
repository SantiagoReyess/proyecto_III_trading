import pandas as pd
import numpy as np

def label(dataframe, alpha):

    window = 7
    buy_threshold = alpha
    sell_threshold = alpha

    # Calcular retorno futuro (para etiquetas)
    dataframe['future_price'] = dataframe['Price'].shift(-window)
    dataframe['future_return'] = (dataframe['future_price'] - dataframe['Price']) / dataframe['Price']

    # Etiquetas: buy / sell / hold
    conditions = [
        dataframe['future_return'] > buy_threshold,
        dataframe['future_return'] < sell_threshold
    ]
    choices = ['2', '0']

    dataframe['signal'] = np.select(conditions, choices, default='1')

    return dataframe
