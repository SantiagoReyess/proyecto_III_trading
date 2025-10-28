import pandas as pd
import numpy as np

def label(dataframe, alpha):

    """
    Generates trading labels (buy, sell, hold) based on future price movements.

    This function implements a simplified version of the "triple-barrier method" for
    labeling financial time-series data. For each data point, it looks ahead a fixed
    number of periods (window) to calculate the future return. Based on this return,
    it assigns a categorical label:
    - '2' (Buy): If the future return exceeds a positive threshold (alpha).
    - '0' (Sell): If the future return is below a negative threshold (-alpha).
    - '1' (Hold): If the future return is between the two thresholds.

    This is used to create a target variable for a supervised machine learning model.


    Args:
        dataframe (pd.DataFrame): The input DataFrame containing at least a 'Price' column.
        alpha (float): The symmetrical threshold for the return (e.g., 0.02 for 2%)
                       that triggers a buy or sell signal.

    Returns:
        pd.DataFrame: The original DataFrame with three new columns:
                      'future_price', 'future_return', and 'signal'.
    """

    window = 15
    buy_threshold = alpha
    sell_threshold = -alpha

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
