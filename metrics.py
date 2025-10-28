import numpy as np
import pandas as pd

def calculate_metrics(portfolio_historic):
    """
    Calculates financial performance metrics from a portfolio's value history.

    This function takes a time series of portfolio values and computes several
    annualized performance and risk metrics. It assumes the input data points
    are on a daily frequency, based on the annualization factor of 252.

    Args:
        portfolio_historic (list or pd.Series): A time series of the
                                                portfolio's historical values.

    Returns:
        dict: A dictionary containing the calculated metrics:
            - "Annualized_Return": The average return on an annual basis.
            - "Annualized_Volatility": The annualized standard deviation of returns (risk).
            - "Max_Drawdown": The largest percentage drop from a peak to a subsequent trough.
            - "Calmar_Ratio": Annualized return divided by the max drawdown.
            - "Sortino_Ratio": Annualized return divided by the downside volatility.
            - "Win_Rate": The percentage of periods (days) with a positive return.
    """

    HOURS = 252 # Number of days per year
    data = pd.DataFrame()

    data['port_value'] = portfolio_historic.copy()
    data['hourly_port_returns'] = data.port_value.pct_change()
    data.dropna(inplace=True)

    # Calculate annualized mean and standard deviation
    hour_mean = data["hourly_port_returns"].mean()
    annual_mean = hour_mean * HOURS

    hour_vol = data["hourly_port_returns"].std()
    annual_vol = hour_vol * np.sqrt(HOURS)

    # Calculate Max Drawdown
    data["Cumulative_Max"] = data["port_value"].cummax()
    data["Drawdown"] = (data["Cumulative_Max"] - data["port_value"]) / data["Cumulative_Max"]
    max_drawdown = data["Drawdown"].max()

    # Calculate Calmar Ratio
    calmar_ratio = annual_mean / abs(max_drawdown) if max_drawdown !=0 else np.nan

    # Calculate Sortino Ratio and Downside Volatility
    negative_returns = data["hourly_port_returns"][data["hourly_port_returns"] < 0]
    downside_vol = negative_returns.std() * np.sqrt(HOURS)
    sortino_ratio = annual_mean / downside_vol if downside_vol != 0 else np.nan

    # Calculate Win Rate
    win_rate = (data['hourly_port_returns'] > 0).mean()

    metrics = {
        "Annualized_Return": float(annual_mean),
        "Annualized_Volatility": float(annual_vol),
        "Max_Drawdown": float(max_drawdown),
        "Calmar_Ratio": float(calmar_ratio),
        "Sortino_Ratio": float(sortino_ratio),
        "Win_Rate": float(win_rate)
    }

    return metrics