import numpy as np
import pandas as pd

def calculate_metrics(portfolio_historic):
    """
        Calcula métricas de desempeño financiero a partir del valor histórico de un portafolio.

        Esta función estima métricas anualizadas de desempeño
        con base en una serie temporal de valores de portafolio.

        Parameters
        ----------
        portfolio_historic :
            Serie temporal con los valores históricos del portafolio

        Returns
        -------
        dict
            Diccionario con las métricas calculadas:

            - **Annualized_Return** : float
              Retorno promedio anualizado del portafolio.
            - **Annualized_Volatility** : float
              Volatilidad anualizada de los rendimientos horarios.
            - **Max_Drawdown** : float
              Máxima caída desde un pico hasta un valle en el valor del portafolio.
            - **Calmar_Ratio** : float
              Relación entre el retorno anualizado y el máximo drawdown.
            - **Sortino_Ratio** : float
              Relación entre el retorno anualizado y la volatilidad de las pérdidas (downside risk).
            - **Win_Rate** : float
              Proporción de rendimientos horarios positivos.

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