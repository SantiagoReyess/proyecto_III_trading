from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_data_drift(X_train, X_test, feature_names, alpha=0.05):
    """
    Aplica la prueba KS para comparar distribuciones entre train y test.
    Devuelve un DataFrame con el estadístico KS y el p-value para cada feature.
    """
    drift_results = []
    for i, col in enumerate(feature_names):
        # Extraemos la columna correspondiente
        train_col = X_train[:, i]
        test_col = X_test[:, i]

        # Removemos NaN e infinitos (pueden venir de % changes)
        train_col = train_col[np.isfinite(train_col)]
        test_col = test_col[np.isfinite(test_col)]

        if len(train_col) == 0 or len(test_col) == 0:
            continue

        # KS test
        ks_stat, p_val = ks_2samp(train_col, test_col)
        drift_results.append({
            "Feature": col,
            "KS_Statistic": ks_stat,
            "P_Value": p_val,
            "Drift": "Sí" if p_val < alpha else "No"
        })

    drift_df = pd.DataFrame(drift_results).sort_values(by="KS_Statistic", ascending=False)
    return drift_df


def temporal_drift_analysis(df, feature_cols, baseline_size=0.3, window_size=0.1, alpha=0.05):
    """
    Analiza el data drift de forma temporal con ventanas móviles.

    Parámetros:
    -----------
    df : DataFrame con tus features (ordenado temporalmente)
    feature_cols : lista de columnas a analizar
    baseline_size : proporción inicial usada como referencia (ej. 0.3 = 30%)
    window_size : tamaño de cada ventana deslizante como proporción (ej. 0.1 = 10%)
    alpha : nivel de significancia para el KS test

    Devuelve:
    ----------
    drift_results : DataFrame con drift promedio por ventana y por feature
    """

    n = len(df)
    base_end = int(n * baseline_size)
    window_len = int(n * window_size)
    baseline = df[feature_cols].iloc[:base_end]

    drift_records = []

    # Ventanas móviles posteriores
    for start in range(base_end, n - window_len, window_len):
        end = start + window_len
        current_window = df[feature_cols].iloc[start:end]

        if isinstance(df.index[start], (np.datetime64, pd.Timestamp)):
            window_label = f"{pd.to_datetime(df.index[start]).date()} → {pd.to_datetime(df.index[end - 1]).date()}"
        else:
            window_label = f"{start} → {end - 1}"

        drift_window = {}
        drift_window["Window"] = window_label
        drift_window["Start_Index"] = start
        drift_window["End_Index"] = end

        drift_flags = []

        for col in feature_cols:
            base_col = baseline[col].dropna()
            current_col = current_window[col].dropna()
            if len(base_col) < 10 or len(current_col) < 10:
                continue
            ks_stat, p_val = ks_2samp(base_col, current_col)
            drift_window[col] = ks_stat
            drift_flags.append(p_val < alpha)

        drift_window["Drift_Features_%"] = np.mean(drift_flags) * 100
        drift_records.append(drift_window)

    drift_df = pd.DataFrame(drift_records)
    drift_df.set_index("Window", inplace=True)
    return drift_df

