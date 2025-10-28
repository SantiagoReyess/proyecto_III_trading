from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_data_drift(X_train, X_test, feature_names, alpha=0.05):
    """
    Analyzes data drift between training and testing sets using the KS test.

    This function iterates through each feature, comparing its distribution in the
    training set (X_train) against the testing set (X_test) using the
    two-sample Kolmogorov-Smirnov (KS) test. It returns a summary DataFrame
    indicating whether a statistically significant drift was detected for each feature.

    Args:
        X_train (np.ndarray): The training data, where columns represent features.
        X_test (np.ndarray): The testing data, with the same feature columns.
        feature_names (list of str): A list of names for the features.
        alpha (float, optional): The significance level to determine drift.
                                 If the p-value is less than alpha, drift is
                                 considered significant. Defaults to 0.05.

    Returns:
        pd.DataFrame: A DataFrame with the results of the drift analysis,
                      containing columns for "Feature", "KS_Statistic", "P_Value",
                      and "Drift" (Yes/No). The DataFrame is sorted by the
                      KS statistic in descending order.
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
    Analyzes data drift over time using a moving window approach.

    This function establishes a baseline dataset from the initial portion of the
    DataFrame. It then creates subsequent, non-overlapping windows of data and
    compares the distribution of each feature in each window against the baseline
    using the KS test. This is useful for identifying when the statistical
    properties of the data change over time.


    Args:
        df (pd.DataFrame): The input DataFrame containing time-series data.
                           Must include a 'Date' column.
        feature_cols (list of str): The names of the feature columns to analyze for drift.
        baseline_size (float, optional): The proportion of the initial data to use
                                         as the stable baseline. Defaults to 0.3 (30%).
        window_size (float, optional): The proportion of data to use for each
                                       subsequent analysis window. Defaults to 0.1 (10%).
        alpha (float, optional): The significance level for the KS test. Defaults to 0.05.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a time window.
                      Columns include the KS statistic for each feature in that
                      window and the overall percentage of features that drifted.
                      The index is set to the start date of the window.
    """


    # Asegurarse de que 'Date' sea datetime
    df['Date'] = pd.to_datetime(df['Date'])

    n = len(df)
    base_end = int(n * baseline_size)
    window_len = int(n * window_size)
    baseline = df[feature_cols].iloc[:base_end]

    drift_records = []

    # Ventanas móviles posteriores
    for start in range(base_end, n - window_len, window_len):
        end = start + window_len
        current_window = df[feature_cols].iloc[start:end]

        # Guardamos la fecha de cada de la ventana
        window_start_date = df['Date'].iloc[start]

        drift_window = {}
        drift_window["Window_Start"] = window_start_date
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
    drift_df.set_index("Window_Start", inplace=True)  # usamos la fecha como índice
    return drift_df

