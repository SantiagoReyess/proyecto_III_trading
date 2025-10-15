import pandas as pd

def data_download(route):
    
    dataframe = pd.read_excel(route) #"PriceHistory.xlsx"
    dataframe = dataframe.sort_index(ascending=False)
    dataframe = dataframe.drop(columns=["Change", "% Change", "% Return", "Total Return (Gross)",
                                        "Cumulative Return %", "Cumulative Change %"])

    return dataframe
