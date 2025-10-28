import pandas as pd
import matplotlib.pyplot as plt


def data_download(route):
    
    dataframe = pd.read_excel(route) #"PriceHistory.xlsx"
    dataframe = dataframe.drop(columns=["Change", "% Change", "% Return", "Total Return (Gross)",
                                        "Cumulative Return %", "Cumulative Change %"])

    plt.figure(figsize=(14, 7))
    plt.plot(dataframe["Date"], dataframe["Price"],  color="darkblue")
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

    return dataframe