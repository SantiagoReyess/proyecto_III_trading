import pandas as pd
import matplotlib.pyplot as plt


def data_download(route):
    """
        Loads historical price data from an Excel file and performs initial cleaning.

        This function reads an Excel file specified by the file path, assumes it
        contains financial time-series data, and removes several columns that are
        not needed for the primary analysis (e.g., percentage changes, cumulative
        returns). The function also contains commented-out code for a basic price plot.

        Args:
            route (str): The file path to the Excel file containing the price history.
                         Example: "PriceHistory.xlsx".

        Returns:
            pd.DataFrame: A pandas DataFrame containing the cleaned data, with
                          unnecessary columns removed.
        """

    dataframe = pd.read_excel(route) #"PriceHistory.xlsx"
    dataframe = dataframe.drop(columns=["Change", "% Change", "% Return", "Total Return (Gross)",
                                        "Cumulative Return %", "Cumulative Change %"])

    # graph stored in directory
    #plt.figure(figsize=(14, 7))
    #plt.plot(dataframe["Date"], dataframe["Price"],  color="darkblue")
    #plt.grid()
    #plt.xlabel("Date")
    #plt.ylabel("Price")
    #plt.show()

    return dataframe