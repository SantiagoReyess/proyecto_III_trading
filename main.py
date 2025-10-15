from data_download import data_download
from technical_indicators import get_signals

def main():

    data = data_download("PriceHistory.xlsx")

    data = get_signals(data)

    return print(data)

if __name__ == "__main__":
    main()