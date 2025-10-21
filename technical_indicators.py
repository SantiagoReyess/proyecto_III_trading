import pandas as pd
import ta

def get_signals(df):

    # Momentum indicators
    ## RSI (7-day window)
    df["RSI_7"] = ta.momentum.rsi(close=df["Price"], window=7)    

    ## RSI (14-day window)
    df["RSI_14"] = ta.momentum.rsi(close=df["Price"], window=14)

    ## RSI (21-day window)
    df["RSI_21"] = ta.momentum.rsi(close=df["Price"], window=21)

    ## Awesome Oscillator
    df["Awesome_Osc"] = ta.momentum.awesome_oscillator(high=df["High"], low=df["Low"], window1=5, window2=34)

    ## Kaufman’s Adaptive Moving Average (KAMA)
    df["Kama"] = ta.momentum.kama(close=df["Price"], window=10, pow1=2, pow2=30)

    ## Rate of Change (ROC)
    df["ROC"] = ta.momentum.roc(close=df["Price"], window=12) 

    ## Stochastic Oscillator
    df["Stochastic_Osc"] = ta.momentum.stoch(high=df["High"], low=df["Low"], close=df["Price"], window=14, smooth_window=3) 

    ## Stochastic RSI
    df["Stochastic_RSI"] = ta.momentum.stochrsi(close=df["Price"], window=14, smooth1= 3, smooth2= 3)

    ## True Strength Index (TSI)
    df["TSI"] = ta.momentum.tsi(close=df["Price"], window_slow=25, window_fast=13)

    ## Ultimate Oscillator 
    df["Ultimate_Osc"] = ta.momentum.ultimate_oscillator(high=df["High"], low=df["Low"], close=df["Price"], 
                                                         window1=7, window2=14, window3=28, weight1=4.0, weight2=2.0, weight3=1.0) 

    # Volume Indicators
    ## Accumulation/Distribution Index (ADI)
    df["ADI"] = ta.volume.acc_dist_index(high=df["High"], low=df["Low"], close=df["Price"], volume=df["CVol"])

    ## Chaikin Money Flow
    df["CMF"] =  ta.volume.chaikin_money_flow(high=df["High"], low=df["Low"], close=df["Price"], volume=df["CVol"], window=20)

    ## Ease of Movement (EoM)
    #df["EoM"] = ta.volume.ease_of_movement(high=df["Price"], low=["Low"], volume=df["CVol"], window=14)

    ## Force Index (FI)
    df["FI"] =  ta.volume.force_index(close=df["Price"], volume=df["CVol"], window=13)

    ## Money FLow Index (MFI)
    df["MFI"] = ta.volume.money_flow_index(high=df["High"], low=df["Low"], close=df["Price"], volume=df["CVol"], window=14)

    ## Negative Volume Index (NVI)
    df["NVI"] = ta.volume.negative_volume_index(close=df["Price"], volume=df["CVol"])

    ## On Balance Volume (OBV)
    df["OBV"] =  ta.volume.on_balance_volume(close=df["Price"], volume=df["CVol"])

    # Volatility Indicators
    ## Average True Range (ATR)
    df["ATR"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Price"], window=14).average_true_range()
     
    ## Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(close=df['Price'], window=20, window_dev=2)

    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low']  = bb_indicator.bollinger_lband()
    df['BB_Mid']  = bb_indicator.bollinger_mavg()
    df['BB_Width'] = bb_indicator.bollinger_hband_indicator() - bb_indicator.bollinger_lband_indicator()

    ## Ulcer Index
    df["Ulcer"] = ta.volatility.UlcerIndex(close=df["Price"], window= 14).ulcer_index()

    df = df.dropna()

    return df