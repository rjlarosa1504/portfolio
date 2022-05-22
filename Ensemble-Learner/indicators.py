"""
Author: Renzo La Rosa
Date: 03/25/2022

NOTE: This script is to be used for the author's personal use only. It is stored in the author's
GitHub portfolio and is for portfolio use only.

Description: Indicator definition

    - momentum: price n days ahead / current price
    - PPO : price percentage oscillator
    - RSI : relative strength index
    - EMA : exponential moving average
    - BBP : Bollinger Band Percentage

"""

import datetime as dt
import numpy as np
import pandas as pd
from util import get_data
import matplotlib.pyplot as plt

def __author__():
    return "Renzo_LaRosa"

def momentum(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 30), lookback = 12, plotGraph = True):
    sd_tmp = pd.to_datetime(sd) - dt.timedelta(days = lookback * 2)
    ed = pd.to_datetime(ed)
    symbol = [symbol]
    df_symbol_price = get_data(symbols=symbol, dates=pd.date_range(sd_tmp, ed))
    df_symbol_price.dropna(inplace = True)

    momentum = df_symbol_price / df_symbol_price.shift(lookback-1)
    momentum = momentum.loc[sd:]
    if plotGraph == True:
        fig, axs = plt.subplots(2, sharex = True)

        axs[0].plot(df_symbol_price, color = 'blue', label = "adj. close price")
        axs[0].set(ylabel=(symbol[0] + "Price ($)"))
        axs[0].grid()
        axs[0].set_title(symbol[0] + "Price and Momentum (" + str(lookback) + "-day lookback)")
        axs[0].legend()

        axs[1].plot(momentum, color = 'red', label = "momentum " + str(lookback) + "-day lookback")
        axs[1].set(ylabel = ("Momentum"))
        axs[1].grid()
        axs[1].legend()
        plt.xlabel("Date")
        plt.xticks(rotation = 45)

        plt.savefig("momentum.png", bbox_inches="tight")
    return momentum

def PPO(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 30), plotGraph = True):

    ema_12 = EMA(symbol = symbol, sd = sd, ed = ed, lookback = 12, plotGraph = False)
    ema_26 = EMA(symbol = symbol, sd = sd, ed = ed, lookback = 26, plotGraph = False)
    PPO = (ema_12 - ema_26)/ema_26 * 100

    lookback = 9

    multiplier = 2 / (lookback + 1)
    signal_line = PPO.rolling(window=lookback, min_periods=lookback).mean()

    for i in range(lookback+26, len(signal_line.values)):
        signal_line.values[i] = PPO.values[i] * multiplier + signal_line.values[i - 1] * (1 - multiplier)

    sd = pd.to_datetime(sd)
    ed = pd.to_datetime(ed)
    symbol = [symbol]
    df_symbol_price = get_data(symbols=symbol, dates=pd.date_range(sd, ed))
    df_symbol_price.dropna(inplace = True)

    if plotGraph == True:
        fig, axs = plt.subplots(2, sharex=True)

        axs[0].plot(df_symbol_price, color='blue', label="adj. close price")
        axs[0].set(ylabel=(symbol[0] + "Price ($)"))
        axs[0].grid()
        axs[0].set_title(symbol[0] + "Price and Percentage Price Oscillator(PPO)")
        axs[0].legend()

        axs[1].plot(PPO, color='red', label="PPO")
        axs[1].plot(signal_line, color = 'black', label = "signal line")
        axs[1].set(ylabel=("PPO and Signal Line"))
        axs[1].grid()
        axs[1].legend()
        plt.xlabel("Date")
        plt.xticks(rotation=45)

        plt.savefig("PPO.png", bbox_inches="tight")

    return PPO

def EMA(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 30), lookback = 20, plotGraph = True):
    sd_tmp = pd.to_datetime(sd) - dt.timedelta(days = lookback * 2)
    ed = pd.to_datetime(ed)
    symbol = [symbol]
    df_symbol_price = get_data(symbols=symbol, dates=pd.date_range(sd_tmp, ed))
    df_symbol_price.dropna(inplace = True)

    multiplier = 2 / (lookback + 1)
    ema = df_symbol_price.rolling(window=lookback, min_periods=lookback).mean()

    for i in range(lookback, len(ema.values)):
        ema.values[i] = df_symbol_price.values[i] * multiplier + ema.values[i-1] * (1 - multiplier)
    ema = ema.loc[sd:]
    if plotGraph == True:
        plt.figure()
        plt.plot(df_symbol_price, label = "adj. close price")
        plt.plot(ema, label = "EMA " + str(lookback) + "-day lookback")
        plt.grid()
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel(symbol[0] + "Price ($)")
        plt.title(symbol[0] + " Price and Exponential Moving Average " + str(lookback) + "-day lookback)")
        plt.xticks(rotation=45)
        plt.savefig("EMA.png", bbox_inches="tight")
    return ema


def RSI(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 30), lookback = 14, plotGraph = True):
    sd_tmp = pd.to_datetime(sd) - dt.timedelta(days = lookback * 2)
    ed = pd.to_datetime(ed)
    symbol = [symbol]
    df_symbol_price = get_data(symbols=symbol, dates=pd.date_range(sd_tmp, ed))
    df_symbol_price.dropna(inplace = True)

    daily_rets = df_symbol_price.copy()
    daily_rets.values[1:,:] = df_symbol_price.values[1:,:] - df_symbol_price.values[:-1,:]
    daily_rets.values[0,:] = np.nan
    up = daily_rets.clip(lower = 0)
    down = daily_rets.clip(upper = 0) * -1

    up_sma = up.rolling(window = lookback, min_periods = lookback).mean()
    down_sma = down.rolling(window = lookback, min_periods = lookback).mean()
    rsi = 100 * (up_sma) / (up_sma + down_sma)
    rsi = rsi.loc[sd:]
    if plotGraph == True:

        fig, axs = plt.subplots(2, sharex=True)

        axs[0].plot(df_symbol_price, color='blue', label="adj. close price")
        axs[0].set(ylabel=(symbol[0] + "Price ($)"))
        axs[0].grid()
        axs[0].set_title(symbol[0] + " Price and Relative Strength Index (RSI)")
        axs[0].legend()

        axs[1].plot(rsi, color='red', label="RSI")
        axs[1].axhline(y=70, color='black')
        axs[1].axhline(y=30, color='black')
        axs[1].set(ylabel=("RSI"))
        axs[1].grid()
        axs[1].legend()
        plt.xlabel("Date")
        plt.xticks(rotation=45)

        plt.savefig("RSI.png", bbox_inches="tight")

    return rsi


def BollingerBandPercentage(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 30), lookback = 20, plotGraph = True):
    # BBP (%) = (Price - Lower Band) / (Upper Band - Lower Band)
    # Default period for BB is 20 days
    # Default Stdev for BB is +/- 2sig
    # Overbought line is at 1
    # Oversold line is at 0
    sd_tmp = pd.to_datetime(sd) - dt.timedelta(days = lookback * 2)
    ed = pd.to_datetime(ed)
    symbol = [symbol]
    df_symbol_price = get_data(symbols=symbol, dates=pd.date_range(sd_tmp, ed))
    df_symbol_price.dropna(inplace = True)
    # df_symbol_price.drop(["SPY"], axis=1, inplace=True)

    sma = df_symbol_price.rolling(window = lookback, min_periods = lookback).mean()
    smstd = df_symbol_price.rolling(window = lookback, min_periods = lookback).std()

    upperBand = sma + 2 * smstd
    lowerBand = sma - 2 * smstd

    BBP = (df_symbol_price - lowerBand) / (upperBand - lowerBand)
    BBP = BBP.loc[sd:]
    if plotGraph == True:

        fig, axs = plt.subplots(2, sharex=True)

        axs[0].plot(df_symbol_price, color='blue', label="adj. close price")
        axs[0].plot(sma, color = "orange", label = "simple moving average")
        axs[0].plot(upperBand, color = "black", label = "Bollinger Bands")
        axs[0].plot(lowerBand, color = "black")
        axs[0].set(ylabel=(symbol[0] + "Price ($)"))
        axs[0].grid()
        axs[0].set_title(symbol[0] + " Price and Bollinger Band Percentage")
        axs[0].legend()

        axs[1].plot(BBP, color='red', label="BBP")
        axs[1].axhline(y=1, color='black')
        axs[1].axhline(y=0, color='black')
        axs[1].set(ylabel=("BBP"))
        axs[1].grid()
        axs[1].legend()
        plt.xlabel("Date")
        plt.xticks(rotation=45)

        plt.savefig("BBP.png", bbox_inches="tight")

    return BBP