""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Author: Renzo La Rosa
Date: 03/25/2022

NOTE: This script is to be used for the author's personal use only. It is stored in the author's
GitHub portfolio and is for portfolio use only.		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import indicators
import RTLearner as rt
import BagLearner as bl
from util import get_data, plot_data
import random
import marketsimcode
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class StrategyLearner(object):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  	  			  		 			     			  	 
    :type verbose: bool  		  	   		  	  			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  	  			  		 			     			  	 
    :type impact: float  		  	   		  	  			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  	  			  		 			     			  	 
    :type commission: float  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    # constructor  		  	   		  	  			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """  		  	   		  	  			  		 			     			  	 
        Constructor method  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        self.verbose = verbose  		  	   		  	  			  		 			     			  	 
        self.impact = impact  		  	   		  	  			  		 			     			  	 
        self.commission = commission  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		  	  			  		 			     			  	 
    def author(self):
        return "rrosa30"

    def add_evidence(
        self,  		  	   		  	  			  		 			     			  	 
        symbol="IBM",  		  	   		  	  			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		  	  			  		 			     			  	 
        sv=10000,  		  	   		  	  			  		 			     			  	 
    ):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		  	  			  		 			     			  	 
        :type symbol: str  		  	   		  	  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
        :type sd: datetime  		  	   		  	  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			     			  	 
        :type ed: datetime  		  	   		  	  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			     			  	 
        :type sv: int  		  	   		  	  			  		 			     			  	 
        """
        n = 5
        ed_tmp = pd.to_datetime(ed) + dt.timedelta(days = 2*n)
        # Get Indicators
        BBP = indicators.BollingerBandPercentage(symbol=symbol, sd=sd, ed=ed, lookback=20, plotGraph=False)
        RSI = indicators.RSI(symbol=symbol, sd=sd, ed=ed, lookback=14, plotGraph=False)
        PPO = indicators.PPO(symbol=symbol, sd=sd, ed=ed, plotGraph=False)

        df_symbol_price = get_data(symbols=[symbol], dates=pd.date_range(sd, ed_tmp))
        df_symbol_price.dropna(inplace = True)

        # prices_SPY = df_symbol_price["SPY"]  # only SPY, for comparison later
        # df_symbol_price.drop(["SPY"], axis=1, inplace=True)

        yTarget = np.ones(shape = (len(BBP.values), 1)) * -999
        yBUY = 0.025
        ySELL = -0.05
        for i in range(len(yTarget)):
            retBUY = df_symbol_price.values[i + n]/ (df_symbol_price.values[i] * (1 + self.impact)) - 1
            retSELL = df_symbol_price.values[i + n]/ (df_symbol_price.values[i] * (1 - self.impact)) - 1
            if retBUY > yBUY:
                yTarget[i] = 1
            elif retSELL < ySELL:
                yTarget[i] = -1
            else: #CASH
                yTarget[i] = 0

        xVariables = np.concatenate((BBP.values, RSI.values, PPO.values), axis = 1)

        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size": 10}, bags = 30, boost = False, verbose = False)
        self.learner.add_evidence(xVariables, yTarget)

  		  	   		  	  			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		  	  			  		 			     			  	 
    def testPolicy(  		  	   		  	  			  		 			     			  	 
        self,  		  	   		  	  			  		 			     			  	 
        symbol="IBM",  		  	   		  	  			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		  	  			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		  	  			  		 			     			  	 
        sv=10000,  		  	   		  	  			  		 			     			  	 
    ):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  	  			  		 			     			  	 
        :type symbol: str  		  	   		  	  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
        :type sd: datetime  		  	   		  	  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			     			  	 
        :type ed: datetime  		  	   		  	  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			     			  	 
        :type sv: int  		  	   		  	  			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		  	  			  		 			     			  	 
        """

        BBP = indicators.BollingerBandPercentage(symbol=symbol, sd=sd, ed=ed, lookback=20, plotGraph=False)
        RSI = indicators.RSI(symbol=symbol, sd=sd, ed=ed, lookback=14, plotGraph=False)
        PPO = indicators.PPO(symbol=symbol, sd=sd, ed=ed, plotGraph=False)
        momentum = indicators.momentum(symbol = symbol, sd=sd, ed=ed, lookback = 12, plotGraph = False)

        xVariables = np.concatenate((BBP.values, RSI.values, PPO.values, momentum.values), axis=1)
        yQuery = self.learner.query(xVariables)

        df_orders = pd.DataFrame(data=({"Date": [sd], "Shares": [0]}))

        tradePosition = 0
        for i in range(len(yQuery)):
            if (i == len(yQuery)- 1):
                # Last i
                df_tmp = pd.DataFrame(
                    data={"Date": [BBP.index[i]], "Shares": [0]})
                df_orders = df_orders.append(df_tmp)
            elif (tradePosition == 0):
                # tradePosition will only be 0 if no other trades have happened yet
                if (yQuery[i] == 1):
                    #First BUY
                    df_tmp = pd.DataFrame(
                        data={"Date": [BBP.index[i]], "Shares": [1000]})
                    df_orders = df_orders.append(df_tmp)
                    tradePosition = 1
                if (yQuery[i] == -1):
                    #First SELL
                    df_tmp = pd.DataFrame(
                        data={"Date": [BBP.index[i]], "Shares": [-1000]})
                    df_orders = df_orders.append(df_tmp)
                    tradePosition = -1
            else:
                if (tradePosition == 1) and (yQuery[i] == -1):
                    # currently in BUYING position and want to go to SELLING position
                    df_tmp = pd.DataFrame(
                        data={"Date": [BBP.index[i]], "Shares": [-2000]})
                    df_orders = df_orders.append(df_tmp)
                    tradePosition = -1
                elif (tradePosition == -1) and (yQuery[i] == 1):
                    # currently in SELLING position and want to go to BUYING position
                    df_tmp = pd.DataFrame(
                        data={"Date": [BBP.index[i]], "Shares": [2000]})
                    df_orders = df_orders.append(df_tmp)
                    tradePosition = 1
                # else: tradePosition = 1 and yQuery[i] = 1 or 0 => do nothing
                #       tradePosition = -1 and yQuery[i] = -1 or 0 => do nothing

        df_orders = df_orders.iloc[1:, :]
        df_orders.index = df_orders["Date"]
        df_orders.drop(["Date"], axis=1, inplace=True)

        return df_orders
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("One does not simply think up a strategy")

    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    start_val = 100000

    learner = StrategyLearner()
    learner.add_evidence(symbol = symbol, sd = sd, ed = ed, sv = start_val)
    df_trades = learner.testPolicy(symbol = symbol, sd = sd, ed = ed, sv = start_val)
    # print(df_trades)
    portvals = marketsimcode.compute_portvals(orders_dataframe = df_trades, symbol = symbol, start_val=start_val, commission=9.95, impact=0.005)
    # print(portvals)
    
    plt.plot(portvals)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.show()

    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    # learner.add_evidence(symbol = symbol, sd = sd, ed = ed, sv = start_val)

    df_trades = learner.testPolicy(symbol = symbol, sd = sd, ed = ed, sv = start_val)
    portvals = marketsimcode.compute_portvals(orders_dataframe = df_trades, symbol = symbol, start_val=start_val, commission=9.95, impact=0.005)
    plt.plot(portvals)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.show()
