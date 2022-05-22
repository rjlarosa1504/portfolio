""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Author: Renzo La Rosa
Date: 03/25/2022

NOTE: This script is to be used for the author's personal use only. It is stored in the author's
GitHub portfolio and is for portfolio use only.		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt

def get_portfolio_metrics(portvals):
    cr = (portvals.iloc[-1] - portvals.iloc[0])/portvals.iloc[0]
    dr= (portvals/ portvals.shift(1)) - 1
    dr = dr[1:]
    adr= dr.mean()
    stddr= dr.std()
    sr = np.sqrt(252) * adr / stddr

    return cr.iloc[0], adr.iloc[0], stddr.iloc[0], sr.iloc[0]

def compute_portvals(orders_dataframe, symbol, start_val=100000, commission=9.95, impact=0.005):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    symbolList = [symbol] * len(orders_dataframe.index)
    orderList = []

    for i in range(len(orders_dataframe.index)):
        if (orders_dataframe["Shares"].iloc[i] >= 0):
            orderList.append("BUY")
        else:
            orderList.append("SELL")

    orders_dataframe.insert(0, "Order", orderList)
    orders_dataframe.insert(0, "Symbol", symbolList)
    orders_dataframe["Shares"] = orders_dataframe["Shares"].abs()
    # print(orders_dataframe)
    df_orders = orders_dataframe.reset_index()
    df_orders.sort_values(by="Date", inplace = True)
    df_orders["Date"]= pd.to_datetime(df_orders["Date"])

    start_date = df_orders.iloc[0,0]
    end_date = df_orders.iloc[-1,0]
    # print(start_date)
    # print(end_date)
    symbols = df_orders["Symbol"].unique()
    # print(symbols)
    df_prices = get_data(symbols, pd.date_range(start_date, end_date))
    df_prices.dropna(inplace = True)
    df_prices["CASH"] = 1

    # print("df_prices")
    # print(df_prices)
    # print("df_orders")
    # print(df_orders)

    for i in df_orders.index:
        if df_orders["Date"].iloc[i] not in df_prices.index:
            nonTradingDay = True
            while nonTradingDay:
                if i == (len(df_orders.index) - 1):
                    df_orders["Date"].iloc[i] = df_orders["Date"].iloc[i] - dt.timedelta(days=1)
                    if df_orders["Date"].iloc[i] in df_prices.index:
                        nonTradingDay = False
                else:
                    df_orders["Date"].iloc[i] = df_orders["Date"].iloc[i] + dt.timedelta(days = 1)
                    if df_orders["Date"].iloc[i] in df_prices.index:
                        nonTradingDay = False
    df_trades = df_prices.copy()
    df_holdings = df_prices.copy()
    for symbol in df_trades:
        df_trades[symbol] = 0
        df_holdings[symbol] = 0


    for index, order in df_orders.iterrows():
        df_tmp = df_trades[df_trades.index == order["Date"]]
        df_tmp_price = df_prices[df_prices.index == order["Date"]]

        if order["Order"] == "BUY":
            # buying...
            df_tmp[order["Symbol"]] = df_tmp[order["Symbol"]] + order["Shares"]
            df_tmp["CASH"] = df_tmp["CASH"] + (order["Shares"] * (1+impact)*(df_tmp_price[order["Symbol"]]) * -1) - commission
        else:
            # selling...
            df_tmp[order["Symbol"]] = df_tmp[order["Symbol"]] + (order["Shares"] * -1)
            df_tmp["CASH"] = df_tmp["CASH"] + (order["Shares"] * (1-impact)*df_tmp_price[order["Symbol"]]) - commission
        df_trades[df_trades.index == order["Date"]] = df_tmp

    for i in range(len(df_trades.index)):
        if (i == 0):
            df_holdings.iloc[i, :-1] = df_trades.iloc[i, :-1]
            df_holdings.iloc[i, -1] = start_val + df_trades.iloc[i,-1]
        else:
            df_holdings.iloc[i,:-1] = df_holdings.iloc[(i-1), :-1] + df_trades.iloc[i, :-1]
            df_holdings.iloc[i, -1] = df_holdings.iloc[(i-1), -1] + df_trades.iloc[i,-1]

    df_values = df_holdings * df_prices
    portvals = df_values.sum(axis = 1)

    portvals = pd.DataFrame(index=portvals.index, data=portvals.values)
  		  	   		  	  			  		 			     			  	 
    return portvals

def author():
    return "Renzo_LaRosa"

def plot_portfolio(portvals):
    plt.plot(portvals.index, portvals.values, label="Portfolio")
    plt.legend()
    plt.show()

def test_code():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Helper function to test code  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		  	  			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		  	  			  		 			     			  	 
    # Define input parameters  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # of = "./orders/orders-01.csv"
    of = "./additional_orders/orders.csv"
    sv = 1000000  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Process orders  		  	   		  	  			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):  		  	   		  	  			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		  	  			  		 			     			  	 
    else:  		  	   		  	  			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		  	  			  		 			     			  	 

    # plot_portfolio(portvals)
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		  	  			  		 			     			  	 
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_metrics(portvals)

    df_orders = pd.read_csv(of)
    df_orders.sort_values(by="Date", inplace=True)
    df_orders["Date"] = pd.to_datetime(df_orders["Date"])

    start_date = df_orders.iloc[0, 0]
    end_date = df_orders.iloc[-1, 0]
    symbols = np.array(["$SPX"], dtype=object)

    df_prices = get_data(symbols, pd.date_range(start_date, end_date))
    df_SPY = df_prices["$SPX"]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_metrics(df_SPY)
  		  	   		  	  			  		 			     			  	 
    # Compare portfolio against $SPX  		  	   		  	  			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		  	  			  		 			     			  	 
    print(f"Sharpe Ratio of SPX : {sharpe_ratio_SPY}")
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		  	  			  		 			     			  	 
    print(f"Cumulative Return of SPX : {cum_ret_SPY}")
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		  	  			  		 			     			  	 
    print(f"Standard Deviation of SPX : {std_daily_ret_SPY}")
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		  	  			  		 			     			  	 
    print(f"Average Daily Return of SPX : {avg_daily_ret_SPY}")
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    test_code()  		  	   		  	  			  		 			     			  	 
