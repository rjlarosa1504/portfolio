"""

LaRosa-Renzo: This script has been copied from the Georgia Tech Machine Learning Course


MLT: Utility code.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Copyright 2017, Georgia Tech Research Corporation  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332-0415  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import os  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import pandas as pd  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def symbol_to_path(symbol, base_dir=None):  		  	   		  	  			  		 			     			  	 
    """Return CSV file path given ticker symbol."""  		  	   		  	  			  		 			     			  	 
    base_dir = os.getcwd()		  	   		  	  			  		 			     			  	 
    return base_dir + '\\' + symbol + ".csv"	  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def get_data(symbols, dates, addSPY=False, colname="Adj Close"):  		  	   		  	  			  		 			     			  	 
    """Read stock data (adjusted close) for given symbols from CSV files."""  		  	   		  	  			  		 			     			  	 
    df = pd.DataFrame(index=dates)  		  	   		  	  			  		 			     			  	 
    if addSPY and "SPY" not in symbols:  # add SPY for reference, if absent  		  	   		  	  			  		 			     			  	 
        symbols = ["SPY"] + list(  		  	   		  	  			  		 			     			  	 
            symbols  		  	   		  	  			  		 			     			  	 
        )  # handles the case where symbols is np array of 'object'  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    for symbol in symbols:  		  	   		  	  			  		 			     			  	 
        df_temp = pd.read_csv(  		  	   		  	  			  		 			     			  	 
            symbol_to_path(symbol),  		  	   		  	  			  		 			     			  	 
            index_col="Date",  		  	   		  	  			  		 			     			  	 
            parse_dates=True,  		  	   		  	  			  		 			     			  	 
            usecols=["Date", colname],  		  	   		  	  			  		 			     			  	 
            na_values=["nan"],  		  	   		  	  			  		 			     			  	 
        )  		  	   		  	  			  		 			     			  	 
        df_temp = df_temp.rename(columns={colname: symbol})  		  	   		  	  			  		 			     			  	 
        df = df.join(df_temp)  		  	   		  	  			  		 			     			  	 
        if symbol == "SPY":  # drop dates SPY did not trade  		  	   		  	  			  		 			     			  	 
            df = df.dropna(subset=["SPY"])  		  	   		  	  			  		 			     			  	 
  	# df.dropna(inplace = True)	  	   		  	  			  		 			     			  	 
    return df  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):  		  	   		  	  			  		 			     			  	 
    import matplotlib.pyplot as plt  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    """Plot stock prices with a custom title and meaningful axis labels."""
    """DO NOT use in code submitted for grading"""  		  	   		  	  			  		 			     			  	 
    ax = df.plot(title=title, fontsize=12)  		  	   		  	  			  		 			     			  	 
    ax.set_xlabel(xlabel)  		  	   		  	  			  		 			     			  	 
    ax.set_ylabel(ylabel)  		  	   		  	  			  		 			     			  	 
    plt.show()  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def get_orders_data_file(basefilename):  		  	   		  	  			  		 			     			  	 
    return open(  		  	   		  	  			  		 			     			  	 
        os.path.join(  		  	   		  	  			  		 			     			  	 
            os.environ.get("ORDERS_DATA_DIR", "orders/"), basefilename  		  	   		  	  			  		 			     			  	 
        )  		  	   		  	  			  		 			     			  	 
    )  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def get_learner_data_file(basefilename):  		  	   		  	  			  		 			     			  	 
    return open(  		  	   		  	  			  		 			     			  	 
        os.path.join(  		  	   		  	  			  		 			     			  	 
            os.environ.get("LEARNER_DATA_DIR", "Data/"), basefilename  		  	   		  	  			  		 			     			  	 
        ),  		  	   		  	  			  		 			     			  	 
        "r",  		  	   		  	  			  		 			     			  	 
    )  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def get_robot_world_file(basefilename):  		  	   		  	  			  		 			     			  	 
    return open(  		  	   		  	  			  		 			     			  	 
        os.path.join(  		  	   		  	  			  		 			     			  	 
            os.environ.get("ROBOT_WORLDS_DIR", "testworlds/"), basefilename  		  	   		  	  			  		 			     			  	 
        )  		  	   		  	  			  		 			     			  	 
    )  		  	   		  	  			  		 			     			  	 
