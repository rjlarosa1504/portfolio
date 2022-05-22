Author: Renzo La Rosa

Description: A stock market strategy that uses ensemble learning (random decision trees - i.e. random forest) and 
optimization parameters for trading purposes. 
The target (dependent) variable is created by looking "n" days ahead in the training data and it is cross-referenced
to the current price of the stock to determine whether the trader should trigger a "BUY" or "SELL" signal.
The independent variables are Bollinger Band Percentage (BBP), Relative Strength Index (RSI), and Percentage Price 
Oscillator (PPO). 
Optimization parameters vary and can be individually optmized for each stock.