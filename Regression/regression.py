""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Author: Renzo La Rosa
Date: 03/25/2022

NOTE: This script is to be used for the author's personal use only. It is stored in the author's
GitHub portfolio and is for portfolio use only.		  	   		  	  			  		 			     			  	 
""" 

import pandas as pd
import numpy as np
from scipy import rand
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("insurance.csv")
df = pd.get_dummies(df, columns = ["sex", "smoker", "region"])

Y = df["charges"].values
df.drop(columns = ["charges"], inplace = True)
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Simple Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.score(X_train, y_train))

plt.plot(y_train, reg.predict(X_train), 'o', color = 'black')
plt.xlabel("Actual Charges ($)")
plt.ylabel("Predicted Charges ($)")
plt.title("Simple Linear Regression")
plt.show()

# KNN Regression
from sklearn.neighbors import KNeighborsRegressor

knn_r_squared_arr = []
for i in range(2, 21, 2):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_r_squared_arr.append(knn.score(X_train, y_train))

plt.plot(range(2,21,2), knn_r_squared_arr)
plt.xlabel("KNN Number of Neighbors")
plt.ylabel("KNN R-squared")
plt.title("R-squared Using Different KNN Number of Neighbors")
plt.show()

# Optimal n_neighbors results in overfitting
# KNN not optimal since high results result from very low n_neighbors

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf_r_squared_arr = []
for i in range(2, 21, 2):
    randomForest = RandomForestRegressor(n_estimators =  30, min_samples_leaf = i, random_state = 42)
    randomForest.fit(X_train, y_train)
    rf_r_squared_arr.append(randomForest.score(X_train, y_train))

plt.plot(range(2,21,2), rf_r_squared_arr)
plt.xlabel("Random Forest Number of Leafs")
plt.ylabel("Random Forest R-squared")
plt.title("R-squared Using Different Random Forest Number of Leafs")
plt.show()