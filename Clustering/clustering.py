""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Author: Renzo La Rosa
Date: 03/25/2022

NOTE: This script is to be used for the author's personal use only. It is stored in the author's
GitHub portfolio and is for portfolio use only.		  	   		  	  			  		 			     			  	 
""" 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("insurance.csv")

region = df["region"]
df = pd.get_dummies(df, columns = ["sex", "smoker", "region"])

# First plot shows if region & sex impact charges
plt.figure()
sns.scatterplot(x = df["sex_male"], y = df["charges"], hue = region)
plt.show()
# Plot didn't show anything particular, but can further analyze for more insights

#Standardizing data for PCA
scaler = StandardScaler()
df_scale = pd.DataFrame(data = scaler.fit_transform(df), columns = df.columns)
df_scale.drop(columns = ["sex_female", "sex_male", "region_southwest", "region_northwest", "region_northeast", "region_southeast"], inplace = True)
#PCA will reduce data dimensionality and along with cos2 value will help show
#which variables have most importance
pca = PCA(n_components = min(df_scale.shape))
pca.fit_transform(df_scale)
x_axis = np.arange(pca.n_components_) + 1
cum_sum_eigenvalues = np.cumsum(pca.explained_variance_ratio_)*100
plt.figure()
plots = sns.barplot(x = x_axis, y=pca.explained_variance_ratio_ * 100, color = "blue")
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.1f'),
        (bar.get_x() + bar.get_width() / 2,
        bar.get_height()), ha='center', va='center',
        size = 8, xytext = (0,8),
        textcoords='offset points')
plt.step(x_axis-1, cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.show()
#PCA shows that ideal number of components is 2
n_PC = 2
#COS2 - Variable Importance
strength = np.transpose(pca.components_) * np.sqrt(pca.explained_variance_)
cos2 = pd.DataFrame(np.square(strength))
PC_arr = []
for j in range(pca.n_components_):
    PC_arr.append("PC" + str(j+1))
cos2.columns = [PC_arr]
cos2.index = df_scale.columns

cos2_PCsUsed = cos2.iloc[:,:n_PC]
cos2_PCsUsed = pd.DataFrame(cos2_PCsUsed.sum(axis = 1), columns = ["Sum of PC Dim1-" + str(n_PC)])
cos2_PCsUsed = cos2_PCsUsed.sort_values(by = ["Sum of PC Dim1-" + str(n_PC)], ascending = False )
print(cos2_PCsUsed)
# The COS2 values states that smoker is the most important variable in clustering data

coordinates = pd.DataFrame(np.dot(df_scale, np.transpose(pca.components_)), index = df_scale.index)
coordinates = coordinates.iloc[:, :min(df_scale.shape)]
#Finding optimal number of clusters

# Starting with 2
kmeans = KMeans(n_clusters= 2, random_state=42)
kmeans.fit(coordinates)
sns.scatterplot(x = kmeans.labels_, y = df["charges"], hue = df["smoker_yes"])
plt.show()
# Plot shows that indeed Kmeans used smoking status as the primary differentiator in the data
# There is a noticeable difference in insurance charges between non-smokers and smokers. If you
# smoke, you will have larger insurance charges. Moving on to 3 clusters

kmeans = KMeans(n_clusters= 3, random_state=42)
kmeans.fit(coordinates)
sns.scatterplot(x = kmeans.labels_, y = df["charges"], hue = df["smoker_yes"])
plt.show()
# With 3 clusters, the non-smoking cluster split into 2

kmeans = KMeans(n_clusters= 4, random_state=42)
kmeans.fit(coordinates)
sns.scatterplot(x = kmeans.labels_, y = df["charges"], hue = df["smoker_yes"])
plt.show()
# With 4 clusters, the non-smoking clusters continue to split

kmeans = KMeans(n_clusters= 5, random_state=42)
kmeans.fit(coordinates)
sns.scatterplot(x = kmeans.labels_, y = df["charges"], hue = df["smoker_yes"])
plt.show()
# With 5 cluster, the smoking cluster finally split and it looks like there is a portion
# of smoking people that pay about the same as non-smoking people. Will cross reference
# with age and bmi as they are the next most important variables
df["cluster"] = kmeans.labels_
# df["age_bin"] = pd.cut(x = df["age"], bins = [0, 30, 40, 50, 60, 70, 75])
df["bmi_bin"] = pd.cut(x = df["bmi"], bins = [0, 22, 25, 28, 32, 35])

palette = ['green', 'lime', 'yellow', 'orange', 'red']
g = sns.FacetGrid(df, col="smoker_yes", hue="bmi_bin", palette = palette)
g.map(sns.scatterplot, "cluster", "charges", alpha=.7)
g.add_legend()
plt.show()
"""
Latest plot supports hypothesis that insurance charges are primarily associated
with smoking status. Additionally, insurance charges are higher with higher
bmi. The cluster results suggest the following:
- Stop smoking and decrease BMI to lower insurance costs
- If non-smoker and have low BMI and still have high insurance cost, time
  shop for other insurance
- If smoker and high BMI but not paying much insurance charges, the company
  should re-assess the risk they are taking
"""
