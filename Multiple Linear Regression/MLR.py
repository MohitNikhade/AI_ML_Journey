import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#----------Importing the Dataset-----------#

dataset = pd.read_csv('K:/Datasets/Multivariant_Linear_Regression/archive/50_Startups.csv')
print(dataset)

#----------Finding length(no. of rows) of dataset-----------#

len(dataset)

#----------Finding shape(no. of rows and columns) of dataset-----------#

dataset.shape

#----------Plotting----------#

plt.scatter(dataset['Marketing Spend'], dataset['Profit'], alpha = 0.5)
plt.title('Scatter plot of Profit with Marketing Spend')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()

plt.scatter(dataset['R&D Spend'], dataset['Profit'], alpha = 0.5)
plt.title('Scatter plot of Profit with R&D Spend')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()

plt.scatter(dataset['Administration'], dataset['Profit'], alpha = 0.5)
plt.title('Scatter plot of Profit with Administration')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.show()

#----------Create the figure object----------#

ax = dataset.groupby(['State'])['Profit'].mean().plot.bar(figsize = (10, 5), fontsize = 14)

#----------Set the Title----------#
ax.set_title('Average profit for different states where the startups operate', fontsize = 20)

#----------Set x and y axis labels----------#

ax.set_xlabel("State", fontsize = 15)
ax.set_ylabel("Profit", fontsize = 15)

dataset.State.value_counts()

#----------Create dummy variables for the categorical variable state----------#

dataset['NewYork_State'] = np.where(dataset['State']=='New York', 1, 0)
dataset['California_State'] = np.where(dataset['State']=='California', 1, 0)
dataset['Florida_State'] = np.where(dataset['State']=='Florida', 1, 0)

#----------Now we will drop the original column state from dataframe----------#

dataset.drop(columns=['State'], axis=1, inplace=True)

dataset

#----------Declaring Dependent Variable----------#

dependent_variable = 'Profit'

dependent_variable

#----------Create a list of independent variables----------#

independent_variables = dataset.columns.tolist()

independent_variables.remove(dependent_variable)

independent_variables

#----------Create the data of Independent variables----------#

X = dataset[independent_variables].values

#----------Create the data of dependent variables----------#

y = dataset[dependent_variable].values

#----------Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train[0:10]

#----------Fitting Multiple Linear Regression to the Training Set----------#


regressor = LinearRegression()
regressor.fit(X_train, y_train)

#----------Predicting the test set results----------#

y_pred = regressor.predict(X_test)

y_pred

RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
RMSE

MAE = mean_absolute_error(y_test, y_pred)
MAE

MSE = mean_squared_error(y_test, y_pred)
MSE

R_sq = r2_score(y_test, y_pred)
R_sq
