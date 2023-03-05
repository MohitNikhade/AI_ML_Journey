#----------Import Libraries----------#

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#----------Load the Dataset----------#

df = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print(df)

#----------Visualize the Dataset----------#

plt.scatter(df['Hours'], df['Scores'])
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

#----------Split the Data into Training and Testing sets----------#

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#----------Train the Model----------#

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept for model is:",model.intercept_)
print("Coefficient of model is:",model.coef_)
a = model.intercept_
b = model.coef_
c = float(input("Enter Hours you studied for predicting what your score can be "))
Y_pred = b + a*c
print(Y_pred)

#----------Visualize the regression line----------#

plt.scatter(X_train, y_train)
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Hours vs Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

#----------Make Predictions----------#

y_pred = model.predict(X_test)

#----------Evaluate the model----------#

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('Coefficient of determination:', r2)
