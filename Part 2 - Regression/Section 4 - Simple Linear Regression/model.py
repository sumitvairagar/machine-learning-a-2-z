
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split;
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=1/3, random_state=0);

#Fit the data in model Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X, train_Y)

#Predicting Test result.
y_pred = regressor.predict(test_X)

#Plot the training data
plt.scatter(train_X, train_Y, color="red")
plt.plot(train_X, regressor.predict(train_X), color="blue")
plt.title("Salary Vs Experience")
plt.xlabel("Exprience")
plt.ylabel("Salary")
plt.show()


#Plot test data
plt.scatter(test_X, test_Y, color="red")
plt.plot(train_X, regressor.predict(train_X), color="blue")
plt.title("Salary Vs Experience")
plt.xlabel("Exprience")
plt.ylabel("Salary")
plt.show()