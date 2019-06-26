#Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

"""from sklearn.model_selection import train_test_split;
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=0); """


#Linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

#Polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#Create new LinearRegression object
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

#Visualizing LinearRegression model.
plt.scatter(X, Y,  color = "red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#Visualizing the Polynomial model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y,  color = "red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#Predict result using Linear Regression
lin_reg.predict([[6.5]])


#Predict result using Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))