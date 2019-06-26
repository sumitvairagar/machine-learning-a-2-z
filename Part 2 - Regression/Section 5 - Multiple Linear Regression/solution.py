#Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


#Avoid Dummy Variable Trap
X = X[:,1:]

from sklearn.model_selection import train_test_split;
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=0);

#Fit the data in model Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X, train_Y)

#Prediction of the test set results
y_pred = regressor.predict(test_X)

#
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit()
regressor_OLS.summary() 

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit()
regressor_OLS.summary() 

X_opt = X[:, [0, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit()
regressor_OLS.summary() 

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3,  5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit()
regressor_OLS.summary()