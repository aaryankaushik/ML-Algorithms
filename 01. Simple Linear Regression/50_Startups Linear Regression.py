import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d=pd.read_csv('50_Startups.csv')
X=d[['R&D Spend']]
y=d[['Profit']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error,mean_squared_error
# Predicting the Test set results
y_pred = regressor.predict(X_test)
print('intercept : ',regressor.intercept_)
print('coefficients : ',regressor.coef_)
print('acc : ',regressor.score(X_test,y_test))
print('MAE : ',mean_absolute_error(y_test,y_pred))
print('MSE : ',mean_squared_error(y_test,y_pred))
print('RMSE : ',np.sqrt(mean_squared_error(y_test,y_pred)))

