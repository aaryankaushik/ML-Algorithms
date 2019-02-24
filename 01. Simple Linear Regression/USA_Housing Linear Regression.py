import pandas as pd
import numpy as np
f=pd.read_csv('USA_Housing.csv')
x=f.drop(['Price','Address'],axis=1)
y=f['Price']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)

ypre=model.predict(xtest)
score=model.score(xtest,ytest)
print('score:',score)

from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE:', mean_absolute_error(ytest,ypre))
print('MSE:', mean_squared_error(ytest,ypre))
print('RMSE:',np.sqrt(mean_squared_error(ytest,ypre)))