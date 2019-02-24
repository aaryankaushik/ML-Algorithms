import pandas as pd
import numpy as np

n=pd.read_csv('insurance.csv')

gen=pd.get_dummies(n['sex'],drop_first=True)
sm=pd.get_dummies(n['smoker'],drop_first=True)
n.drop(['sex','smoker'],axis=1,inplace=True)
n=pd.concat([n,gen,sm],axis=1)

x=n.drop(['region','expenses'],axis=1)
y=n['expenses']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(xtrain,ytrain)

ypre=lin.predict(xtest)

print('acc:',lin.score(xtest,ytest))

from sklearn.metrics import mean_squared_error,mean_absolute_error
print('mae:',mean_absolute_error(ytest,ypre))
print('mse:',mean_squared_error(ytest,ypre))
print('rmse:',np.sqrt(mean_squared_error(ytest,ypre)))