import pandas as pd
import numpy as np

ky=pd.read_csv('kyphosis.csv')

target=pd.get_dummies(ky['Kyphosis'],drop_first=True)

ky.drop('Kyphosis',axis=1,inplace=True)

ky=pd.concat([ky,target],axis=1)

x=ky.drop('present',axis=1)
y=ky['present']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)

ypre=lr.predict(xtest)

print('acc:',lr.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,ypre))
print('cr:',classification_report(ytest,ypre))