import pandas as pd
import numpy as np

cd=pd.read_csv('Classified Data.csv')
cd.drop('Unnamed: 0',axis=1,inplace=True)

x=cd.drop(['TARGET CLASS'],axis=1)
y=cd['TARGET CLASS']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(xtrain,ytrain)

ypred=lg.predict(xtest)

print('acc : ',lg.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,ypred))
print('cr:',classification_report(ytest,ypred))