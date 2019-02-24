import pandas as pd
import numpy as np

ad=pd.read_csv('advertising.csv')

ad.drop(['Country','Timestamp','City','Ad Topic Line'],axis=1,inplace=True)

x=ad.drop('Clicked on Ad',axis=1)
y=ad['Clicked on Ad']

from sklearn.model_selection import train_test_split
## cross_validation
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)

ypre=lr.predict(xtest)

print('acc:',lr.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,ypre))
print('cr:',classification_report(ytest,ypre))
