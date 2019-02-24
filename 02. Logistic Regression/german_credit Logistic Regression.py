import pandas as pd

gc=pd.read_csv('german_credit.csv')

x=gc.drop('Creditability',axis=1)
y=gc['Creditability']

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