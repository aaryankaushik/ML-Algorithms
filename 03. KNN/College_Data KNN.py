import pandas as pd

cd=pd.read_csv('College_Data.csv')

a=pd.get_dummies(cd['Private'],drop_first=True)
cd=pd.concat([cd,a],axis=1)
cd.drop(['Private','Name'],axis=1,inplace=True)

x=cd.drop('Yes',axis=1)
y=cd['Yes']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)

ypre=knn.predict(xtest)

print('acc knn:',knn.score(xtest,ytest))

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)

lrpre=lr.predict(xtest)

print('acc logistic:',lr.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,ypre))
print('cf:',classification_report(ytest,ypre))
