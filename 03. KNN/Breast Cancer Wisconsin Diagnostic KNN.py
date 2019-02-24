import pandas as pd

bc=pd.read_csv('Breast Cancer Wisconsin Diagnostic.csv')

dia=pd.get_dummies(bc['diagnosis'],drop_first=True)
bc.drop(['id','diagnosis','Unnamed: 32'],axis=1,inplace=True)
bc=pd.concat([bc,dia],axis=1)

x=bc.drop('M',axis=1)
y=bc['M']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)

ypre=knn.predict(xtest)

print('acc:',knn.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,ypre))
print('cr:',classification_report(ytest,ypre))

from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(xtrain,ytrain)
lgpre=lg.predict(xtest)

print('acc lg:',lg.score(xtest,ytest))

