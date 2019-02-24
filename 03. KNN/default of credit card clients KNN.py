import pandas as pd
c=pd.read_excel('default of credit card clients.xls')

c.isnull().any()
c.drop(c.index[:1], inplace=True) 
print(c.info())

c[:-1]= c[:-1].astype(int) 
print(c.info())
print(c.head(5))

x=c.drop('Y',axis=1)
y=c['Y']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)

ypre=knn.predict(xtest)

print('acc:',knn.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,ypre))
print('cf:',classification_report(ytest,ypre))
