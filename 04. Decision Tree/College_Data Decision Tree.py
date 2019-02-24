import pandas as pd
cd=pd.read_csv('College_Data.csv')

p=pd.get_dummies(cd['Private'],drop_first=True)

cd=pd.concat([cd,p],axis=1)

cd.isnull().any()

x=cd.drop(['Name','Private','Yes'],axis=1)
y=cd['Yes']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(xtrain,ytrain)

ypre=dt.predict(xtest)

print('acc:',dt.score(xtest,ytest))

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100)
rf.fit(xtrain,ytrain)

rfp=rf.predict(xtest)

print('acc:',rf.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,ypre))
print('cr:',classification_report(ytest,ypre))