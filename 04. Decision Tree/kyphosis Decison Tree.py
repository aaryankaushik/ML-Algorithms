import pandas as pd
ky=pd.read_csv('kyphosis.csv')

ab=pd.get_dummies(ky['Kyphosis'],drop_first=True)
ky=pd.concat([ky,ab],axis=1)

x=ky.drop(['Kyphosis','present'],axis=1)
y=ky['present']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()

dtree.fit(xtrain,ytrain)

ypred=dtree.predict(xtest)

print('DT acc',dtree.score(xtest,ytest))

from sklearn.ensemble import RandomForestClassifier
rn=RandomForestClassifier()
rn.fit(xtrain,ytrain)

rfp=rn.predict(xtest)

print('RF acc:',rn.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,ypred))
print('cr:',classification_report(ytest,ypred))