import pandas as pd
cl=pd.read_csv('Classified Data.csv',index_col=0)

x=cl.drop('TARGET CLASS',axis=1)
y=cl['TARGET CLASS']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()

dtree.fit(xtrain,ytrain)

ypred=dtree.predict(xtest)

print('DT acc:',dtree.score(xtest,ytest))

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100)
rf.fit(xtrain,ytrain)

rfp=rf.predict(xtest)

print('RF acc:',rf.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,ypred))
print('cr:',classification_report(ytest,ypred))