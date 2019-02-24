import pandas as pd
dataset=pd.read_csv('loan_data.csv')

p=pd.get_dummies(dataset['purpose'],drop_first=True)
dataset=pd.concat([dataset,p],axis=1)

x=dataset.drop(['purpose','not.fully.paid'],axis=1)
y=dataset['not.fully.paid']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(xtrain,ytrain)

ypre=dtree.predict(xtest)

print('DT acc:',dtree.score(xtest,ytest))

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(xtrain,ytrain)

ypre=rfc.predict(xtest)

print('RF acc:',dtree.score(xtest,ytest))
 
from sklearn.metrics import classification_report,confusion_matrix
print('Classsification Matrix',classification_report(ytest,ypre))
print('Confusion Matrix',confusion_matrix(ytest,ypre))