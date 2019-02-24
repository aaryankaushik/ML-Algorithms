import pandas as pd

sn=pd.read_csv('Social Network Ads.csv')

gen=pd.get_dummies(sn['Gender'],drop_first=True)
sn.drop(['Gender'],axis=1,inplace=True)
sn=pd.concat([sn,gen],axis=1)

x=sn.drop('Purchased',axis=1)
y=sn['Purchased']

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