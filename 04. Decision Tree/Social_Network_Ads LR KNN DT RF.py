import pandas as pd
sn=pd.read_csv('Social_Network_Ads.csv')

gen=pd.get_dummies(sn['Gender'],drop_first=True)
sn=pd.concat([sn,gen],axis=1)

x=sn.drop(['User ID','Gender','Purchased'],axis=1)
y=sn['Purchased']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
lrpe=lr.predict(xtest)
print('lr acc:',lr.score(xtest,ytest))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)
knpre=knn.predict(xtest)
print('knn acc:',knn.score(xtest,ytest))

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
dtpre=dt.predict(xtest)
print('dt acc:',dt.score(xtest,ytest))

from sklearn.ensemble import RandomForestClassifier
rn=RandomForestClassifier()
rn.fit(xtrain,ytrain)
rnpre=rn.predict(xtest)
print('ran acc:',rn.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:',confusion_matrix(ytest,knpre))
print('cr:',classification_report(ytest,knpre))
