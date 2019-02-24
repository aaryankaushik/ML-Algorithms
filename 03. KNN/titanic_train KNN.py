import pandas as pd

tt=pd.read_csv('titanic_train.csv')

sex=pd.get_dummies(tt['Sex'],drop_first=True)
emb=pd.get_dummies(tt['Embarked'],drop_first=True)

tt.drop(['PassengerId','Name','Sex','Embarked','Ticket','Cabin'],axis=1,inplace=True)

tt=pd.concat([tt,sex,emb],axis=1)

import seaborn as sns
sns.boxplot(x=tt['Pclass'],y=tt['Age'])

def filage(a):
    Age=a[0]
    Pclass=a[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 30
        elif Pclass==3:
            return 25
    else:
        return Age

tt['Age']=tt[['Age','Pclass']].apply(filage,axis=1)

x=tt.drop('Survived',axis=1)
y=tt['Survived']

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
tt=sc.fit(tt)

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