import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

d = pd.read_csv('titanic_train.csv')

sns.heatmap(d.isnull(),cmap='viridis',yticklabels=False,cbar=False)

sns.set_style('whitegrid')
sns.boxplot(x='Pclass',y='Age',data=d)

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        elif Pclass==3:
            return 24
    else:
        return Age
    
d['Age']=d[['Age','Pclass']].apply(impute_age,axis=1)

d.drop('Cabin',axis=1,inplace=True)
d.dropna(inplace=True)

sex=pd.get_dummies(d['Sex'],drop_first=True)
embarked=pd.get_dummies(d['Embarked'],drop_first=True)

d.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
#inplace used for permanent change

d=pd.concat([d,sex,embarked],axis=1)

X=d.drop('Survived',axis=1)
y=d['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
l=LogisticRegression()
l.fit(X_train, y_train)

y_pred = l.predict(X_test)

print('acc : ',l.score(X_test,y_test))

from sklearn.metrics import confusion_matrix,classification_report
print('cf',confusion_matrix(y_test,y_pred))
print('cr',classification_report(y_test,y_pred))
