import pandas as pd

ad=pd.read_csv('audit_risk trial.csv')

ad.isnull().any()
ad.fillna(value=0,inplace=True)

print(ad.info())

x=ad.drop(['Risk','LOCATION_ID'],axis=1)
y=ad['Risk']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)

ypre=knn.predict(xtest)

print('acc:',knn.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf:', confusion_matrix(ytest,ypre))
print('cf:', classification_report(ytest,ypre))