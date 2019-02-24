import pandas as pd

ad=pd.read_csv('advertising.csv')

x=ad.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'],axis=1)
y=ad['Clicked on Ad']

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