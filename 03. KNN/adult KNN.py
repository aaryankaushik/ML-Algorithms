import pandas as pd
a=pd.read_csv('adult.csv')

a.isnull().any()

b=pd.get_dummies(a,prefix=['sex','salary'],
                 columns=['sex','salary'],drop_first=True)

b.drop(['workclass','fnlwgt','education','marital-status','occupation',
        'relationship','race','country','capital-gain',
        'capital-loss'],axis=1,inplace=True)
b.info()

x=b.drop('salary_ >50K',axis=1)
y=b['salary_ >50K']

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