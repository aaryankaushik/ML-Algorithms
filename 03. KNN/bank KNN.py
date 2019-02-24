import pandas as pd
bn=pd.read_csv('bank.csv',sep=';')

bn.isnull().any()

a=pd.get_dummies(bn, prefix=['House', 'Loan','Y'],
                 columns=['housing', 'loan','y'],drop_first=True)

a.drop(['job','marital','education','default',
        'contact','month','poutcome'],axis=1,inplace=True)

x=a.drop(['Y_yes','day','duration'],axis=1)
y=a['Y_yes']

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