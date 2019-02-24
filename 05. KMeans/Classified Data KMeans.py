import pandas as pd
cl=pd.read_csv('Classified Data.csv',index_col=0)

from sklearn.cluster import KMeans
km=KMeans(n_clusters=2)
km.fit(cl.drop(['TARGET CLASS'],axis=1))

print(km.cluster_centers_)
p=(km.labels_)

from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(cl['TARGET CLASS'],km.labels_))
print(confusion_matrix(cl['TARGET CLASS'],km.labels_))