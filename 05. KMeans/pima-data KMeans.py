import pandas as pd
a=pd.read_csv('pima-data.csv',index_col=0)

from sklearn.cluster import KMeans

model=KMeans(n_clusters=2)
model.fit(a.drop(['diabetes',],axis=1))

print(model.cluster_centers_)
print(model.labels_)

def converter(diabetes):
    if(diabetes==True):
        return 1
    else:
        return 0

a['Cluster']=a['diabetes'].apply(converter)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(a['Cluster'],model.labels_))
print(confusion_matrix(a['Cluster'],model.labels_))

