import pandas as pd
df=pd.read_csv('kyphosis.csv')

from sklearn.cluster import KMeans
km=KMeans(n_clusters=2)
km.fit(df.drop(['Kyphosis'],axis=1))

print(km.cluster_centers_)
print(km.labels_)

def conv(kyphosis):
    if(kyphosis=='present'):
        return 1
    else:
        return 0
    
df['Cluster']=df['Kyphosis'].apply(conv)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(df['Cluster'],km.labels_))
print(confusion_matrix(df['Cluster'],km.labels_))