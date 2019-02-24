import pandas as pd

cd=pd.read_csv('College_Data.csv',index_col=0)

from sklearn.cluster import KMeans

model=KMeans(n_clusters=2)
model.fit(cd.drop(['Private'],axis=1))

print(model.cluster_centers_)
print(model.labels_)

def converter(private):
    if(private=='Yes'):
        return 1
    else:
        return 0

cd['Cluster']=cd['Private'].apply(converter)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(cd['Cluster'],model.labels_))
print(confusion_matrix(cd['Cluster'],model.labels_))
