from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

# @title Apply pca function

def apply_pca(features):
  n = 11
  pca = PCA(n_components=n)
  data = pca.fit_transform(features.iloc[:,2:])
  features = features.drop([str(i) for i in range(363)],axis=1)

  data = pd.DataFrame(data,columns=range(n))

  features = pd.concat([features,data],axis=1)
  return features,pca



def cluster(features):
  features.columns = features.columns.astype(str)

  cluster = KMeans(n_clusters=15,n_init='auto')
  d1 = cluster.fit_predict(features)
  features['cluster'] = d1



  return features,cluster

def load_transform_features():
    features = pd.read_csv("features.csv")

    features,model_pca = apply_pca(features)
    features,model_cluster = cluster(features)

    return features

