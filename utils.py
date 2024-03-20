import pandas as pd
# map coordinates to features
def convert_dataset(data,features,func):
  result = func(data,features)
  _right = features.copy()
  _right['idx'] = _right.index

  new_train = pd.merge(data,result,how='left',left_on='id',right_on='idx_x')
  data = pd.merge(new_train,_right,how='left',left_on='idx_y',right_on='idx')

  data = data.drop(['id','idx_x','idx_y','idx','lat_y','lon_y'],axis=1)

  return data

# map points by closed points
def map_closed_points(data,features):
  _left = data.loc[:,['lat', 'lon']].copy()
  _left['idx'] = _left.index

  _right = features.loc[:,['lat', 'lon']].copy()
  _right['idx'] = _right.index


  df_dist = pd.merge(left=_left,
          right=_right,
          how='cross')

  df_dist['dist'] = ((df_dist.lat_x - df_dist.lat_y)**2 + (df_dist.lon_x - df_dist.lon_y)**2)**0.5


  result = df_dist.loc[df_dist.groupby('idx_x')['dist'].idxmin()]
  result[['idx_x', 'idx_y', 'dist']]
  return result[['idx_x', 'idx_y']].copy()

