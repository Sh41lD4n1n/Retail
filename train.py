from catboost import CatBoostRegressor
import pandas as pd

from features import load_transform_features
from utils import convert_dataset, map_closed_points
def load_transform_dataset():
    train = pd.read_csv("train.csv")
    

    features = load_transform_features()
    data = convert_dataset(train,features,map_closed_points)
    return data

def get_model():
    data = load_transform_dataset()
    model = CatBoostRegressor(learning_rate=0.5, depth=10, loss_function='RMSE')
    fit_model = model.fit(data.drop('score',axis=1),data['score'])

    return fit_model


if __name__=="__main__":
    
    get_model()

    