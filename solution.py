import pandas as pd

from features import load_transform_features
from train import get_model
from utils import convert_dataset, map_closed_points
def load_transform_dataset():
    
    test = pd.read_csv("test.csv")

    features = load_transform_features()
    data = convert_dataset(test,features,map_closed_points)
    
    return data

def generate_predictions():
    # features = apply_pca(features)
    # data = convert_dataset(train,features,map_closed_points)

    # features,model_pca = apply_pca(features)
    # features,model_cluster = cluster(features)
    from catboost import CatBoostRegressor
    
    data = load_transform_dataset()

    model = get_model()
    y_pred = model.predict(data)
    
    data['score'] = y_pred
    return data

def save_csv(data):
    data[['id','score']].to_csv('submission_sample.csv')


if __name__=="__main__":
    
    data = generate_predictions()
    save_csv(data)

    