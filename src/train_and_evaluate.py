## load the train and test data
## train algo
## save metrics and params

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json 

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config['split_data']['train_path']
    test_data_path = config['split_data']['test_path']
    random_state = config['base']['random_state']
    model_dir = config['model_dir']

    alpha = config['estimators']['ElasticNet']['params']['alpha']
    l1_ratio = config['estimators']['ElasticNet']['params']['l1_ratio']

    target = config['base']['target_col']

    train = pd.read_csv(train_data_path, sep=',')
    test = pd.read_csv(test_data_path, sep=',')

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis = 1)
    test_x = test.drop(target, axis = 1)

    enr = ElasticNet(
        alpha= alpha,
        l1_ratio= l1_ratio,
        random_state = random_state
    )
    enr.fit(train_x, train_y)

    predicted_qualities = enr.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)
    print("ElasticNet model (alpha = {}, l1_ratio = {}):".format(alpha,l1_ratio))
    print("RMSE : {}".format(rmse))
    print("MAE : {}".format(mae))
    print("R2 Score : {}".format(r2))

    params_file = config['reports']['params']
    scores_file = config['reports']['scores']

    with open(params_file, 'w') as f:
        scores = {
            "RMSE" : rmse,
            "MAE" : mae,
            "R2_SCORE" : r2
        }
        json.dump(scores, f, indent=4)
    
    with open(scores_file, 'w') as f:
        params = {
            "alpha" : alpha,
            "l1_ratio" : l1_ratio
        }
        json.dump(params, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(enr, model_path)


def eval_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)

    return rmse, mae, r2



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config) 
    