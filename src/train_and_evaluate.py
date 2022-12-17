## load the train and test data
## train algo
## save metrics and params

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from get_data import read_params
from urllib.parse import urlparse
import argparse
import joblib
import json 
import mlflow

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config['split_data']['train_path']
    test_data_path = config['split_data']['test_path']
    random_state = config['base']['random_state']
    model_dir = config['model_dir']
    web_model_dir = config["webapp_model_dir"]

    max_depth = config['estimators']['RandomForestRegressor']['params']['max_depth']
    min_samples_leaf = config['estimators']['RandomForestRegressor']['params']['min_samples_leaf']
    min_samples_split = config['estimators']['RandomForestRegressor']['params']['min_samples_split']
    n_estimators = config['estimators']['RandomForestRegressor']['params']['n_estimators']

    target = config['base']['target_col']

    train = pd.read_csv(train_data_path, sep=',')
    test = pd.read_csv(test_data_path, sep=',')

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis = 1)
    test_x = test.drop(target, axis = 1)

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name = mlflow_config["run_name"]) as mlops_run:

        rfr = RandomForestRegressor(
            max_depth= max_depth,
            min_samples_leaf= min_samples_leaf,
            min_samples_split = min_samples_split,
            n_estimators = n_estimators,
            random_state= random_state
        )
        rfr.fit(train_x, train_y)

        predicted_qualities = rfr.predict(test_x)

        joblib.dump(rfr, web_model_dir)

        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)
        print(f"RandomForestRegressor model (max_depth = {max_depth}, min_samples_leaf = {min_samples_leaf}, min_samples_split = {min_samples_split}, n_estimators = {n_estimators})")
        print("RMSE : {}".format(rmse))
        print("MAE : {}".format(mae))
        print("R2 Score : {}".format(r2))

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("n_estimators", n_estimators)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                rfr, 
                'model', 
                registered_model_name = mlflow_config["registered_model_name"])

        else:
            mlflow.sklearn.load_model(rfr, 'model')


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