import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import yaml
from datetime import datetime

def perform_rcv():
    train_data = pd.read_csv(".//data//processed//train_winequality.csv")
    test_data = pd.read_csv(".//data//processed//test_winequality.csv")

    train_x = train_data.drop("TARGET", axis = 1)
    test_x = test_data.drop('TARGET', axis = 1)

    train_y = train_data["TARGET"]
    test_y = test_data["TARGET"]

    params_path = 'hyper_param_testing//param_grid.yaml'

    param_config = read_params(params_path)

    param_grid = param_config["RandomForestRegressor"]

    x_train_scaled, x_test_scaled = standard_scaler(train_x, test_x)

    rcv_result = randomsearch_tuning(param_grid=param_grid, x_train=x_train_scaled, y_train=train_y, x_test=x_test_scaled,y_test=test_y )

    return rcv_result

def standard_scaler(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled


def randomsearch_tuning(param_grid , x_train, y_train, x_test, y_test):

    hptune_result = {}

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    regressor = RandomForestRegressor()

    rcv = RandomizedSearchCV(estimator = regressor, param_distributions = param_grid, n_iter = 100, cv = 5, verbose = 2)
    rcv.fit(x_train, y_train)

    best_params = rcv.best_params_
    
    best_estimator = rcv.best_estimator_
    y_pred = best_estimator.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    rcv_result = 'hyper_param_testing//rcv_result.yaml'
    
    params_and_metrics = {"best_params" : best_params, "metrics" : {"rmse" : float(rmse), "mae": float(mae), "r2" : float(r2)}}

    with open(rcv_result) as res:
        hptune_result = yaml.safe_load(res)
        hptune_result[timestamp] = params_and_metrics

    with open(rcv_result, 'w+') as res:
        yaml.dump(hptune_result, res)

    return hptune_result

def read_params(params_path):
    with open(params_path) as yaml_file:
        params_config = yaml.safe_load(yaml_file)
    return params_config

perform_rcv()




