#first line creates the file in the trainer folder

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import argparse
import hypertune

def get_args_xgboost():
    """
        Function that will takes params from VertexAi configuration

        Returns:
            args (ArgumentParser): 
                corresponding params for xgboost
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate', required=True, type=float, help='learning rate')
    parser.add_argument(
        '--n_estimators', required=True, type=int, help='n_estimators')
    parser.add_argument(
        '--max_depth', required=True, type=int, help='max_depth')
    parser.add_argument(
        '--subsample', required=True, type=float, help='subsample')
    args = parser.parse_args()
    
    return args


def load_dataset():
    """
    Load the Boston housing dataset

    Returns:
        data (pd.DataFrame): 
            the housing dataset
    """
    
    data = pd.read_csv("/trainer/Housing.csv")
    return data

def process_dataset(data: pd.DataFrame):
    """Before training the model we need to process the data (missing values, labelize...)

        Args:
            data (pd.DataFrame): 
                the dataset we need to process
                

        Returns:
            x_train (pd.DataFrame):
                    training dataset
                    
            x_val (pd.DataFrame):
                validation dataset
                
            y_train (pd.Series): 
                label for training dataset
                
            y_val (pd.Series):
                label for validation dataset
    """
    
    # non tabular columns
    column_non_tabular = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
    encoder = LabelEncoder()
    
    for column in column_non_tabular:
        data[column] = encoder.fit_transform(data[column])

    target = data["price"]
    data = data.drop(["price"], axis=1)
    
    x_train, x_val, y_train, y_val = train_test_split(data, target, test_size = 0.3, random_state = 42)
    
        
    return x_train, x_val, y_train, y_val

def create_xgboost(n_estimators: int, max_depth: int, learning_rate: float, subsample: float):
    """
        init the xgboost regressor model with hyperparameters
        
        Args:
        
            n_estimators (int):
            max_depth (int):
            learning_rate (float):
            subsample (float):
            
        Returns:
        
            xgb (XGBRegressor):
                the model with custom params
        
    
    """
    
    xgb = XGBRegressor(n_estimators  = n_estimators, 
                       max_depth     =max_depth, 
                       learning_rate =learning_rate, 
                       subsample     =subsample)
    
    return xgb
    
    
def create_rfr(n_estimators: int, max_depth: int, min_samples_split: int, min_samples_leaf: int):
    """
        Init RandomForest model with corresponding params
        
        Args:
            n_estimators (int):
            
            max_depth (int):
            
            learning_rate (float):
            
            subsample (float):
        
        Returns:
        
            rfr (RandomForestRegressor):
                the model with custom params
        
    """
    rfr = RandomForestRegressor(n_estimators      = n_estimators, 
                                max_depth         = max_depth, 
                                min_samples_split = min_samples_split, 
                                min_samples_leaf  = min_samples_leaf)
    
    return rfr


def main():
    args_xgb = get_args_xgboost()
    data = load_dataset()
    x_train, x_val, y_train, y_val = process_dataset(data)
    
    xgb = create_xgboost(n_estimators  = args_xgb.n_estimators, 
                         max_depth     = args_xgb.max_depth, 
                         learning_rate = args_xgb.learning_rate, 
                         subsample     = args_xgb.subsample)  
    
    xgb.fit(x_train, y_train)

    pred_xgb = xgb.predict(x_val)
    hpt = hypertune.HyperTune()
    
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='R-squared (R²)',
        metric_value=r2_score(y_val, pred_xgb),)
    
if __name__ == '__main__':
    main()
