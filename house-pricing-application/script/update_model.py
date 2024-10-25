from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
import sys
sys.path.append("..")
import script.utilities as ut


JOB_NAME = "house-pricing-hyperparam-job"
PROJECT  = "trans-sunset-439207-f2"
LOCATION = "us-central1"

# By default

SCORE_XGB         = 60
N_ESTIMATORS_XGB  = 100
MAX_DEPTH_XGB     = 6
LEARNING_RATE     = 0.1
SUBSAMPLE         = 0.8

SCORE_RFR         = 54
N_ESTIMATORS_RFR  = 100
MAX_DEPTH_RFR     = 6
MIN_SAMPLES_SPLIT = 3
MIN_SAMPLES_LEAF  = 10


def get_and_save_hyerparams():
    
    aiplatform.init(project=PROJECT, location=LOCATION)
    hpt_job = aiplatform.HyperparameterTuningJob.get(
        resource_name=JOB_NAME,
    )
    
    return hpt_job


def create_xgboost(n_estimators: int    = N_ESTIMATORS_XGB, 
                   max_depth: int       = MAX_DEPTH_XGB, 
                   learning_rate: float = LEARNING_RATE, 
                   subsample: float     = SUBSAMPLE)-> XGBRegressor:
    """
    create the xgboost regressor model with hyperparameters
    
    Args:
        n_estimators (int):
        
        max_depth (int):
        
        learning_rate (float):
        
        subsample (float):
        
        
    Returns:
        xgb (XGBRegressor):
            model init with hyperparameters

    """
    xgb = XGBRegressor(n_estimators= n_estimators, 
                       max_depth=max_depth, 
                       learning_rate=learning_rate, 
                       subsample=subsample)
    
    return xgb

def create_rfr(n_estimators: int       = N_ESTIMATORS_RFR,
               max_depth: int          = MAX_DEPTH_RFR,
               min_samples_split: int  = MIN_SAMPLES_SPLIT
               , min_samples_leaf: int = MIN_SAMPLES_LEAF)-> RandomForestRegressor:
    """
        Args:
            n_estimators (int):
            
            max_depth (int):
            
            min_samples_split (int):

            min_samples_leaf (int):
        
        Returns:
            rfr (RandomForestRegressor):
                model init with hyperparameters
    """
    rfr = RandomForestRegressor(n_estimators= n_estimators, 
                                max_depth= max_depth, 
                                min_samples_split= min_samples_split, 
                                min_samples_leaf= min_samples_leaf)
    
    return rfr
    
    


def train_and_save_xgboost(x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series)->XGBRegressor:
    """
        train and then save the xgboost model

        Args:
            x_train (pd.DataFrame):
                    training dataset
                    
            x_val (pd.DataFrame):
                validation dataset
                
            y_train (pd.Series): 
                label for training dataset
                
            y_val (pd.Series):
                label for validation dataset

        Returns:
            XGBRegressor: 
                trained model
        
        Raise:
        ------
            - ValueType Error
            - if input are not numpy.array
    """
    
    if not isinstance(x_train, pd.DataFrame) or not isinstance(x_val, pd.DataFrame) or not isinstance(y_train, pd.Series) or not isinstance(y_val, pd.Series):
        raise TypeError(f"Wrong type for data, expected pd.DataFrame or pd.Series got x_train: {type(x_train).__name__}, "
                        f"x_val: {type(x_val).__name__}, y_train: {type(y_train).__name__}, "
                        f"y_val: {type(y_val).__name__}")
        
    xgb = create_xgboost() 
    xgb.fit(x_train, y_train)
    pred = xgb.predict(x_val)
    score = r2_score(y_val, pred)
    
    print(f"r2 score :{score}")
    
    # xgb.save_model('static/xgboost_model.json')
    
    bst = xgb.get_booster()
    bst.save_model('./static/xgboost_model.bst')
    
    return xgb

def train_and_save_randomforest(x_train: numpy.array, x_val: numpy.array, y_train: numpy.array, y_val: numpy.array)->RandomForestRegressor:
    """
        train and then save the randomforest model

        Args:
            x_train (pd.DataFrame):
                    training dataset
                    
            x_val (pd.DataFrame):
                validation dataset
                
            y_train (pd.Series): 
                label for training dataset
                
            y_val (pd.Series):
                label for validation dataset

        Returns:
            XGBRegressor: 
                trained model
        
        Raise:
        ------
            - ValueType Error
            - if input are not numpy.array
    """
    
    if not isinstance(x_train, pd.DataFrame) or not isinstance(x_val, pd.DataFrame) or not isinstance(y_train, pd.Series) or not isinstance(y_val, pd.Series):
        raise TypeError(f"Wrong type for data, expected pd.DataFrame or pd.Series got x_train: {type(x_train).__name__}, "
                        f"x_val: {type(x_val).__name__}, y_train: {type(y_train).__name__}, "
                        f"y_val: {type(y_val).__name__}")

    rfr = create_rfr() 

    rfr.fit(x_train, y_train)

    pred = rfr.predict(x_val)
    score = r2_score(y_val, pred)
    
    rfr.save_model('static/rfr_model.json')
    
    return rfr

def main_xgboost():
    
    # get new params from hypertune, they will be saved in the config.ini file
    #get_and_save_hyerparams()
    
    # load data    
    x_train, y_train, x_val, y_val = ut.load_data()
    
    # train and save the new model
    train_and_save_xgboost(x_train, x_val, y_train, y_val)
    

if __name__ == '__main__':
    main_xgboost()