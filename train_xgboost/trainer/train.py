#first line creates the file in the trainer folder

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy
import configparser
from google.cloud import storage
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing


JOB_NAME  = "house-pricing-train-job"
PROJECT   = "trans-sunset-439207-f2"
LOCATION  = "us-central1"
BUCKET_ID = "trans-sunset-439207-f2-house-pricing"

parser = argparse.ArgumentParser()
parser.add_argument('--l_r', dest='lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--n_estimators', dest='n_estimators', default=100, type=int, help='n_estimators')
parser.add_argument( '--max_depth',dest='max_depth',  default=6, type=int, help='max_depth')
parser.add_argument( '--subsample', dest='subsample', default=0.8, type=float, help='subsample')

args = parser.parse_args()


def save_model_to_bucket(model_trained : XGBRegressor):
    """Save the trained model to the bucket

    Args:
        model_trained (XGBRegressor): 
            trained model
    """
    
    # save it to the bucket
    bst = model_trained.get_booster()
    bst.save_model('/tmp/xgboost_model.bst')
    storage_client = storage.Client()

    bucket = storage_client.bucket(BUCKET_ID)
    blob = bucket.blob("xgboost-model/model_xgboost.bst")

    # Uploader le fichier local vers le blob
    blob.upload_from_filename('/tmp/xgboost_model.bst')
    


def create_xgboost(n_estimators: int, 
                   max_depth: int, 
                   learning_rate: float, 
                   subsample: float):
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




def train_xgboost(x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series):
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
        
        
    # get params from bucket
    l_r, n_estimators, max_d, subsamples = get_hyperparams_xgboost()
    
    xgb = create_xgboost(n_estimators, max_d, l_r, subsamples) 
    xgb.fit(x_train, y_train)
    pred = xgb.predict(x_val)
    score = r2_score(y_val, pred)
    
    return xgb

def load_data():
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )

    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", 
        "TAX", "PTRATIO", "B", "LSTAT"
    ]

    # Convertir les ensembles de données en DataFrames pandas
    return pd.DataFrame(x_train, columns=columns), pd.Series(y_train), pd.DataFrame(x_test, columns=columns), pd.Series(y_test)


with strategy.scope():
  # Creation of dataset, and model building/compiling need to be within
  # `strategy.scope()`.
  xgboost = create_xgboost(args.n_estimators, args.max_depth, args.l_r, args.subsample)
    
x_train, y_train, x_val, y_val = load_data()

# train and save the new model
xgb_trained = train_xgboost(x_train, x_val, y_train, y_val)

save_model_to_bucket(xgb_trained)
    


