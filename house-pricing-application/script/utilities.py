import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from google.cloud import storage
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

BUCKET_ID     = "trans-sunset-439207-f2-house-pricing"
XGBOOST_MODEL = "xgboost-model/xgboost_model.bst"

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

def load_xgb_model_from_bucket():
    """Load xgboost model from GCP Bucket

    Returns:
        loaded_xgb (XGBRegressor): 
            the pretrained model loaded
    """
    
    loaded_xgb = XGBRegressor()

    # local model
    #loaded_xgb.load_model('static/xgboost_model.bst')  # Load from JSON
    
    # model from GCP Bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_ID)
    blob = bucket.blob(XGBOOST_MODEL)
    
    model_path = "/tmp/xgboost_model.bst"
    blob.download_to_filename(model_path)
    
    loaded_xgb.load_model(model_path)
    
    return loaded_xgb

def load_xgb_model_locally():
    """Load xgboost model from GCP Bucket

    Returns:
        loaded_xgb (XGBRegressor): 
            the pretrained model loaded
    """
    
    loaded_xgb = XGBRegressor()
    
    loaded_xgb.load_model('static/xgboost_model.bst')
    
    return