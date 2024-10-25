# house-pricing-application

house-pricing-application :

Access the app thanks to this url : 


Use this repo :

clone git

install requirements.txt

python app.py



script : 
    utilities = prediction


    update_model = - get hyperparams from aiplatform.HyperparameterTuningJob stored in bucket
                    - if better score
                    - get the new params
                    - update xgboost/ randomforest

deploy the prediction app

static : 
 - save model/ load the weights from buckets instead
 - save csv/ BigQuery ? / already created a dataset 

template :
html display

app.py = API flask to predict a pricing

Dockerfile for the container
Push it to Google artefact registry
Publish app = Google Cloud Run

Update model = Google Function Run




