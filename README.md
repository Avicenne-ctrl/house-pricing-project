# house-pricing-project

# Vertex Ai Hyperparams Tuning, Model Training and deploying API Flask Prediction on Google Cloud

The goal of this repo is to explain how to fine tune hyper parameters for 2 models (xgboost and RandomForest) and then train it on Vertex Ai, in order to deploy these models for real-time prediction.
Let’s explain the steps of this project thanks to this schema : 

￼

- We will create container for hypertune task for each model ( f« house-pricing-finetune-{model}»)
- Launch the CustomJob and save the best hyperparams on the destination bucket
- Then we will create new container for training task for each model ( f « {model}-train »), in which we will import the hyperparams from the bucket
- Launch the CustomJob and save the model.bst on the destination bucket
- Finally we can create the container application in which we will use the model.bst model from the bucket in order to make prediction
- Deploy the API Flask app on Cloud Run, here is the link of the app if you want to try it : 



However I’m facing some issue when I try to launch a CustomJob : 

￼

Instead, I trained the 2 models locally on my computer and then download the weights (model.bst) and the params (config.ini) manually on the bucket  We will still keep the hypertune and train containers.

￼

How to use this repo : 

Git clone

Install GKP
Install VKP

However you can’t use the app as it is because you need Admin authorization. Create your own Google Cloud account and project, then follow the step on each ReadMe.

Then you can refer to the ReadMe of each sub folder to understand how to create hypertuning, training CustomJob
