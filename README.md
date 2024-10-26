# house-pricing-project

# Vertex Ai Hyperparams Tuning, Model Training and deploying API Flask Prediction on Google Cloud

The goal of this repo is to explain how to fine tune hyper parameters for 2 models (xgboost and RandomForest) and then train it on Vertex Ai, in order to deploy these models for real-time prediction.
Let’s explain the steps of this project thanks to this schema : 

![First](images-readme/first-schema.png)

- We will create a container for the hyperparameter tuning task for each model.
- Start the CustomJob and save the best hyperparameters to the destination bucket.
- Then, we will create a new container for the training task for each model, importing the hyperparameters from the previous bucket.
- Start the CustomJob and save the ```model.bst``` file to the destination bucket.
- Finally, we create the application container, using the ```model.bst``` file from the bucket to make predictions.
- Deploy the Flask API app on Cloud Run. Here is the link to the app if you’d like to try it:


However I’m facing some issue when I try to start a CustomJob : 

![Issue](images-readme/issue.png)


Instead, I focused on training a single model (XGBoost) locally on my computer, then manually uploaded the weights (`model.bst`) and parameters (`config.ini`) to the bucket. We will still keep the containers for hyperparameter tuning and training.

![Issue](images-readme/solution.png)
￼
How to use this repo : 

```Git clone https://github.com/Avicenne-ctrl/house-pricing-project.git```

However, you won’t be able to use the app directly, as it requires Admin authorization. To get started, create your own Google Cloud account and project, then follow the steps provided in each Notebook subfolder.






