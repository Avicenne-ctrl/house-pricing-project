# House Pricing API

In this folder, I developed a script in order to get the xgboost model stored in a Google Cloud bucket and use it in a Flask API deployed with Google Run.

Here is a detailed explaination for each file :   

```script/utilities```  
- Py function to load the Xgboost model stored in the Google Cloud bucket
- Py function to predict the pricing given an user query

```script/update_model```  
- Py function to train xgboost model locally
- Py function to save the bst model in the bucket
- Py function to load the tensorflow dataset
  
```script/xgboost_model.bst```  
- bst file

```templates/```  
- index and results html web pages

```app.py```
- the API Flask 

```how_to_deploy_app```
- notebook with instruction to deploy the app container on Cloud Run




