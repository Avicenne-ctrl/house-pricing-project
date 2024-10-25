from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import sys
import os
sys.path.append('..')
import script.utilities as ut
import script.update_model as update_model


# if no model created
if not os.path.exists("static/xgboost_model.json"):
    print("training model")
    update_model.main_xgboost()

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret_key"

@app.route("/")
@app.route('/', methods=['GET', 'POST'])

def index():
    
    if request.method == 'POST':
        
        query_dict = {
                    "CRIM":    float(request.form.get('CRIM', 0)),
                    "ZN":      float(request.form.get('ZN', 0)),
                    "INDUS":   float(request.form.get('INDUS', 0)),
                    "CHAS":    float(request.form.get('CHAS', 0)),
                    "NOX":     float(request.form.get('NOX', 0)),
                    "RM":      float(request.form.get('RM', 0)),
                    "AGE":     float(request.form.get('AGE', 0)),
                    "DIS":     float(request.form.get('DIS', 0)),
                    "RAD":     float(request.form.get('RAD', 0)),
                    "TAX":     float(request.form.get('TAX', 0)),
                    "PTRATIO": float(request.form.get('PTRATIO', 0)),
                    "B":       float(request.form.get('B', 0)),
                    "LSTAT":   float(request.form.get('LSTAT', 0)),
                }

    
                
        xgb = ut.load_xgb_model_from_bucket()
        # xgb = ut.load_xgb_model_locally()
        
        price = f"{int(ut.make_prediction(pd.DataFrame([query_dict]), xgb))}$"
        
        return render_template('resultats.html', prediction_value= price)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="8080", threaded=False)


