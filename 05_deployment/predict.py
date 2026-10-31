# Loading the model

import pickle
from flask import Flask
from flask import request, jsonify

model_file = f'model_c=0.01.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# customer={"customerid": "7590-vhveg",
#   "seniorcitizen": 0,
#   "gender": "female",
#   "partner": "yes",
#   "dependents": "no",
#   "tenure": 1,
#   "phoneservice": "no",
#   "multiplelines": "no_phone_service",
#   "internetservice": "dsl",
#   "onlinesecurity": "no",
#   "onlinebackup": "yes",
#   "deviceprotection": "no",
#   "techsupport": "no",
#   "streamingtv": "no",
#   "streamingmovies": "no",
#   "contract": "month-to-month",
#   "paperlessbilling": "yes",
#   "paymentmethod": "electronic_check",
#   "monthlycharges": 29.85,
#   "totalcharges": 29.85}



app=Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    
    customer=request.get_json()
    x=dv.transform([customer])
    y_pred= model.predict_proba(x)[0,1]
    churn=y_pred>0.5
    
    result= { "churn_probability": float(y_pred), 
             "churn": bool(churn)}
    
    return jsonify(result)


if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)