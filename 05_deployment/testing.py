
import requests

# 
host='churn-serving-env.eba-kuxcmdmf.us-east-2.elasticbeanstalk.com'
url=f'http://{host}/predict'

# 
customer= {"customerid": "7590-vhveg",
  "seniorcitizen": 0,
  "gender": "female",
  "partner": "yes",
  "dependents": "no",
  "tenure": 400,
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "monthlycharges": 19.85,
  "totalcharges": 1900.85}


response=requests.post(url,json=customer).json()
response


if response['churn']== True:
    print(f'Sending the Promotional Email to the Customer: {customer['customerid']}')
    
else:
    print(f'The Customer is loyal: {customer['customerid']}')

