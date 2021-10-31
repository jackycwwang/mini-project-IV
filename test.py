import numpy as np
import pandas as pd
import requests
import json

URL = 'http://127.0.0.1:5000/predict'
# URL = "http://ec2-3-143-221-43.us-east-2.compute.amazonaws.com:5000/predict"

df = pd.read_csv('data.csv')
sample = df.sample(1, replace=False)
sample = sample.drop(columns=['Loan_Status'])


# json_data = {'Loan_ID': 'LP001233',
#             'Gender': 'Male',
#             'Married': 'Yes',
#             'Dependents': '1',
#             'Education': 'Graduate',
#             'Self_Employed': 'No',
#             'ApplicantIncome': 10750,
#             'CoapplicantIncome': 0,
#             'LoanAmount': 312,
#             'Loan_Amount_Term': 360.0,
#             'Credit_History': 1.0,
#             'Property_Area': 'Urban'}
# json_data = request.get_json()

json_data = sample.to_json(orient='records')
json_data = json.loads(json_data)[0]


res = requests.post(url = URL, json=json_data)

if res.status_code == 200:
    print('...')
    print('request successful')
    print('...')
    
    result = res.json()[0]
    print(f'Estimated Score: {result[1]}\n')
    if result[0] > result[1]:
        print('Your loan application is disapproved. :-(')
    else:
        print('Congradulations! Your loan application is approved. :-)\n') 
    
else:
    print('request failed')