#Import the required plugins
from flask import Flask, request, jsonify
from flask_restful import Api, Resource 
import pickle
import numpy as np
import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
 
app = Flask(__name__)
api = Api(app)


# Customer Transformer 
# class DataframeTransformer():
#     def __init__(self, func):
#         self.func = func
        
#     def fit(self, X, y=None, **fit_params):
#         return self
    
#     def transform(self, input_df, **transform_params):
#         return self.func(input_df)

#     def fit_transform(self, X, y=None, **fit_params):
#         self.fit(X)
#         return self.transform(X)

def process_dataframe(df):
    df = df.astype({'Loan_Amount_Term': 'category',
                    'Credit_History': 'category',
                    'Self_Employed': 'category',
                    'Dependents': 'category',
                    'Gender': 'category',
                    'Credit_History': 'category',
                    'LoanAmount': 'float',
                    'ApplicantIncome': 'float',
                    'CoapplicantIncome': 'float'
                    })
    df['LoanAmountLog'] = np.log(df['LoanAmount'])
    df['TotalIncomeLog'] = np.log(df['ApplicantIncome'] + df['CoapplicantIncome'])
    df.drop(columns=['Loan_ID' ,'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'], inplace=True)
    return df


def todense(x):
    return x.todense()


model = pickle.load(open('model.pkl', 'rb'))


# our app class for the endpoint
class Predict(Resource):

    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), 
                          index=json_data.keys()).transpose()
        # return df.to_json()
        result = model.predict_proba(df)
        return result.tolist()

# link to the endpoint
api.add_resource(Predict, '/predict')

# running our api app
if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0', port=5000)

        