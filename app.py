#import flask
from flask import Flask , render_template, url_for, request
import joblib
import pickle

# EDA packages
import pandas as pd
import numpy as np 
import os

from sklearn import linear_model
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import metrics


from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier


app = Flask(__name__)
#utils packages

import os 
import joblib

# function for loading Model
model_file = r'/models/model.pkl'

def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
	return loaded_model


# To get value -- value mapping
def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value

def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key




#Routes
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/dataset')
def dataset():
	df = pd.read_csv("data/BankChurners.csv")
	return render_template('dataset.html', df_table = df)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		Customer_Age = request.form['Customer_Age']
		Gender = request.form['Gender']
		Dependent_count = request.form['Dependent_count']
		Education_level = request.form['Education_level']
		Marital_Status = request.form['Marital_Status']
		Income_Category = request.form['Income_Category']
		Card_Category = request.form['Card_Category']
		Total_Relationship_Count = request.form['Total_Relationship_Count']
		Months_Inactive_12_mon = request.form['Months_Inactive_12_mon']
		Contacts_Count_12_mon = request.form['Contacts_Count_12_mon']
		Credit_Limit = request.form['Credit_Limit']
		Total_Revolving_Bal = request.form['Total_Revolving_Bal']
		Total_Amt_Chng_Q4_Q1 = request.form['Total_Amt_Chng_Q4_Q1']
		Total_Trans_Amt = request.form['Total_Trans_Amt']
		Total_Ct_Chng_Q4_Q1 = request.form['Total_Ct_Chng_Q4_Q1']
		Avg_Utilization_Ratio = request.form['Avg_Utilization_Ratio']
		

		"""
		pretty_data = {'Customer_Age': Customer_Age, 'Gender': Gender, 'Dependent_count': Dependent_count, 'Education_level': Education_level, 'Marital_Status': Marital_Status, 'Income_Category': Income_Category, 'Card_Category': Card_Category,'Total_Relationship_Count': Total_Relationship_Count, 'Months_Inactive_12_mon': Months_Inactive_12_mon, 'Contacts_Count_12_mon': Contacts_Count_12_mon, 'Credit_Limit': Credit_Limit,'Total_Revolving_Bal': Total_Revolving_Bal, 'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1, 'Total_Trans_Amt': Total_Trans_Amt,'Total_Ct_Chng_Q4_Q1': Total_Ct_Chng_Q4_Q1, 'Avg_Utilization_Ratio': Avg_Utilization_Ratio }

		single_data = [Customer_Age, Gender, Dependent_count, Education_level, Marital_Status, Income_Category, Card_Category, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon,Credit_Limit,Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio]

		print(single_data)
		print(len(single_data))
		numerical_encoded_data = [[ float(int(x)) for x in single_data ]]
		#encoded_data = [int(i) for i in sample_data]
		model = load_model('models/model.pkl')
		prediction = model.predict(numerical_encoded_data)
		print(prediction)
		prediction_label = {"Churn": 1, "Stayed":2}
		final_result = get_key(prediction[0], prediction_label)
		pred_prob = model.predict_proba(np.array(numerical_encoded_data).reshape(1,-1))

		"""

		pretty_data = {'Customer_Age': Customer_Age, 'Gender': Gender, 'Dependent_count': Dependent_count, 'Education_level': Education_level, 'Marital_Status': Marital_Status, 'Income_Category': Income_Category, 'Card_Category': Card_Category,
						'Total_Relationship_Count': Total_Relationship_Count, 'Months_Inactive_12_mon': Months_Inactive_12_mon, 'Contacts_Count_12_mon': Contacts_Count_12_mon, 'Credit_Limit': Credit_Limit,
						'Total_Revolving_Bal': Total_Revolving_Bal, 'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1, 

						#'Contacts_Count_12_mon': Contacts_Count_12_mon, 
						'Total_Trans_Amt': Total_Trans_Amt,
						 'Total_Ct_Chng_Q4_Q1': Total_Ct_Chng_Q4_Q1, 'Avg_Utilization_Ratio': Avg_Utilization_Ratio }

		sample_data = [Customer_Age, Gender, Dependent_count, Education_level, Marital_Status, Income_Category, Card_Category, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon,Credit_Limit,
						Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio]
		
		encoded_data = [(int(x)) for x in sample_data ]
						#[int(i) for i in sample_data]
		prediction_label = {"Churn": 1, "Stayed":0}
		filename = 'finalized_model_version2 - LR.sav'
		loaded_model = pickle.load(open(filename, 'rb'))
		model = load_model('finalized_model_version2 - XGB.sav')
		#prediction = model.predict(sample_data)
		#prediction = model.predict(np.array(encoded_data).reshape(1,-1))

		model_predictor = load_model("finalized_model_version2 - XGB.sav")
		prediction = model.predict(np.array(encoded_data).reshape(1,-1))


		#model.predict(encoded_data)
		print(prediction)

		
		final_result = get_key(prediction, prediction_label)
		print (final_result)

	return render_template('index.html', sample_data = sample_data, pretty_data=pretty_data, encoded_data=encoded_data, final_result = final_result, prediction=prediction)
	#return render_template("index.html",sample_result=sample_result,prediction=final_result,pred_probalility_score=pred_probalility_score)




if __name__ == '__main__':
	app.run(debug=True)
