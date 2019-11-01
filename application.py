import pandas as pd, numpy as np
# Sınıflandırma Modellerine Ait Kütüphaneler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper

from flask import Flask, request, redirect, url_for, flash, jsonify
import os
import pickle

# Models
with open("models.pickle", 'rb') as data:
    models = pickle.load(data)
# X_train
with open("X_train.pickle", 'rb') as data:
    X_train = pickle.load(data)
# y_train
with open("y_train.pickle", 'rb') as data:
    y_train = pickle.load(data)

# mapper_features
with open("mapper_features.pickle", 'rb') as data:
    mapper_features = pickle.load(data)

numerical_cols=['krediMiktari', 'yas', 'aldigi_kredi_sayi']

categorical_cols=['evDurumu', 'telefonDurumu']

""" sample_data={ 'krediMiktari': 4000,
             'yas': 50,
             'aldigi_kredi_sayi': 5,
             'evDurumu': 'evsahibi',
             'telefonDurumu': 'var'}
"""
def process(sample_data):
    data=list(sample_data.values())
    colz=list(sample_data.keys())
    dfx=pd.DataFrame(data=[data], columns=colz)
    XX1=mapper_features.transform(dfx)
    XX2=dfx[numerical_cols]
    clean_sample = np.hstack((XX1,XX2))
    
    model_name = []
    acc_score_model = []
    
    for name, model in models:
        model = model.fit(X_train, y_train.ravel())
        model_name.append(name)
        acc_score_model.append(((model.predict_proba(clean_sample)[:,0][0])*100))
    
    columns = {'Model':model_name,'Oran':acc_score_model}
    results = pd.DataFrame(data=columns)

    results=results.sort_values('Oran', ascending=False)
    return results

app = Flask(__name__)

@app.route('/')
def home():
	return "Test Credit Scoring!"

@app.route('/test', methods=['POST'])
def predict():
    data = request.get_json()
    response=process(data)
    return jsonify(response.to_json())

if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.secret_key = "secret_key"    
    app.run(HOST, port=PORT)
