import pandas as pd, numpy as np
# Sınıflandırma Modellerine Ait Kütüphaneler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
import pickle as p
from flask import Flask, request, jsonify
import os

# Modelleri Hazırlayalım
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('GradientBoosting', GradientBoostingClassifier(n_estimators=100)))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (NoParam)', DecisionTreeClassifier())) 
models.append(('Decision Tree (GridSearch)', GridSearchCV(DecisionTreeClassifier(), {'max_depth':[5, 10, 15, 20, 25, 32]}, cv=5)))
models.append(('RandomForestClassifier (GridSearch)', GridSearchCV(RandomForestClassifier(), {'max_depth':[5, 15], 'n_estimators':[10,30]})))
models.append(('RandomForestClassifier (2 Param)', RandomForestClassifier(n_estimators=10, criterion='entropy')))
models.append(('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=5, metric='minkowski')))
# models.append(('Support Vector Regression', SVR(kernel='rbf')))
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
models.append(('AdaBoostClassifier', AdaBoostClassifier(learning_rate=0.5)))
models.append(('BaggingClassifier', BaggingClassifier()))

# X_train
path_X_train = os.path.join(os.path.dirname( __file__ ), 'kredi/X_train.pickle')
with open(path_X_train, 'rb') as data:
    X_train = p.load(data)
# y_train
path_y_train = os.path.join(os.path.dirname( __file__ ), 'kredi/y_train.pickle')
with open(path_y_train, 'rb') as data:
    y_train = p.load(data)

# mapper_features
path_mapper_features = os.path.join(os.path.dirname( __file__ ), 'kredi/mapper_features.pickle')
with open(path_mapper_features, 'rb') as data:
    mapper_features = p.load(data)

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
    return jsonify(response.to_json(orient='values'))

if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.secret_key = "secret_key"    
    app.run(HOST, port=PORT)
