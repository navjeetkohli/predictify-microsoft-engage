from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pickle

from imports import *
from clean import *
from process import *

#Training and saving the model in pickle files
cars_data = process_data()
x = cars_data.iloc[:,1:]
y = cars_data['selling_price']
best_model = RandomForestRegressor(n_estimators=400,
                                  min_samples_split=10,
                                  min_samples_leaf=1,
                                  max_features='sqrt',
                                  max_depth=60,
                                  bootstrap=False)

numeric_features = [0,2,5,6,7,8]
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

categorical_features = [1,3,4,9]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", best_model)]
)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.20,random_state=25)

# training the classifier on the dataset
clf.fit(xtrain, ytrain)

with open('model_pkl', 'wb') as files:
    pickle.dump(best_model, files)

with open('pipe_pkl', 'wb') as files:
    pickle.dump(clf, files)

#Loading the pickle files and Calling the predict function
def do_prediction(features):
    
    with open('model_pkl', 'rb') as f:
        pickle_best_model = pickle.load(f)

    with open('pipe_pkl', 'rb') as f:
        pickle_clf = pickle.load(f)

    #Do prediction and evaluting the prediction
    prediction = pickle_clf.predict(features)

    return prediction


