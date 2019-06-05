# Simple Linear Regression

'''
This model predicts the salary of the employ based on experience using simple linear regression model.
'''

# Importing the libraries
import numpy as np
import pandas as pd
import pickle
import requests
import json
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


PATH='/home/mohamed-mosad/Repos/chloe_flask_docker_demo/web'

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Saving model to disk
with open (PATH+'/joblib_model.pkl','wb') as file:
	joblib.dump(regressor,file)
#pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
with open (PATH+'/joblib_model.pkl','rb') as file:
	model=joblib.load(file)
# testing dataframe 
testdf=pd.concat([pd.DataFrame(y_test[:3]),pd.DataFrame(X_test[:3])],axis=1,keys=['ytest','xtest'])
testdf.to_csv(PATH+'/testdf.csv')
print("model trained and saved")










