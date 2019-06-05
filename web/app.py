from flask import Flask, render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)

with open ('joblib_model.pkl','rb') as file:
	model=joblib.load(file)

testdf=pd.read_csv('testdf.csv')

@app.route('/')
def hello_whale():
    return render_template("whale_hello.html")

@app.route('/predict')
def predict():

	return "Salary Predictions : "+str(model.predict(np.array(testdf['xtest']).reshape(-1,1)))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
