from flask import Flask, render_template, request

from logging import debug
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('hiring_model.pkl')

@app.route('/')
def welcome():
    return render_template('base.html')

@app.route('/contact')
def contact():
    return 'Welcome to the Contact page!'

@app.route('/help')
def help():
    return 'Welcome to the HELP page!'

@app.route('/predict', methods = ['post'])
def post():

    exp = request.form.get('experience')
    score = request.form.get('test_score')
    int_score = request.form.get('interview_score')

    prediction = model.predict([[exp, score, int_score]])

    output = round(prediction[0], 2)

    return render_template('base.html', prediction_text = f'The employee salary should be ${output}.')
 
if __name__ == '__main__':
    app.run(debug= True)