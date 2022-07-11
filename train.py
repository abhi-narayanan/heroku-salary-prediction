import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression

data = pd.read_csv('hiring.csv')
print(data)

data.experience.fillna(0, inplace = True)

data.test_score.fillna(data.test_score.mean(), inplace = True)

X = data.iloc[:, :3]

def convert_to_int(word):
    word_dic = {0:0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nice': 9, 'ten': 10, 'eleven': 11}
    return word_dic[word]

X['experience'] = X.experience.apply(lambda x: convert_to_int(x))

y = data.iloc[:, -1]

model = LinearRegression()
model.fit(X,y)

print('Model training is done.')

print(model.predict([[1,8,9]]))

joblib.dump(model, 'hiring_model.pkl')