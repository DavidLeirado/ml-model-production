import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pickle

df = pd.read_csv('train.csv')

scaler = StandardScaler()
X = scaler.fit_transform(df[['X1', 'X2']])

y = df['y']

model = LogisticRegression()

model.fit(X, y)

filename = './app/model.sav'
pickle.dump(model, open(filename, 'wb'))