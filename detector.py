#Importing necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("parkinsons.data")
dataset.head()

#Extratcting features and labels
x = dataset.loc[ :, dataset.columns != 'status'].values[:, 1:]
y = dataset.loc[ :, 'status'].values

#Count of each label in y
print(y[y == 1].shape[0], y[y == 0].shape[0])

#Scale features between -1 and 1
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(x)

#Spliting dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 0)

#Traing the model
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

# DataFlair - Calculate the accuracy
y_pred = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

#Plotting training data vs model preictions
num = range(len(y_pred))
plt.scatter(range(len(y_pred)), y_test, color = "lightcoral")
plt.scatter(range(len(y_pred)), y_pred, color = "cyan")
plt.show()