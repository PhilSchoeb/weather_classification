import random
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pymc3 as pm


# Read file to get data
file1 = open("train.csv")
file2 = open("test.csv")

csvreader1 = csv.reader(file1)
csvreader2 = csv.reader(file2)

header_train = []
header_train = next(csvreader1)

next(csvreader2)

data_train = []
for row in csvreader1:
    data_train.append(row)

data_test = []
for row in csvreader2:
    data_test.append(row)

file1.close()
file2.close()

# Turn our data in np.array and remove the first attribue which is just numeration
data_train = np.array(data_train, dtype=float)
data_train = np.delete(data_train, 0, axis=1)

data_test = np.array(data_test, dtype=float)
data_test = np.delete(data_test, 0, axis=1)



X=Data[:,0:15]
y=Data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_samples = 100
n_features = 16
n_classes = 3




# Predicted labels for the test set
predicted_labels = np.argmax(ppc['y_obs'], axis=1)

print("Predicted labels for the test set:")
print(predicted_labels)


