# Code by Philippe Schoeb, Amine Obeid and Abubakar Garibar Mama
# November 8th 2023

# imports
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier

# Hyperparameter
max_depth = 6 # Max depth of the decision tree


# Data processing
def data_preprocessing():
    # Read file to get data
    # Directly input the csv into numpy array and skipping the header
    csv1 = np.genfromtxt("train.csv", delimiter=",", skip_header=1, dtype=float)
    csv2 = np.genfromtxt("test.csv", delimiter=",", skip_header=1, dtype=float)

    # Remove index numbers
    data_train = csv1[:, 1:]
    data_test = csv2[:, 1:]

    # Feature manip, separate time YYYYMMDD into two columns : YYYY and MM
    date_column_train = data_train[:, -2]
    date_column_test = data_test[:, -1]

    # Extract YYYY and MM from train and test data
    year_column_train = date_column_train // 10000
    month_column_train = (date_column_train // 100) % 100
    year_column_test = date_column_test // 10000
    month_column_test = (date_column_test // 100) % 100

    # Replace YYYYMMDD by YYYY and MM for train and test data
    data_train = np.column_stack((data_train[:, :-2], year_column_train, month_column_train, data_train[:, -1]))
    data_test = np.column_stack((data_test[:, :-1], year_column_test, month_column_test))

    # Add a fake feature of value 1 at the end of each point to replace the bias (+b) for our linear model
    data_train = np.insert(data_train, -1, 1, axis=1)
    index = len(data_test[0])
    data_test = np.insert(data_test, index, 1, axis=1)

    # Normalize our data so every attribute has a mean = 0 and std = 1
    mu = np.mean(data_train[:, :-2], axis=0, dtype=float)
    sigma = np.std(data_train[:, :-2], axis=0, dtype=float)

    data_train[:, :-2] = (data_train[:, :-2] - mu) / sigma
    data_test[:, :-1] = (data_test[:, :-1] - mu) / sigma
    return data_train, data_test


data_train, data_test = data_preprocessing()
X_train = data_train[:, :-1]
y_train = data_train[:, -1]
X_test = data_test

# Decision tree classifier
clf = DecisionTreeClassifier(max_depth=max_depth)

# Training
clf.fit(X_train, y_train)

# Predictions
pred = clf.predict(X_test)

# Output predictions
title = "tree_pred_" + str(max_depth) + ".csv"
with open(title, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["SNo", "Label"])
    for i in range(len(pred)):
        writer.writerow([i + 1, pred[i]])
