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



X=data_train[:,0:15]
y=data_train[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)








# Define the PyMC3 model
with pm.Model() as model:
    # Prior for class probabilities
    p = pm.Dirichlet('p', a=np.ones(3))

    # Multinomial distribution for class assignment
    class_assignment = pm.Categorical('class_assignment', p=p, shape=(35808,15))

    # Likelihood for observed data
    likelihood = pm.Normal(
        'likelihood', mu=X_train_scaled[class_assignment], sigma=1, observed=X_train_scaled)

# Perform Bayesian inference using PyMC3
with model:
    trace = pm.sample(2000, tune=1000)

# Extract the posterior class probabilities
posterior_class_probs = trace['p'].mean(axis=0)

# Make predictions using the class probabilities
predicted_labels = np.argmax(posterior_class_probs, axis=0)

# Evaluate the model (e.g., accuracy, confusion matrix, etc.)
accuracy = (predicted_labels == y_train).mean()
print(f'Accuracy: {accuracy}')







