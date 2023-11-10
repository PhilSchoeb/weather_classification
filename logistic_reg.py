# Code by Philippe Schoeb, Amine Obeid and Abubakar Garibar Mama
# November 8th 2023

# imports
import random
import numpy as np
import csv

# Hyperparameters
gradient_max = 0.0005 # If the gradient has a norm inferior to this, we stop the gradient descent
number_steps = 2000 # If the number of iterations for gradient descent is superior to this, we stop the gradient
# descent
stepsize = 1 # Multiplies the gradient when lowering w in the gradient descent
reg = 0 # Regularisation to avoid overfitting

# Modify all our data to only keep specific features. Train and test data are modified by keeping features identified
# in the kept_features array. For example :
# if kept_features = [2, 5, 19], we only keep the third, sixth and twentieth features.
def keep_important_features(train_data, test_data, kept_features):
    kept_features_and_label = np.append(np.copy(kept_features), len(train_data[0])-1)
    train_data = train_data[:, kept_features_and_label]
    test_data = test_data[:, kept_features]
    return train_data, test_data

# The beginning of data processing
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


# Modify our training data, so we only do binary classification.
# label_analysed is either 0, 1 or 2 here and since we use a one vs rest algorithms, it is the label that we'll
# consider +1 and the two other labels will be -1.
def modify_data_train(data_train, label_analysed):
    new_data_train = []
    num_equals = 0
    num_other = 0
    for line in data_train:
        if int(line[-1]) == label_analysed:
            line[-1] = 1
            num_equals += 1
        else:
            line[-1] = -1
            num_other += 1
        new_data_train.append(line)

    new_data_train = np.array(new_data_train, dtype=float)
    print("Number of labels = " + str(label_analysed) + ", is : " + str(num_equals) + " and number of others : " +
          str(num_other))
    return new_data_train


# Seperate the training data into training and validation data. We do so by randomly adding lines from data_train to
# data_validation, and we remove them from data_train. prob_valid is between 0 and 1. It is the probability of a line
# getting transferred to data_valid. For example : if prob_valid = 0.2, then a about fifth of the lines in data_train
# will be transferred to data_valid.
def separate_train_valid(data_train, prob_valid):
    new_data_train = []
    data_valid = []
    for line in data_train:
        if random.random() < prob_valid:
            data_valid.append(line)
        else:
            new_data_train.append(line)
    new_data_train = np.array(new_data_train, dtype=float)
    data_valid = np.array(data_valid, dtype=float)
    return new_data_train, data_valid

# Dot product between w^transposed and X (data) which gives a vector of size len(X) (number of data points).
def f(X, w):
    return np.dot(X, w)

# Calculates the error rate for binary classification by checking if each label is the same sign as its prediction and
# by taking the mean for all points.
def error_rate(X, w, y):
    return np.mean(np.transpose(f(X, w)) * y < 0)

# Calculates the loss function of the binary classification. We never used it but the gradient is derived from it.
def loss_function(X, w, y):
    return np.mean(np.log(1 + np.exp(-y * f(X, w)))) + .5 * reg * np.sum(w ** 2)

# Calculates the gradient of the model. Essential for gradient descent.
def gradient(X, w, y):
    return (((1 / (1 + np.exp(y * f(X, w)))) * -y)[:, np.newaxis] * X).mean(axis=0) + reg * w

# Training the model by modeling of the best w vector for binary classification from w0.
def train(data, w0):
    # Initialization
    w = np.copy(w0)
    X = data[:, :-1]
    y = data[:, -1]
    errors = []
    gradient_norm = 100
    i = 0

    # Condition to continue the gradient descent
    while gradient_norm > gradient_max and i < number_steps:
        # Gradient descent
        grad = gradient(X, w, y)
        gradient_norm = np.linalg.norm(grad)
        w -= stepsize * grad

        errors += [error_rate(X, w, y)]
        i += 1

    # Information
    print("Gradient of norm : " + str(gradient_norm))
    print("Training completed. The train error is {:.2f}%".format(errors[-1]*100))
    print('Initial weights: ', w0)
    print('Final weights: ', w)
    return w

# Training of three different models, one for each class. Thus creating a w vector for each label. Given a proportion,
# the train and validation data are split in two sets.
def three_models_w(valid_proportion):
    # Initialization
    data_train, data_test = data_preprocessing()
    d = len(data_test[0])
    first_w = [0.] * d
    first_w = np.array(first_w)

    # Prepare training data
    data_train_all, data_valid_all = separate_train_valid(data_train.copy(), valid_proportion)
    data_train_0 = modify_data_train(data_train_all.copy(), 0)
    data_train_1 = modify_data_train(data_train_all.copy(), 1)
    data_train_2 = modify_data_train(data_train_all.copy(), 2)

    # Training of three models
    w1 = train(data_train_0, first_w)
    w2 = train(data_train_1, first_w)
    w3 = train(data_train_2, first_w)

    return w1, w2, w3, data_test, data_valid_all

# Get global predictions based on the predictions of the three models. This is called
# "normal" because it is a fair selection of the most likely label according to the
# models. pred0 are the predictions of the label 0 classifier and so on.
def get_preds_normal(pred0, pred1, pred2):
    num0 = 0
    num1 = 1
    num2 = 0
    pred = [-1] * len(pred0)

    for i in range(len(pred0)):
        # We predict the label of the highest prediction
        maxi = max(pred0[i], pred1[i], pred2[i])
        if maxi == pred0[i]:
            num0 += 1
            pred[i] = 0
        elif maxi == pred1[i]:
            pred[i] = 1
            num1 += 1
        else:
            pred[i] = 2
            num2 += 1
    return pred, num0, num1, num2

# Same utility as the function above, but there is a bias for choosing the label 1. If the prediction of the label 1
# classifier is positive, the prediction is automatically 1, even if some predictions for this point might be higher
# for other labels. This bias was, sometimes, used because the label 1 classifier is by far the most accurate according
# to our tests.
def get_preds_bias1(pred0, pred1, pred2):
    num0 = 0
    num1 = 1
    num2 = 0
    pred = [-1] * len(pred0)

    for i in range(len(pred0)):
        # Here is the bias condition
        if pred1[i] > 0:
            num1 += 1
            pred[i] = 1
        # If not, we do as if it was the normal case
        else:
            maxi = max(pred0[i], pred1[i], pred2[i])
            if maxi == pred0[i]:
                num0 += 1
                pred[i] = 0
            elif maxi == pred1[i]:
                pred[i] = 1
                num1 += 1
            else:
                pred[i] = 2
                num2 += 1
    return pred, num0, num1, num2

# Function to use a validation set. We used it sometimes, but it did not give us much information. Specify the
# proportion of data to be in the validation set (between 0 and 1).
def valid_predictions(valid_proportion):
    w1, w2, w3, data_test, data_valid = three_models_w(valid_proportion)
    data_valid_attributes = data_valid[:, :-1]
    data_valid_labels = data_valid[:, -1]

    # Get predictions of the three models
    pred0 = np.dot(data_valid_attributes, w1)
    pred1 = np.dot(data_valid_attributes, w2)
    pred2 = np.dot(data_valid_attributes, w3)

    num_pos0 = 0
    num_pos1 = 0
    num_pos2 = 0
    num_neg0 = 0
    num_neg1 = 0
    num_neg2 = 0
    num_no = 0

    # Gather information around the three models
    for i in range(len(pred0)):
        if pred0[i] > 0:
            num_pos0 += 1
        else:
            num_neg0 += 1
        if pred1[i] > 0:
            num_pos1 += 1
        else:
            num_neg1 += 1
        if pred2[i] > 0:
            num_pos2 += 1
        else:
            num_neg2 += 1
        if pred0[i] <= 0 and pred1[i] <= 0 and pred2[i] <= 0:
            num_no += 1
    print("For class 0 prediction, we have : " + str(num_pos0) + " positive predictions and : " + str(num_neg0) +
          " negative predictions.")
    print("For class 1 prediction, we have : " + str(num_pos1) + " positive predictions and : " + str(num_neg1) +
          " negative predictions.")
    print("For class 2 prediction, we have : " + str(num_pos2) + " positive predictions and : " + str(num_neg2) +
          " negative predictions.")
    print("Number of points without any positive prediction : " + str(num_no))

    # Get global predictions
    pred, num0, num1, num2 = get_preds_normal(pred0, pred1, pred2)

    # Count errors of classification
    error_count = 0
    for i in range(len(pred)):
        if int(pred[i]) != int(data_valid_labels[i]):
            error_count += 1

    # Print information
    print(pred)
    print("We have class 0 : " + str(num0) + ", class 1 : " + str(num1) + ", class 2 : " + str(num2))
    error_rate = float(error_count) / len(pred0)
    success_rate = 1 - error_rate
    print("Accuracy : " + str(success_rate))

# Function to get the predictions for the test_data.
def test_predictions():
    w1, w2, w3, data_test, data_valid = three_models_w(0.)

    # Get predictions of the three models
    pred0 = np.dot(data_test, w1)
    pred1 = np.dot(data_test, w2)
    pred2 = np.dot(data_test, w3)

    num_pos0 = 0
    num_pos1 = 0
    num_pos2 = 0
    num_neg0 = 0
    num_neg1 = 0
    num_neg2 = 0
    num_no = 0

    # Gather information around the three models
    for i in range(len(pred0)):
        if pred0[i] > 0:
            num_pos0 += 1
        else:
            num_neg0 += 1
        if pred1[i] > 0:
            num_pos1 += 1
        else:
            num_neg1 += 1
        if pred2[i] > 0:
            num_pos2 += 1
        else:
            num_neg2 += 1
        if pred0[i] <= 0 and pred1[i] <= 0 and pred2[i] <= 0:
            num_no += 1
    print("For class 0 prediction, we have : " + str(num_pos0) + " positive predictions and : " + str(num_neg0) +
          " negative predictions.")
    print("For class 1 prediction, we have : " + str(num_pos1) + " positive predictions and : " + str(num_neg1) +
          " negative predictions.")
    print("For class 2 prediction, we have : " + str(num_pos2) + " positive predictions and : " + str(num_neg2) +
          " negative predictions.")
    print("Number of points without any positive prediction : " + str(num_no))

    # Get global predictions
    pred, num0, num1, num2 = get_preds_normal(pred0, pred1, pred2)
    print(pred)
    print("We have class 0 : " + str(num0) + ", class 1 : " + str(num1) + ", class 2 : " + str(num2))
    return pred

# Output our solution in the kaggle submission format (csv).
def write_predictions():
    pred = test_predictions()
    title = "pred_" + str(gradient_max) + "_" + str(number_steps) + "_" + str(stepsize) + "_" + \
            str(reg) + ".csv"
    with open(title, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["SNo", "Label"])
        for i in range(len(pred)):
            writer.writerow([i+1, pred[i]])

write_predictions()