import random
import sys

import numpy as np
import csv

# Hyperparameters
gradient_max = 0.0008
number_steps = 2000
stepsize = 0.5
reg = 0


def data_preprocessing():
    # Read file to get data
    # Directly input the csv into numpy array and skipping the header
    csv1 = np.genfromtxt("train.csv", delimiter=",", skip_header=1, dtype=float)
    csv2 = np.genfromtxt("test.csv", delimiter=",", skip_header=1, dtype=float)

    data_train = csv1[:, 1:]
    data_test = csv2[:, 1:]

    # Add an attibute just before the label to replace the bias (+b) for our linear model
    data_train = np.insert(data_train, -1, 1, axis=1)
    index = len(data_test[0])
    data_test = np.insert(data_test, index, 1, axis=1)

    # Normalize our data so every attribute has a mean = 0 and std = 1
    mu = np.mean(data_train[:, :-2], axis=0, dtype=float)
    sigma = np.std(data_train[:, :-2], axis=0, dtype=float)

    data_train[:, :-2] = (data_train[:, :-2] - mu) / sigma
    data_test[:, :-1] = (data_test[:, :-1] - mu) / sigma
    return data_train, data_test



# label_analysed is either 0, 1 or 2 here and since we use here a one vs rest algorithms, it is the label that we'll
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
    print("Number of labels = " + str(label_analysed) + ", is : " + str(num_equals) + " and number of others : " + str(num_other))
    return new_data_train


# Seperate the training data into training and validation data. We do so by randomly adding
# lines from data_train to data_validation and we remove them from data_train.
# prob_valid is between 0 and 1. It is the probability of a line getting transferred to
# data_valid. For example : if prob_valid = 0.2, then a fifth of the lines in data_train
# will be transferred to data_valid.
def separate_train_valid(data_train, prob_valid):
    new_data_train = []
    data_valid = []
    num_equals = 0
    num_other = 0
    for line in data_train:
        if random.random() < prob_valid:
            data_valid.append(line)
        else:
            new_data_train.append(line)
    new_data_train = np.array(new_data_train, dtype=float)
    data_valid = np.array(data_valid, dtype=float)
    return new_data_train, data_valid


# This next part is code modified from tp4:
class LinearModel:
    """"Classe parent pour tous les modèles linéaires.
    """

    def __init__(self, w0, reg, gradient_max):
        """Les poids et les biais sont définis dans w.
        L'hyperparamètre de régularisation est reg.
        """
        self.w = np.array(w0, dtype=float)
        self.reg = reg
        self.gradient_max = gradient_max

    def predict(self, X):
        """Retourne f(x) pour un batch X
        """
        return np.dot(X, self.w)

    def error_rate(self, X, y):
        """Retourne le taux d'erreur pour un batch X
        """
        return np.mean(np.transpose(self.predict(X)) * y < 0)

    # les méthodes loss et gradient seront redéfinies dans les classes enfants
    def loss(self, X, y):
        return 0

    def gradient(self, X, y):
        return self.w

    def train(self, data, stepsize, n_steps):
        """Faire la descente du gradient avec batch complet pour n_steps itération
        et un taux d'apprentissage fixe. Retourne les tableaux de loss et de
        taux d'erreur vu apres chaque iteration.
        """

        X = data[:, :-1]
        y = data[:, -1]
        losses = []
        errors = []
        gradient_norm = 5
        #for i in range(n_steps):
        i = 0
        while gradient_norm > self.gradient_max:
            # Gradient Descent
            gradient = self.gradient(X, y)
            gradient_norm = np.linalg.norm(gradient)
            self.w -= (stepsize) * gradient

            # Update losses
            losses += [self.loss(X, y)]

            # Update errors
            errors += [self.error_rate(X, y)]
            i += 1
        print("Gradient of norm : " + str(np.linalg.norm(self.gradient(X, y))))
        print("Training completed: the train error is {:.2f}%".format(errors[-1]*100))
        return np.array(losses), np.array(errors)

def test_model(modelclass, w0, data_train, gradient_max):
    """Crée une instance de modelclass, entraîne la, calcule le taux d'erreurs sur un
    test set, trace les courbes d'apprentissage et la frontieres de decision.
    """
    # Might not need to seperate the data into train and validation !!!!!!!
    model = modelclass(w0, reg, gradient_max)
    training_loss, training_error = model.train(data_train, stepsize, number_steps)
    #print("The validation error is {:.2f}%".format(model.error_rate(valid[:, :-1], valid[:, -1]) * 100))
    print('Initial weights: ', w0)
    print('Final weights: ', model.w)
    return model.w

class LogisticRegression(LinearModel):

    def __init__(self, w0, reg, gradient_max):
        super().__init__(w0, reg, gradient_max)

    def loss(self, X, y):
        return np.mean(np.log(1 + np.exp(-y * self.predict(X)))) + .5 * self.reg * np.sum(self.w ** 2)

    def gradient(self, X, y):
        probas = 1 / (1 + np.exp(y * self.predict(X)))
        return ((probas * -y)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w


def three_models_w(valid_proportion, gradient_max):
    data_train, data_test = data_preprocessing()
    first_w = [0] * 20

    data_train_all, data_valid_all = separate_train_valid(data_train.copy(), valid_proportion)
    data_train_0 = modify_data_train(data_train_all.copy(), 0)
    data_train_1 = modify_data_train(data_train_all.copy(), 1)
    data_train_2 = modify_data_train(data_train_all.copy(), 2)

    w1 = test_model(LogisticRegression, first_w, data_train_0, gradient_max)

    w2 = test_model(LogisticRegression, first_w, data_train_1, gradient_max)

    w3 = test_model(LogisticRegression, first_w, data_train_2, gradient_max)

    return w1, w2, w3, data_test, data_valid_all

# Use validation set to optimize gradient_max
def valid_predictions(gradient_max):
    gradient_max_tab = []
    success_rate_tab = []
    w1_tab = []
    w2_tab = []
    w3_tab = []
    while gradient_max <= 1:
        w1, w2, w3, data_test, data_valid = three_models_w(0.25, gradient_max)
        w1_tab.append(w1)
        w2_tab.append(w2)
        w3_tab.append(w3)
        pred = [-1] * len(data_valid)
        print("Validation set of length : " + str(len(data_valid)))
        data_valid_attributes = data_valid[:, :-1]
        data_valid_labels = data_valid[:, -1]

        pred0 = np.dot(data_valid_attributes, w1)
        pred1 = np.dot(data_valid_attributes, w2)
        pred2 = np.dot(data_valid_attributes, w3)

        num0 = 0
        num1 = 0
        num2 = 0

        num_pos0 = 0
        num_pos1 = 0
        num_pos2 = 0
        num_neg0 = 0
        num_neg1 = 0
        num_neg2 = 0

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
        print("For class 0 prediction, we have : " + str(num_pos0) + " positive predictions and : " + str(num_neg0) +
              " negative predictions.")
        print("For class 1 prediction, we have : " + str(num_pos1) + " positive predictions and : " + str(num_neg1) +
              " negative predictions.")
        print("For class 2 prediction, we have : " + str(num_pos2) + " positive predictions and : " + str(num_neg2) +
              " negative predictions.")
        error_count = 0
        for i in range(len(pred0)):
            maxi = max(pred0[i], pred1[i], pred2[i])
            if maxi == pred0[i]:
                num0 += 1
                pred[i] = 0
                if data_valid_labels[i] != 0:
                    error_count += 1
            elif maxi == pred1[i]:
                pred[i] = 1
                num1 += 1
                if data_valid_labels[i] != 1:
                    error_count += 1
            else:
                pred[i] = 2
                num2 += 1
                if data_valid_labels[i] != 2:
                    error_count += 1
        print(pred)
        print("We have class 0 : " + str(num0) + ", class 1 : " + str(num1) + ", class 2 : " + str(num2))
        error_rate = float(error_count) / len(pred0)
        success_rate = 1 - error_rate
        success_rate_tab.append(success_rate)
        gradient_max_tab.append(gradient_max)
        gradient_max = gradient_max * 2

    print(gradient_max_tab)
    print(success_rate_tab)
    index = np.argmax(success_rate_tab) # Get index of best model according to success rate of validation set
    print(index)
    gradient_max = gradient_max_tab[index]
    print(gradient_max)
    best_w1 = w1_tab[index]
    best_w2 = w2_tab[index]
    best_w3 = w3_tab[index]
    return best_w1, best_w2, best_w3, data_test

def test_predictions(gradient_max):
    w1, w2, w3, data_test = valid_predictions(gradient_max)
    pred = [-1] * len(data_test)
    pred0 = np.dot(data_test, w1)
    pred1 = np.dot(data_test, w2)
    pred2 = np.dot(data_test, w3)

    num0 = 0
    num1 = 0
    num2 = 0

    num_pos0 = 0
    num_pos1 = 0
    num_pos2 = 0
    num_neg0 = 0
    num_neg1 = 0
    num_neg2 = 0

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
    print("For class 0 prediction, we have : " + str(num_pos0) + " positive predictions and : " + str(num_neg0) +
          " negative predictions.")
    print("For class 1 prediction, we have : " + str(num_pos1) + " positive predictions and : " + str(num_neg1) +
          " negative predictions.")
    print("For class 2 prediction, we have : " + str(num_pos2) + " positive predictions and : " + str(num_neg2) +
          " negative predictions.")

    for i in range(len(pred0)):
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
    print(pred)
    print("We have class 0 : " + str(num0) + ", class 1 : " + str(num1) + ", class 2 : " + str(num2))
    return pred


def write_predictions(gradient_max):
    pred = test_predictions(gradient_max)
    title = "pred" + str(gradient_max) + "_" + str(stepsize) + "_" + str(reg) + ".csv"
    with open(title, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["SNo", "Label"])
        for i in range(len(pred)):
            writer.writerow([i+1, pred[i]])

# Useless function
def prediction(X, w):
    prediction = np.array([0] * len(X))
    sigmoid = 1 / (1 + np.exp(-np.dot(X, w)))
    for i in range(len(sigmoid)):
        if sigmoid[i] > 0.5:
            prediction[i] = 1
        else:
            prediction[i] = -1
    return prediction

write_predictions(gradient_max)
