import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

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


class GaussianMaxLikelihood:
    def __init__(self, n_dims, cov_type='isotropic'):
        self.cov_type = cov_type
        self.n_dims = n_dims
        self.mu = np.zeros(n_dims)
        # Nous avons un scalaire comme écart-type car notre modèle est une loi gaussienne isotropique
        self.sigma_sq = 1.0

    # Pour un jeu d'entraînement, la fonction devrait calculer les estimateur ML de l'espérance et de la variance
    def train(self, train_data):
        # Ici, nous devons trouver la moyenne et la variance dans train_data et les définir dans self.mu and self.

        self.mu = np.mean(train_data, axis=0)

        # here we will create the covariance matrix
        if self.cov_type == 'isotropic':
            # Identity times sigma square
            self.covariance = np.eye(
                self.n_dims) * np.sum((train_data - self.mu) ** 2.0) / (self.n_dims * train_data.shape[0])
        elif self.cov_type == 'diagonal':
            # put the variance on the diagonal
            self.covariance = np.diag(np.var(train_data, axis=0))
        else:
            # Calculate the full covariance matrix
            self.covariance = np.cov(train_data, rowvar=False)

    # Retourne un vecteur de dimension égale au nombre d'ex. test qui contient les log probabilité de chaque
    # exemple test

    def loglikelihood(self, test_data):
        # Calculer la constante de normalisation de la façon standard, sans raccourci
        c = -(np.log(np.sqrt(np.linalg.det(self.covariance))) +
              (self.n_dims / 2) * np.log(2 * np.pi))
        # Ensuite la log prob
        # Notez l'absence d'un second np.dot. Pouvez-vous deviner pourquoi?
        log_prob = c - (np.dot((test_data - self.mu), np.linalg.inv(self.covariance))
                        * (test_data - self.mu)).sum(axis=1) / 2
        return log_prob


class BayesClassifier:
    def __init__(self, maximum_likelihood_models, priors):
        self.maximum_likelihood_models = maximum_likelihood_models
        self.priors = priors
        if len(self.maximum_likelihood_models) != len(self.priors):
            print('The number of ML models must be equal to the number of priors!')
        self.n_classes = len(self.maximum_likelihood_models)

    # Retourne une matrice de dimension [nb d'ex. test, nb de classes] contenant les log
    # probabilités de chaque ex. test sous le modèle entrainé par le MV.
    def loglikelihood(self, test_data):

        log_pred = np.zeros((test_data.shape[0], self.n_classes))

        for i in range(self.n_classes):
            # Ici, nous devrons utiliser maximum_likelihood_models[i] et priors pour remplir
            # chaque colonne de log_pred (c'est plus efficace de remplir une colonne à la fois)

            log_pred[:, i] = self.maximum_likelihood_models[i].loglikelihood(
                test_data) + np.log(self.priors[i])

        return log_pred

def get_accuracy(data, labels):
    # Nous pouvons calculez les log-probabilités selon notre modèle
    log_prob = classifier.loglikelihood(data)
    # Il reste à calculer les classes prédites
    classes_pred = log_prob.argmax(1)
    # Retournez l'exactitude en comparant les classes prédites aux vraies étiquettes
    acc = np.mean(classes_pred == labels)
    return acc



X_test = data_test
####################################
##Trouver nombre de points synthétique pour la classe 1 pour avoir best
#Taux de precision sur test de validation
preci=[]
nbClass=[]

for i in range(1825,35179,1000):
    strategy = {0: 35179, 1: i, 2:7756}
    oversample = SMOTE(sampling_strategy=strategy)
    X, y = oversample.fit_resample(data_train[:, 0:19], data_train[:, -1])






    X_train, X_val, Y_train, Y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    DataTrainClass = np.hstack((X_train, Y_train.reshape(-1, 1)))#split data and use X_Train and YTrain to create classifier

    Data_trainClass0=DataTrainClass [DataTrainClass [:,-1]==0]
    Data_trainClass1 = DataTrainClass [DataTrainClass [:, -1] == 1]
    Data_trainClass2 = DataTrainClass[DataTrainClass[:, -1] == 2]

    cov_type = 'full'
    #cov_type = 'isotropic'
    #cov_type = 'diagonal'


    model_class1 = GaussianMaxLikelihood(19,cov_type)
    model_class2 = GaussianMaxLikelihood(19,cov_type)
    model_class3 = GaussianMaxLikelihood(19, cov_type)
    model_class1.train(Data_trainClass0[:,0:19])
    model_class2.train(Data_trainClass1[:,0:19])
    model_class3.train(Data_trainClass2[:,0:19])

    model_ml = [model_class1, model_class2, model_class3]


    # Calculate the class priors
    total_samples = DataTrainClass.shape[0]

    # Count the number of samples in each class
    num_samples_class0 = len(Data_trainClass0)
    num_samples_class1 = len(Data_trainClass1)
    num_samples_class2 = len(Data_trainClass2)

    # Calculate the class priors
    prior_class0 = num_samples_class0 / total_samples
    prior_class1 = num_samples_class1 / total_samples
    prior_class2 = num_samples_class2 / total_samples

    #print("Class Priors:")
    #print("Class 0 Prior:", prior_class0)
    #print("Class 1 Prior:", prior_class1)
    #print("Class 2 Prior:", prior_class2)

    priors=np.array([prior_class0,prior_class1,prior_class2])
    classifier = BayesClassifier(model_ml, priors)
   
    nbClass.append(i)
    
    preci.append(get_accuracy(X_val, Y_val))
    #print("The training accuracy is : {:.1f} % ".format(
     #   100 * get_accuracy(X_val, Y_val)))
plt.figure()
plt.plot(nbClass,preci)
plt.xlabel("Nombre de points synthétique de la classe 1")
plt.ylabel("Précision")
plt.savefig("FigureBestoverSamplingClasse1Indep.png")
#plt.show()
nbClass=np.array(nbClass)
preci=np.array(preci)

bestNumberClasse1=nbClass[np.argmax(preci)]



####################################
##Trouver nombre de points synthétique pour la classe 2 pour avoir best
#Taux de precision sur test de validation
preci=[]
nbClass=[]

for i in range(7756,35179,1000):
    strategy = {0: 35179, 1: 1825, 2:i}
    oversample = SMOTE(sampling_strategy=strategy)
    X, y = oversample.fit_resample(data_train[:, 0:19], data_train[:, -1])






    X_train, X_val, Y_train, Y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    DataTrainClass = np.hstack((X_train, Y_train.reshape(-1, 1)))#split data and use X_Train and YTrain to create classifier

    Data_trainClass0=DataTrainClass [DataTrainClass [:,-1]==0]
    Data_trainClass1 = DataTrainClass [DataTrainClass [:, -1] == 1]
    Data_trainClass2 = DataTrainClass[DataTrainClass[:, -1] == 2]

    cov_type = 'full'
    #cov_type = 'isotropic'
    #cov_type = 'diagonal'


    model_class1 = GaussianMaxLikelihood(19,cov_type)
    model_class2 = GaussianMaxLikelihood(19,cov_type)
    model_class3 = GaussianMaxLikelihood(19, cov_type)
    model_class1.train(Data_trainClass0[:,0:19])
    model_class2.train(Data_trainClass1[:,0:19])
    model_class3.train(Data_trainClass2[:,0:19])

    model_ml = [model_class1, model_class2, model_class3]


    # Calculate the class priors
    total_samples = DataTrainClass.shape[0]

    # Count the number of samples in each class
    num_samples_class0 = len(Data_trainClass0)
    num_samples_class1 = len(Data_trainClass1)
    num_samples_class2 = len(Data_trainClass2)

    # Calculate the class priors
    prior_class0 = num_samples_class0 / total_samples
    prior_class1 = num_samples_class1 / total_samples
    prior_class2 = num_samples_class2 / total_samples

    #print("Class Priors:")
    #print("Class 0 Prior:", prior_class0)
    #print("Class 1 Prior:", prior_class1)
    #print("Class 2 Prior:", prior_class2)

    priors=np.array([prior_class0,prior_class1,prior_class2])
    classifier = BayesClassifier(model_ml, priors)
   
    nbClass.append(i)
    preci.append(get_accuracy(X_val, Y_val))


    #print("The training accuracy is : {:.1f} % ".format(
    #    100 * get_accuracy(X_val, Y_val)))
plt.figure()
plt.plot(nbClass,preci)
plt.xlabel("Nombre de points synthétique de la classe 2")
plt.ylabel("Precision")
plt.savefig("FigureBestoverSamplingClasse2Indep.png")
#plt.show()

bestNumberClasse2Solo=nbClass[np.argmax(preci)]
#print(bestNumberClasse2Solo)



##################
##Test sur ensemble de validation

#strategy = {0: 35179, 1: bestNumberClasse1, 2:bestNumberClasse2Solo}
strategy = {0: 35179, 1: 1825, 2:7756}
#0:35179 1:1825 2:7756 de base
oversample = SMOTE(sampling_strategy=strategy)
X, y = oversample.fit_resample(data_train[:, 0:19], data_train[:, -1])






X_train, X_val, Y_train, Y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
DataTrainClass = np.hstack((X_train, Y_train.reshape(-1, 1)))#split data and use X_Train and YTrain to create classifier

Data_trainClass0=DataTrainClass [DataTrainClass [:,-1]==0]
Data_trainClass1 = DataTrainClass [DataTrainClass [:, -1] == 1]
Data_trainClass2 = DataTrainClass[DataTrainClass[:, -1] == 2]

cov_type = 'full'
#cov_type = 'isotropic'
#cov_type = 'diagonal'


model_class1 = GaussianMaxLikelihood(19,cov_type)
model_class2 = GaussianMaxLikelihood(19,cov_type)
model_class3 = GaussianMaxLikelihood(19, cov_type)
model_class1.train(Data_trainClass0[:,0:19])
model_class2.train(Data_trainClass1[:,0:19])
model_class3.train(Data_trainClass2[:,0:19])

model_ml = [model_class1, model_class2, model_class3]


# Calculate the class priors
total_samples = DataTrainClass.shape[0]

# Count the number of samples in each class
num_samples_class0 = len(Data_trainClass0)
num_samples_class1 = len(Data_trainClass1)
num_samples_class2 = len(Data_trainClass2)

    # Calculate the class priors
prior_class0 = num_samples_class0 / total_samples
prior_class1 = num_samples_class1 / total_samples
prior_class2 = num_samples_class2 / total_samples

print("Class Priors:")
print("Class 0 Prior:", prior_class0)
print("Class 1 Prior:", prior_class1)
print("Class 2 Prior:", prior_class2)

priors=np.array([prior_class0,prior_class1,prior_class2])
classifier = BayesClassifier(model_ml, priors)
   

    
preci.append(get_accuracy(X_val, Y_val))
print("The training accuracy is : {:.1f} % ".format(
    100 * get_accuracy(X_val, Y_val)))


log_prob = classifier.loglikelihood(X_test)
# Il reste à calculer les classes prédites
classes_pred = log_prob.argmax(1)



csv_file = 'output.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['SNo', 'Label'])  # Write the header
    for sno, label in enumerate(classes_pred, start=1):
        writer.writerow([sno, label])