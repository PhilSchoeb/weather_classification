# weather_classification
IFT3395 Compétition Kaggle 1

Philippe Schoeb, Amine Obeid et Abubakar Garibar Mama

8 novembre 2023

Voici nos deux modèles les mieux réussis.
D'abord, il y a notre modèle de régression logistique qui était obligatoire.
Ensuite, il y a notre modèle d'arbre de décision.

Pour la lecture des données, les deux modèles lisent les fichiers : train.csv et test.csv qui doivent se trouver dans
le même dossier que les deux fichiers python des modèles.

Pour l'exécution du code, simplement exécuter les fichiers logistic_reg.py ainsi que decision_tree.py et les prédictions
pour les données de test (test.csv) seront calculées puis mises en fichier csv dans le même dossier.

Pour modifier certains hyper paramètres, il suffit de les modifier au tout début de chacun des fichiers pythons. Les
valeurs initiales des hyper paramètres sont les meilleurs ou les plus efficaces selon nos tests.

Pour le modèle de Bayes, il suffit de rouler le code et fermer les graphes pour avoir la prédiction des classes (Output.csv) et
la précision pour l'ensemble de validation.Les graphes sont la visualisation suites a la modification de deux hyperparamètres
(proportions de points de classe 1 et 2) en faisant du suréchantillonnage.

Sinon, à vous de modifier les fichiers python pour utiliser nos fonctions intermédiaires. Nous avons gardé certaines
fonctions inutilisées seulement si elles nous ont aidé à atteindre une conclusion durant la compétition.

Voilà, merci!
