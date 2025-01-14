# weather_classification
Compétition Kaggle 1 en apprentissage automatique (IFT3395) 

Voici le [rapport](https://www.overleaf.com/read/cfpxkmycwpqb#b90697) que nous avons écrit pour ce travail.

Philippe Schoeb, Amine Obeid et Abubakar Garibar Mama

8 novembre 2023

Voici nos trois modèles pour cette compétition Kaggle.
D'abord, il y a notre modèle de régression logistique qui était obligatoire.
Ensuite, il y a notre modèle d'arbre de décision qui a eu le meilleur taux de succès sur l'ensemble de test.
Puis, il y a notre classifieur de Bayes.

Pour la lecture des données, les trois modèles lisent les fichiers : train.csv et test.csv qui doivent se trouver dans
le même dossier que les trois fichiers python des modèles.

Pour les deux premiers modèles :
Pour l'exécution du code, simplement exécuter les fichiers logistic_reg.py ainsi que decision_tree.py et les prédictions
pour les données de test (test.csv) seront calculées puis mises en fichier csv dans le même dossier.

Pour modifier certains hyper paramètres, il suffit de les modifier au tout début de chacun des fichiers pythons. Les
valeurs initiales des hyper paramètres sont les meilleurs ou les plus efficaces selon nos tests.

Pour le modèle de Bayes, il suffit de rouler le code et fermer les graphes pour avoir la prédiction des classes 
(Output.csv) et la précision pour l'ensemble de validation. Les graphes sont la visualisation suites à la modification 
de deux hyper-paramètres (proportions de points de classe 1 et 2) en faisant du sur-échantillonnage.

Sinon, à vous de modifier les fichiers python pour utiliser nos fonctions intermédiaires. Nous avons gardé certaines
fonctions inutilisées seulement si elles nous ont aidé à atteindre une conclusion durant la compétition.
