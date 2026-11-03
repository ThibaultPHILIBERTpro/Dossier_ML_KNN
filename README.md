# Dossier_ML_KNN : Prédiction de l'Admission à l'Agrégation de Droit

Ce projet implémente un modèle K-Nearest Neighbors (KNN) basé exclusivement sur 3 variables explicatives : la distance géographique à Paris (`Distance_univ_soutenance-paris`), le score académique (`Publication`), et le prestige éditorial (`Editeur_de_la_these`). L'optimisation du modèle a convergé vers les hyperparamètres suivants : un voisinage **K=14**, une pondération par la distance (`weights='distance'`), et une métrique Euclidienne (`p=2`). Cette configuration permet d'obtenir un score de validation croisée (CV) moyen de **69.66 %** (Accuracy).

> **Projet Académique (Master 2 EADE - Université Côte d'Azur)**
> *Réalisé en binôme par Thibault PHILIBERT & Thibault AZAIS.*
> *Note : Les données de ce dépôt (`data_agreg.xlsx`) sont des données fictives générées pour illustrer le fonctionnement du pipeline, remplaçant la base académique originale pour des raisons de confidentialité.*

## Pipeline & Feature Engineering
L'architecture repose sur un pipeline Scikit-Learn complet intégrant des transformateurs personnalisés (`FunctionTransformer`) pour éviter toute fuite de données :
* **Sélection stricte :** Filtrage initial pour isoler les 3 variables d'intérêt.
* **Feature Engineering :** Transformation de la variable `Editeur_de_la_these` en 3 catégories binaires (Top, Bottom, Non-publié) selon un seuil statistique optimisé à 16 candidats.
* **Gestion des Outliers :** Écrêtage (clipping) basé sur l'écart interquartile (IQR) calculé sur le Train set pour limiter l'impact des valeurs extrêmes sur le calcul des distances.
* **Normalisation :** Application d'un `RobustScaler` pour harmoniser les échelles.

## Règle de Décision
La sélection finale privilégie une **stratégie de stabilité**. Le modèle retenu présente l'écart-type le plus faible parmi les configurations les plus performantes (GridSearchCV), garantissant une robustesse confirmée par une Accuracy de **70.15 %** sur le jeu de test externe.
