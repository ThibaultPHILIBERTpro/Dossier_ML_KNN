import pandas as pd
import dill
import numpy as np
import pdb

from functools import partial
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Constante
# Les variables que la pipeline doit conserver dès le début
var_keep = ['Distance_univ_soutenance-paris', 'Publication', 'Editeur_de_la_these']
data_file = "data_agreg_fict.xlsx"

################################################################################
# Fonction pour la pipeline
################################################################################

def column_selection(X, vars_to_keep):
    return X[vars_to_keep].copy()

def feature_engineering(X, top_list, bottom_list):
    X = X.copy()
    X['Is_Top'] = X['Editeur_de_la_these'].apply(lambda x: 1 if str(x) in top_list else 0)
    X['Is_Bottom'] = X['Editeur_de_la_these'].apply(lambda x: 1 if str(x) in bottom_list else 0)
    X['Is_NonPublie'] = X['Editeur_de_la_these'].apply(lambda x: 1 if str(x) == 'non_publie' else 0)
    if 'Editeur_de_la_these' in X.columns:
        X = X.drop(columns=['Editeur_de_la_these'])
    return X

def outliers(X, stats_dict):
    X = X.copy()
    # Traitement numérique avec seuils fixes
    for col in X.select_dtypes(include=["number"]).columns:
        if col in stats_dict:
            params = stats_dict[col]
            mask_inf = X[col] < params['min']
            mask_sup = X[col] > params['max']
            
            # On clip qui si ça dépasse
            if mask_inf.any() or mask_sup.any():
                X[col] = X[col].clip(lower=params['low_limit'], upper=params['high_limit'])

    # Traitement catégoriel
    cols_categ = X.select_dtypes(include=["object"]).columns
    for col in cols_categ:
        frequences = X[col].value_counts(normalize=True)
        categories_rares = frequences[frequences < 0.01].index
        if not categories_rares.empty:
            mode_cat = frequences.index[0] 
            X[col] = X[col].replace(categories_rares, mode_cat)
    return X


def get_stats_from_train(X_train, y_train):
    #Calcul des Top/Bottom éditeurs
    temp_train = X_train.copy()
    temp_train['resultat_bin'] = y_train
    
    stats = temp_train[temp_train['Editeur_de_la_these'] != 'non_publie'].groupby('Editeur_de_la_these')['resultat_bin'].agg(['mean', 'count'])
    stats_fiables = stats[stats['count'] >= 16] 
    
    liste_top = stats_fiables.sort_values(by='mean', ascending=False).head(3).index.tolist()
    liste_bottom = stats_fiables.sort_values(by='mean', ascending=True).head(3).index.tolist()
    liste_bottom = [editeur for editeur in liste_bottom if editeur not in liste_top] #Si dans les deux, alors privilégie le top
    
    #Calcul des bornes Outliers
    stats_outliers = {}
    cols_num = X_train.select_dtypes(include=["number"]).columns
    for col in cols_num:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        median = X_train[col].median()
        stats_outliers[col] = {
            'min': median - 1.5 * IQR,
            'max': median + 1.5 * IQR,
            'low_limit': X_train[col].quantile(0.05),
            'high_limit': X_train[col].quantile(0.95)
        }
        
    return liste_top, liste_bottom, stats_outliers

################################################################################
# PHASE 1 :Pipline et modèle
################################################################################

if True:

    df = pd.read_excel(data_file)

    df["resultat_bin"] = df["resultat"].apply(lambda x: 1 if x == "admis" else 0)
    X = df.drop(columns=["resultat_bin", "resultat"]) 
    y = df["resultat_bin"]

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calcul des stat

    X_train_filtered = X_train[var_keep]
    LISTE_TOP, LISTE_BOTTOM, STATS_OUTLIERS = get_stats_from_train(X_train_filtered, y_train)

    #on créé la pipeline
    
    # Selector garde uniquement les colonnes voulues
    selector_fixed = partial(column_selection, vars_to_keep=var_keep)
    selector_transformer = FunctionTransformer(selector_fixed)

    # Feature Engineering crée les variables Top/Bottom
    features_engineering_fixed = partial(feature_engineering, top_list=LISTE_TOP, bottom_list=LISTE_BOTTOM)
    features_engineering_transformer = FunctionTransformer(features_engineering_fixed)
    
    # Outliers gère les valeurs extrêmes
    
    outliers_fixed = partial(outliers, stats_dict=STATS_OUTLIERS)
    outliers_transformer = FunctionTransformer(outliers_fixed)

    # Gère les valeurs manquantes et centre et reduit
    num_selector = make_column_selector(dtype_include=np.number)
    cat_selector = make_column_selector(dtype_include=object)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), num_selector),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_selector)
        ])
    
    # pipeline complete
    
    
    KNN_classifier = KNeighborsClassifier()

    param_grid = {
        'classifier__n_neighbors': range(1, 30),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1,2]}

    full_pipeline = Pipeline(steps=[
        ('features_engineering', features_engineering_transformer),
        ('outliers', outliers_transformer),
        ('preprocessor', preprocessor),
        ('classifier', KNN_classifier)
    ])

    
    grid_research = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='accuracy')
    ##################################################################################
    # PHASE 2 : Entrainement et tunning
    ##################################################################################

    grid_research.fit(X_train, y_train)

    print(f"Meilleurs paramètres : {grid_research.best_params_}")
    print(f"Meilleur score CV : {grid_research.best_score_:.3f}")


    Data = []
    result = grid_research.cv_results_
    sorted_indices = np.argsort(result['mean_test_score'])[::-1][:3]
    for ide, n in enumerate(sorted_indices):
        param = result['params'][n]
        rank = result['rank_test_score'][n]
        mean_score = result['mean_test_score'][n]
        std = result['std_test_score'][n]

        print('Paramètres : {}'.format(param))
        print('Rang : {}'.format(rank))
        print('Mean Test Score : {}'.format(mean_score))
        print('Standard Deviation : {}'.format(std))

        knn_classifier_ext = KNeighborsClassifier(
            n_neighbors= param['classifier__n_neighbors'],
            weights= param['classifier__weights'],
            p = param['classifier__p']
        )

        pipeline_external = Pipeline([
            ('features_engineering', features_engineering_transformer),
            ('outliers', outliers_transformer),
            ('preprocessor', preprocessor),
            ('knn', knn_classifier_ext)
        ])

        pipeline_external.fit(X_train, y_train)
        y_pred_test = pipeline_external.predict(X_test)
        accuracy_test = round(accuracy_score(y_test, y_pred_test), 5)
        print('Validation externe')
        print('Test Accuracy: {}'.format(accuracy_test))
        print('Classification Report:')
        print(classification_report(y_test, y_pred_test))
        print('--------------------')
        
        Data.append([ide, 'k_neighbors', param, rank, mean_score, std, accuracy_test])
    
    df_result = pd.DataFrame(Data, columns=['identifiant', 'model', 'parametres', 'classement CV', 'Moyenne accuracy', "Ecart-type accuracy", "Accuracy (external test)"])

    ##################################################################################
    # PHASE 3 : Selection modèle 
    ##################################################################################
    results_df = pd.DataFrame(grid_research.cv_results_)
    
    #On prend le meilleur score moyen
    best_mean_score = results_df['mean_test_score'].max()
    
    #On met un seuil de 0.5% sous le meilleur score
    threshold = best_mean_score - 0.005
    elite_models = results_df[results_df['mean_test_score'] >= threshold].copy()
    
    #On prend le modèle le plus stable parmi les meilleurs
    best_stable_model_row = elite_models.sort_values(by='std_test_score', ascending=True).iloc[0]
    smart_best_params = best_stable_model_row['params']
    
    print("--------------------------------")
    print("Selection modèle")
    print(f"Meilleur score (CV) : {best_mean_score:.4f}")
    print(f"Meilleur modèle avec filtres {len(elite_models)}")
    print("--------------------------------")
    print("Le plus petit std parmi les meilleurs")
    print(f"Paramètres: {smart_best_params}")
    print(f"Score moy: {best_stable_model_row['mean_test_score']:.4f}")
    print(f"STD: {best_stable_model_row['std_test_score']:.4f}")

    # Construction du modèle final
    knn_final = KNeighborsClassifier(
        n_neighbors=smart_best_params["classifier__n_neighbors"],
        weights=smart_best_params["classifier__weights"],
        p=smart_best_params["classifier__p"]
    )

    pipeline_final = Pipeline([
        ('features_engineering', features_engineering_transformer),
        ('outliers', outliers_transformer),
        ('preprocessor', preprocessor),
        ('model', knn_final)
    ])

    # train final sur tout le Train
    pipeline_final.fit(X_train, y_train)

    ##################################################################################
    # PHASE 4 : EVAL FINALE
    ##################################################################################
    print("Diagnostic du modèle final sur le test set :")
    
    # Prédictions
    y_pred_test = pipeline_final.predict(X_test)
    y_pred_train = pipeline_final.predict(X_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)

    print(f"Accuracy Test  : {acc_test:.4f}")
    print("-----------")

    # Matrice de Confusion
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("Matrice de Confusion :")
    print(cm)
    
    # Sauvegarde
    with open("pipeline_knn_final.dill", "wb") as output:
        dill.dump(pipeline_final, output)

    print("STOP")

##################################################################################
# PHASE 5 : ENTRAINEMENT FINAL SUR LA BASE ENTIERE
##################################################################################
print("PHASE 5 : ENTRAINEMENT FINAL SUR TOUTE LA BASE")

# Reconstruction du pipeline avec les mêmes paramètres optimaux
knn_production = KNeighborsClassifier(
    n_neighbors=smart_best_params["classifier__n_neighbors"],
    weights=smart_best_params["classifier__weights"],
    p=smart_best_params["classifier__p"]
)
X_full_filtered = X[var_keep]
LISTE_TOP_FULL, LISTE_BOTTOM_FULL, STATS_OUTLIERS_FULL = get_stats_from_train(X_full_filtered, y)

print(f"Statistiques mises à jour sur {len(X)} observations totales")
print(f"Top éditeurs (base complète) : {LISTE_TOP_FULL}")
print(f"Bottom éditeurs (base complète) : {LISTE_BOTTOM_FULL}")

# Création des transformateurs avec les nouvelles statistiques
features_engineering_full = partial(feature_engineering, top_list=LISTE_TOP_FULL, bottom_list=LISTE_BOTTOM_FULL)
features_engineering_transformer_full = FunctionTransformer(features_engineering_full)

outliers_full = partial(outliers, stats_dict=STATS_OUTLIERS_FULL)
outliers_transformer_full = FunctionTransformer(outliers_full)

# Pipeline client
pipeline_production = Pipeline([
    ('selector', selector_transformer),
    ('features_engineering', features_engineering_transformer_full),
    ('outliers', outliers_transformer_full),
    ('preprocessor', preprocessor),
    ('model', knn_production)
])

# ENTRAINEMENT SUR TOUTE LA BASE 100%
print("Entraînement sur base complète")
pipeline_production.fit(X, y)

y_pred_full = pipeline_production.predict(X)
acc_full = accuracy_score(y, y_pred_full)

print(f"Accuracy sur la base complète: {acc_full:.4f}")
print(f"Nombre d'observations utilisées: {len(X)}")

# Sauvegarde du modèle de PRODUCTION
with open("pipeline_knn_client.dill", "wb") as output:
    dill.dump(pipeline_production, output)

print("--------------------------------")
print("Résumé final :") 
print(f"Modèle développement (sur 80% des données): Accuracy Test = {acc_test:.4f}")
print(f"Modèle client(sur 100% desdonnées): Accuracy Full = {acc_full:.4f}")
print(f"Paramètres optimaux : {smart_best_params}")

