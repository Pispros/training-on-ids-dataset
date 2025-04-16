import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import xgboost as xgb
import time
import joblib
import shutil
import warnings
warnings.filterwarnings('ignore')

# Imports spécifiques pour Hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hpsklearn import any_classifier, any_preprocessing, gaussian_process_regressor
from hpsklearn import HyperoptEstimator

# Pour Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Définition des chemins pour Colab
input_file = "/content/drive/MyDrive/Datasets/KDD/file.csv"
temp_dir = "/content/temp"
output_dir = "/content/output/hyperopt"

# Création des répertoires s'ils n'existent pas
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "hyperopt"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "seeds"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "roc_data"), exist_ok=True)  # Nouveau dossier pour les données ROC

# Si le fichier source est unique, on le copie dans le répertoire temporaire
if os.path.exists(input_file):
    temp_file = os.path.join(temp_dir, "kdd_data.csv")
    shutil.copyfile(input_file, temp_file)
    print(f"Fichier copié vers {temp_file}")
else:
    print(f"Le fichier {input_file} n'existe pas. Veuillez vérifier le chemin.")
    print(f"Nous utiliserons les fichiers présents dans {temp_dir} s'il y en a.")

# Configuration de visualisation
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Liste des seeds à utiliser (réduite pour l'optimisation)
SEEDS = [42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768]

# Structure pour stocker les résultats
results_by_model = {
    'Logistic Regression': [],
    'Random Forest': [],
    'XGBoost': []
}

# Structure pour stocker les meilleurs paramètres
best_params = {
    'Logistic Regression': {},
    'Random Forest': {},
    'XGBoost': {}
}

# Nouvelle structure pour stocker les données ROC
roc_data_by_model = {
    'Logistic Regression': {},
    'Random Forest': {},
    'XGBoost': {}
}

# Fonction pour optimiser les hyperparamètres
def optimize_hyperparams(X_train, y_train, model_type, n_classes, max_evals=50, seed=42):
    print(f"\n{'='*30} OPTIMISATION {model_type} {'='*30}")

    # Définir l'espace de recherche en fonction du type de modèle
    if model_type == 'Logistic Regression':
        estim = HyperoptEstimator(
            classifier=any_classifier('lr'),
            preprocessing=any_preprocessing('pp'),
            algo=tpe.suggest,
            max_evals=max_evals,
            trial_timeout=180,
            seed=seed
        )
    elif model_type == 'Random Forest':
        estim = HyperoptEstimator(
            classifier=any_classifier('rf'),
            preprocessing=any_preprocessing('pp'),
            algo=tpe.suggest,
            max_evals=max_evals,
            trial_timeout=180,
            seed=seed
        )
    elif model_type == 'XGBoost':
        estim = HyperoptEstimator(
            classifier=any_classifier('xgb'),
            preprocessing=any_preprocessing('pp'),
            algo=tpe.suggest,
            max_evals=max_evals,
            trial_timeout=180,
            seed=seed
        )
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")

    # Ajuster l'estimateur pour trouver les meilleurs hyperparamètres
    print(f"Début de l'optimisation des hyperparamètres pour {model_type}...")
    start_time = time.time()
    estim.fit(X_train, y_train)
    opt_time = time.time() - start_time
    print(f"Optimisation terminée en {opt_time:.2f} secondes")

    # Récupérer et afficher les meilleurs paramètres
    best_model = estim.best_model()
    print(f"Score de validation: {estim.score(X_train, y_train):.4f}")
    print(f"Meilleurs paramètres pour {model_type}:")
    params = {}

    if model_type == 'Logistic Regression':
        # Extraire les paramètres de la régression logistique
        clf = best_model['learner']
        params = {
            'C': clf.C if hasattr(clf, 'C') else 1.0,
            'solver': clf.solver if hasattr(clf, 'solver') else 'lbfgs',
            'max_iter': clf.max_iter if hasattr(clf, 'max_iter') else 1000,
            'multi_class': clf.multi_class if hasattr(clf, 'multi_class') else 'auto'
        }
    elif model_type == 'Random Forest':
        # Extraire les paramètres du random forest
        clf = best_model['learner']
        params = {
            'n_estimators': clf.n_estimators if hasattr(clf, 'n_estimators') else 100,
            'max_depth': clf.max_depth if hasattr(clf, 'max_depth') else None,
            'min_samples_split': clf.min_samples_split if hasattr(clf, 'min_samples_split') else 2,
            'min_samples_leaf': clf.min_samples_leaf if hasattr(clf, 'min_samples_leaf') else 1
        }
    elif model_type == 'XGBoost':
        # Extraire les paramètres du XGBoost
        clf = best_model['learner']
        params = {
            'n_estimators': clf.n_estimators if hasattr(clf, 'n_estimators') else 100,
            'learning_rate': clf.learning_rate if hasattr(clf, 'learning_rate') else 0.1,
            'max_depth': clf.max_depth if hasattr(clf, 'max_depth') else 3,
            'subsample': clf.subsample if hasattr(clf, 'subsample') else 0.8,
            'colsample_bytree': clf.colsample_bytree if hasattr(clf, 'colsample_bytree') else 0.8
        }

    for key, value in params.items():
        print(f"  {key}: {value}")

    return params

# Fonction pour calculer et sauvegarder les données ROC
def calculate_roc_data(model, X_test, y_test, model_name, seed, n_classes):
    # Si classification binaire
    if n_classes == 2:
        # Pour les modèles qui supportent la méthode predict_proba
        if hasattr(model, 'predict_proba'):
            # Calculer les scores de probabilité
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculer la courbe ROC
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            return {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': roc_auc,
                'seed': seed
            }
    # Si classification multiclasse
    else:
        # Approche "one vs rest" pour multiclasse
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)

            # Initialiser les structures pour stocker les résultats
            all_fprs = []
            all_tprs = []
            all_aucs = []

            # Pour chaque classe, calculer la courbe ROC "one vs rest"
            for i in range(n_classes):
                # Créer des labels binaires pour cette classe (1 pour la classe actuelle, 0 pour les autres)
                y_test_binary = (y_test == i).astype(int)

                # Calculer la courbe ROC pour cette classe
                fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, i])
                roc_auc = auc(fpr, tpr)

                all_fprs.append(fpr.tolist())
                all_tprs.append(tpr.tolist())
                all_aucs.append(roc_auc)

            return {
                'all_fprs': all_fprs,
                'all_tprs': all_tprs,
                'all_aucs': all_aucs,
                'avg_auc': np.mean(all_aucs),
                'seed': seed
            }

    # Si le modèle ne supporte pas predict_proba ou autres cas
    return None

# Fonction pour évaluer un modèle
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, seed, n_classes):
    # Mesure du temps d'entraînement
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)

    # Gérer le cas multiclasse vs binaire pour les métriques precision/recall/f1
    if n_classes > 2:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

    # Calculer les données ROC
    roc_data = calculate_roc_data(model, X_test, y_test, model_name, seed, n_classes)

    # Affichage des résultats
    print(f"\n----- {model_name} (Seed {seed}) -----")
    print(f"Temps d'entraînement: {train_time:.2f} secondes")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Afficher AUC si disponible
    if roc_data:
        if n_classes == 2:
            print(f"AUC: {roc_data['auc']:.4f}")
        else:
            print(f"AUC moyen: {roc_data['avg_auc']:.4f}")

    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'seed': seed,
        'roc_data': roc_data
    }

# Boucle principale
print("\n" + "="*80)
print("PHASE 1: OPTIMISATION DES HYPERPARAMÈTRES")
print("="*80)

# Extraire un sous-ensemble pour l'optimisation
try:
    # Nous supposons que PreprocessTabularData est disponible
    preprocessor = PreprocessTabularData(
        data_path=temp_dir,
        target_column='label',
        max_rows=10000,  # Limité pour l'optimisation
        apply_pca=True,
        pca_components=0.95,
        apply_feature_selection=False,
        random_state=42
    )

    X_opt, y_opt, feature_names = preprocessor.preprocess()
    print(f"Dimensions des données pour l'optimisation: {X_opt.shape}")

    # Division pour l'optimisation
    unique_classes, class_counts = np.unique(y_opt, return_counts=True)
    n_classes = len(unique_classes)
    print(f"Nombre de classes: {n_classes}")

    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_opt, y_opt, test_size=0.3, random_state=42,
        stratify=y_opt if len(unique_classes[class_counts < 2]) == 0 else None
    )

    # Optimiser chaque type de modèle
    for model_type in results_by_model.keys():
        try:
            params = optimize_hyperparams(X_train_opt, y_train_opt, model_type, n_classes, max_evals=30)
            best_params[model_type] = params
        except Exception as e:
            print(f"Erreur lors de l'optimisation de {model_type}: {str(e)}")
            # Paramètres par défaut en cas d'erreur
            if model_type == 'Logistic Regression':
                best_params[model_type] = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'multi_class': 'auto'}
            elif model_type == 'Random Forest':
                best_params[model_type] = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
            elif model_type == 'XGBoost':
                best_params[model_type] = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8}

    # Sauvegarder les meilleurs paramètres
    hyperparams_df = pd.DataFrame([
        {'Model': model, **params} for model, params in best_params.items()
    ])
    hyperparams_path = os.path.join(output_dir, 'hyperopt', 'best_hyperparameters.csv')
    hyperparams_df.to_csv(hyperparams_path, index=False)
    print(f"\nMeilleurs hyperparamètres sauvegardés dans {hyperparams_path}")

except Exception as e:
    print(f"Erreur lors de la phase d'optimisation: {str(e)}")
    # En cas d'erreur, utiliser des paramètres par défaut
    best_params = {
        'Logistic Regression': {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'multi_class': 'auto'},
        'Random Forest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8}
    }

print("\n" + "="*80)
print("PHASE 2: ENTRAINEMENT DES MODÈLES AVEC HYPERPARAMÈTRES OPTIMISÉS")
print("="*80)

# Boucle principale pour chaque seed
for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\n{'='*80}")
    print(f"EXPÉRIENCE AVEC SEED {seed} ({seed_idx}/{len(SEEDS)})")
    print(f"{'='*80}")

    try:
        # Initialisation du préprocesseur avec le seed actuel
        print("\nPRÉTRAITEMENT DES DONNÉES")
        preprocessor = PreprocessTabularData(
            data_path=temp_dir,
            target_column='label',
            max_rows=50000,
            apply_pca=True,
            pca_components=0.95,
            apply_feature_selection=False,
            random_state=seed
        )

        # Prétraitement des données
        X, y, feature_names = preprocessor.preprocess()
        print(f"Dimensions finales des données: {X.shape}")

        # Déterminer si nous avons des classes rares
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)
        print(f"Nombre de classes: {n_classes}")

        # Identifier les classes avec peu de membres
        rare_classes = unique_classes[class_counts < 2]
        if len(rare_classes) > 0:
            print(f"ATTENTION: {len(rare_classes)} classes ont moins de 2 membres!")
            # Division sans stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )
        else:
            # Division standard avec stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )

        print(f"Ensemble d'entraînement: {X_train.shape}")
        print(f"Ensemble de test: {X_test.shape}")

        # Création des modèles avec les hyperparamètres optimisés
        models = []

        # Régression logistique avec paramètres optimisés
        lr_params = best_params['Logistic Regression']
        models.append({
            'name': 'Logistic Regression',
            'model': LogisticRegression(
                C=lr_params['C'],
                max_iter=lr_params['max_iter'],
                solver=lr_params['solver'],
                multi_class=lr_params['multi_class'],
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1
            )
        })

        # Random Forest avec paramètres optimisés
        rf_params = best_params['Random Forest']
        models.append({
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth'],
                min_samples_split=rf_params['min_samples_split'],
                min_samples_leaf=rf_params['min_samples_leaf'],
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1
            )
        })

        # XGBoost avec paramètres optimisés
        xgb_params = best_params['XGBoost']
        models.append({
            'name': 'XGBoost',
            'model': xgb.XGBClassifier(
                n_estimators=xgb_params['n_estimators'],
                learning_rate=xgb_params['learning_rate'],
                max_depth=xgb_params['max_depth'],
                subsample=xgb_params['subsample'],
                colsample_bytree=xgb_params['colsample_bytree'],
                tree_method='hist' if X.shape[0] > 10000 else 'auto',
                objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
                random_state=seed,
                n_jobs=-1
            )
        })

        # Évaluation des modèles
        print("\nENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
        for model_info in models:
            try:
                result = evaluate_model(
                    model_info['model'],
                    X_train, X_test, y_train, y_test,
                    model_info['name'], seed, n_classes
                )
                # Sauvegarder le résultat
                results_by_model[model_info['name']].append(result)

                # Stocker les données ROC par modèle et par seed
                if result['roc_data']:
                    roc_data_by_model[model_info['name']][seed] = result['roc_data']

                # Sauvegarder le modèle entraîné
                model_path = os.path.join(output_dir, "seeds", f"{model_info['name'].lower().replace(' ', '_')}_seed{seed}.joblib")
                joblib.dump(model_info['model'], model_path)
                print(f"Modèle sauvegardé dans {model_path}")
            except Exception as e:
                print(f"Erreur lors de l'évaluation de {model_info['name']}: {str(e)}")

    except Exception as e:
        print(f"Erreur avec le seed {seed}: {str(e)}")
        continue  # Passer au seed suivant

# Analyse des résultats
print("\n" + "="*80)
print("ANALYSE DES RÉSULTATS")
print("="*80)

# Structure pour stocker les stats
stats_data = []

# Structure pour stocker les données ROC moyennes
mean_roc_data = {}

for model_name, results in results_by_model.items():
    if not results:
        print(f"Aucun résultat pour {model_name}")
        continue

    # Calculer les moyennes et écarts types
    accuracies = [r['accuracy'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]
    train_times = [r['train_time'] for r in results]  # Ajout des temps d'entraînement

    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    mean_train_time = np.mean(train_times)  # Moyenne des temps d'entraînement

    std_accuracy = np.std(accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1_scores)
    std_train_time = np.std(train_times)  # Écart type des temps d'entraînement

    print(f"\n----- {model_name} (sur {len(results)} seeds) -----")
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Temps d'entraînement: {mean_train_time:.2f}s ± {std_train_time:.2f}s")  # Affichage des temps d'entraînement

    # Ajouter les stats pour le CSV
    stats_data.append({
        'Model': model_name,
        'Accuracy_Mean': mean_accuracy,
        'Accuracy_Std': std_accuracy,
        'Precision_Mean': mean_precision,
        'Precision_Std': std_precision,
        'Recall_Mean': mean_recall,
        'Recall_Std': std_recall,
        'F1_Mean': mean_f1,
        'F1_Std': std_f1,
        'Train_Time_Mean': mean_train_time,  # Ajout du temps moyen
        'Train_Time_Std': std_train_time,    # Ajout de l'écart type du temps
        'Num_Seeds': len(results)
    })

    # Calculer les données ROC moyennes pour les classifications binaires (n_classes = 2)
    try:
        seed_roc_data = roc_data_by_model[model_name]
        if seed_roc_data and len(seed_roc_data) > 0:
            # Vérifier si on a une classification binaire
            first_seed = list(seed_roc_data.keys())[0]
            sample_roc = seed_roc_data[first_seed]

            if 'fpr' in sample_roc:  # Classification binaire
                mean_roc_data[model_name] = {
                    'model': model_name,
                    'avg_auc': np.mean([seed_roc_data[seed]['auc'] for seed in seed_roc_data if 'auc' in seed_roc_data[seed]]),
                    'std_auc': np.std([seed_roc_data[seed]['auc'] for seed in seed_roc_data if 'auc' in seed_roc_data[seed]]),
                    'seeds': list(seed_roc_data.keys())
                }
            elif 'all_aucs' in sample_roc:  # Classification multiclasse
                mean_roc_data[model_name] = {
                    'model': model_name,
                    'avg_auc': np.mean([seed_roc_data[seed]['avg_auc'] for seed in seed_roc_data if 'avg_auc' in seed_roc_data[seed]]),
                    'std_auc': np.std([seed_roc_data[seed]['avg_auc'] for seed in seed_roc_data if 'avg_auc' in seed_roc_data[seed]]),
                    'seeds': list(seed_roc_data.keys())
                }
    except Exception as e:
        print(f"Erreur lors du calcul des données ROC moyennes pour {model_name}: {str(e)}")

# Exporter les statistiques en CSV
stats_df = pd.DataFrame(stats_data)
stats_csv_path = os.path.join(output_dir, 'model_performance_stats.csv')
stats_df.to_csv(stats_csv_path, index=False)
print(f"\nStatistiques exportées vers {stats_csv_path}")

# Exporter les données ROC pour chaque modèle et seed
print("\nExport des données ROC...")
for model_name, seed_data in roc_data_by_model.items():
    for seed, roc_data in seed_data.items():
        # Pour les classifications binaires
        if 'fpr' in roc_data:
            roc_df = pd.DataFrame({
                'false_positive_rate': roc_data['fpr'],
                'true_positive_rate': roc_data['tpr'],
                'thresholds': roc_data['thresholds']
            })
            roc_df['model'] = model_name
            roc_df['seed'] = seed
            roc_df['auc'] = roc_data['auc']

            # Sauvegarder en CSV
            roc_csv_path = os.path.join(output_dir, 'roc_data', f"{model_name.lower().replace(' ', '_')}_seed{seed}_roc.csv")
            roc_df.to_csv(roc_csv_path, index=False)
            print(f"Données ROC pour {model_name} (seed {seed}) exportées vers {roc_csv_path}")

        # Pour les classifications multiclasses
        elif 'all_fprs' in roc_data:
            for class_idx, (fpr, tpr) in enumerate(zip(roc_data['all_fprs'], roc_data['all_tprs'])):
                roc_df = pd.DataFrame({
                    'false_positive_rate': fpr,
                    'true_positive_rate': tpr
                })
                roc_df['model'] = model_name
                roc_df['seed'] = seed
                roc_df['class'] = class_idx
                roc_df['auc'] = roc_data['all_aucs'][class_idx]

                # Sauvegarder en CSV
                roc_csv_path = os.path.join(output_dir, 'roc_data', f"{model_name.lower().replace(' ', '_')}_seed{seed}_class{class_idx}_roc.csv")
                roc_df.to_csv(roc_csv_path, index=False)

            print(f"Données ROC multiclasses pour {model_name} (seed {seed}) exportées")

# Exporter un fichier CSV combinant toutes les données ROC pour faciliter les comparaisons
try:
    all_roc_data = []

    for model_name, seed_data in roc_data_by_model.items():
        for seed, roc_data in seed_data.items():
            if 'fpr' in roc_data:  # Classification binaire
                for i, (fpr, tpr) in enumerate(zip(roc_data['fpr'], roc_data['tpr'])):
                    all_roc_data.append({
                        'model': model_name,
                        'seed': seed,
                        'false_positive_rate': fpr,
                        'true_positive_rate': tpr,
                        'auc': roc_data['auc']
                    })
            elif 'all_fprs' in roc_data:  # Classification multiclasse
                for class_idx, (class_fpr, class_tpr) in enumerate(zip(roc_data['all_fprs'], roc_data['all_tprs'])):
                    for i, (fpr, tpr) in enumerate(zip(class_fpr, class_tpr)):
                        all_roc_data.append({
                            'model': model_name,
                            'seed': seed,
                            'class': class_idx,
                            'false_positive_rate': fpr,
                            'true_positive_rate': tpr,
                            'auc': roc_data['all_aucs'][class_idx]
                        })

    # Créer un DataFrame à partir de toutes les données ROC collectées
    if all_roc_data:
        all_roc_df = pd.DataFrame(all_roc_data)
        all_roc_csv_path = os.path.join(output_dir, 'roc_data', 'all_models_roc_data.csv')
        all_roc_df.to_csv(all_roc_csv_path, index=False)
        print(f"\nToutes les données ROC combinées exportées vers {all_roc_csv_path}")
except Exception as e:
    print(f"Erreur lors de la création du fichier de données ROC combinées: {str(e)}")

# Créer un CSV avec les résultats détaillés par seed
detailed_data = []
for model_name, results in results_by_model.items():
    for r in results:
        detailed_data.append({
            'Model': model_name,
            'Seed': r['seed'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1': r['f1'],
            'Train_Time': r['train_time']
        })

detailed_df = pd.DataFrame(detailed_data)
detailed_csv_path = os.path.join(output_dir, 'detailed_seed_results.csv')
detailed_df.to_csv(detailed_csv_path, index=False)
print(f"Résultats détaillés exportés vers {detailed_csv_path}")