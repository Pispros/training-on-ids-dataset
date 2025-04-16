import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import time
import gc
import warnings
import json
from tqdm import tqdm
import psutil
import shutil
from google.colab import drive
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt.pyll import scope
import pickle


# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# Monter Google Drive
drive.mount('/content/drive')

# Définition des chemins
data_dir = "/content/drive/MyDrive/Datasets"
output_dir = "/content/output/hyperopt"

# Créer le répertoire de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

def get_metrics(y_true, y_pred, y_proba=None):
    """Calcule toutes les métriques demandées pour une prédiction"""
    accuracy = accuracy_score(y_true, y_pred)

    # Gérer le cas multiclasse ou binaire pour precision, recall et f1
    if len(np.unique(y_true)) > 2:
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    else:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Calcul des données ROC pour les problèmes binaires
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            # Utiliser la colonne de la classe positive pour les calculs ROC
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_score = y_proba[:, 1]  # Proba de la classe positive (1)
            else:
                y_score = y_proba  # Déjà un vecteur

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            metrics_dict['fpr'] = fpr
            metrics_dict['tpr'] = tpr
            metrics_dict['thresholds'] = thresholds
            metrics_dict['roc_auc'] = roc_auc
        except Exception as e:
            print(f"Erreur lors du calcul des métriques ROC: {e}")

    return metrics_dict

def create_hyperopt_space(model_type, is_binary, n_classes, seed):
    """Définir l'espace de recherche pour Hyperopt"""
    if model_type == 'Logistic Regression':
        space = {
            'C': hp.loguniform('C', np.log(0.001), np.log(10.0)),
            'solver': hp.choice('solver', ['liblinear', 'saga']),
            'max_iter': scope.int(hp.quniform('max_iter', 100, 2000, 100)),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'penalty': hp.choice('penalty', ['l1', 'l2']),
            'dual': False,  # Force dual=False pour éviter les problèmes numériques
            'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-3)),
        }

    elif model_type == 'Random Forest':
        space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
            'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
        }

    elif model_type == 'XGBoost':
        objective = 'binary:logistic' if is_binary else 'multi:softmax'
        num_class = None if is_binary else n_classes

        space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'objective': objective,
            'num_class': num_class,
            'tree_method': 'hist',  # Plus efficace en mémoire
            'use_label_encoder': False,
        }

    return space

def build_model(model_type, params, seed):
    """Construire le modèle en fonction des paramètres"""
    if model_type == 'Logistic Regression':
        return LogisticRegression(
            random_state=seed,
            n_jobs=-1,
            **params
        )

    elif model_type == 'Random Forest':
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **params
        )

    elif model_type == 'XGBoost':
        return xgb.XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            **params
        )

def objective_function(params, X_train, y_train, X_test, y_test, model_type, seed):
    """Fonction objectif pour Hyperopt"""
    try:
        # Construire le modèle
        model = build_model(model_type, params, seed)

        # Temps d'entraînement
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Temps de prédiction
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time

        # Récupération des probabilités pour ROC
        y_proba = None
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
        except Exception as e:
            print(f"Impossible de calculer les probabilités pour ROC: {e}")

        # Calculer les métriques
        metrics = get_metrics(y_test, y_pred, y_proba)
        metrics['train_time'] = train_time
        metrics['predict_time'] = predict_time

        # Nous cherchons à maximiser le F1-score (Hyperopt minimise par défaut)
        return {
            'loss': -metrics['f1'],  # Négatif car Hyperopt minimise
            'status': STATUS_OK,
            'metrics': metrics,
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba,
            'params': params
        }
    except Exception as e:
        print(f"Erreur lors de l'évaluation d'un jeu de paramètres: {e}")
        # Retourner une valeur très élevée pour indiquer que ces paramètres sont à éviter
        return {
            'loss': 1.0,  # La pire valeur possible pour un score (normalement entre 0 et 1)
            'status': STATUS_OK,
            'error': str(e)
        }

def optimize_hyperparameters(model_type, X_train, X_test, y_train, y_test, max_evals=20, seed=42):
    """Optimiser les hyperparamètres avec Hyperopt (implémentation corrigée)"""
    print(f"Optimisation des hyperparamètres pour {model_type}...")

    # Déterminer si c'est un problème binaire ou multiclasse
    n_classes = len(np.unique(y_train))
    is_binary = (n_classes == 2)

    # Définir l'espace de recherche
    space = create_hyperopt_space(model_type, is_binary, n_classes, seed)

    # Créer l'objet Trials pour stocker les résultats
    trials = Trials()

    # Définir la fonction objectif avec les données fixes
    objective = lambda params: objective_function(params, X_train, y_train, X_test, y_test, model_type, seed)

    # CORRECTION: Utiliser np.random.RandomState(seed) au lieu de rstate
    # ou utiliser le paramètre random_state directement pour tpe.suggest
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        # CORRECTION: Supprimer rstate=np.random.RandomState(seed)
        # ou le remplacer par random_state=seed si disponible
    )

    # Vérifier si nous avons des résultats valides
    valid_trials = [t for t in trials.trials if 'model' in t['result']]

    if not valid_trials:
        raise ValueError(f"Aucun essai valide trouvé pour {model_type}. Vérifiez les données et les paramètres.")

    # Identifier le meilleur essai
    best_trial_idx = np.argmin([t['result']['loss'] for t in valid_trials])
    best_trial = valid_trials[best_trial_idx]['result']

    best_metrics = best_trial['metrics']
    best_model = best_trial['model']
    best_predictions = best_trial['predictions']
    best_probabilities = best_trial.get('probabilities', None)  # Récupérer les probas si disponibles
    best_params = best_trial['params']

    print(f"  Meilleurs paramètres pour {model_type}:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    print(f"  Meilleure accuracy: {best_metrics['accuracy']:.4f}, F1: {best_metrics['f1']:.4f}")
    if 'roc_auc' in best_metrics:
        print(f"  AUC-ROC: {best_metrics['roc_auc']:.4f}")
    print(f"  Temps d'entraînement: {best_metrics['train_time']:.2f}s, Temps de prédiction: {best_metrics['predict_time']:.2f}s")

    return best_model, best_params, best_metrics, best_predictions, best_probabilities, trials

def export_roc_data(model_name, metrics, file_idx, seed, output_dir):
    """Exporte les données ROC dans un fichier CSV"""
    if 'fpr' in metrics and 'tpr' in metrics:
        roc_data = pd.DataFrame({
            'model': model_name,
            'file_idx': file_idx,
            'seed': seed,
            'fpr': metrics['fpr'],
            'tpr': metrics['tpr'],
            'auc': metrics['roc_auc']
        })

        # Créer le répertoire si nécessaire
        roc_dir = os.path.join(output_dir, "roc_data")
        os.makedirs(roc_dir, exist_ok=True)

        # Sauvegarder le CSV
        roc_path = os.path.join(roc_dir, f"{model_name.replace(' ', '_')}_file_{file_idx}_seed_{seed}_roc.csv")
        roc_data.to_csv(roc_path, index=False)

        return roc_path
    return None

def plot_global_roc_curves(all_roc_data, output_dir):
    """Création d'une courbe ROC globale pour tous les fichiers"""
    try:
        plt.figure(figsize=(12, 10))

        models = all_roc_data['model'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

        # Tracer les courbes ROC pour chaque modèle (moyenne de toutes les seeds et fichiers)
        for i, model_name in enumerate(models):
            model_data = all_roc_data[all_roc_data['model'] == model_name]

            # Calculer la courbe ROC moyenne et l'écart-type
            # On va interpoler les FPR pour obtenir des points uniformes
            mean_tpr = np.linspace(0, 1, 100)
            fprs = []
            tprs = []
            aucs = []

            # Grouper par file_idx et seed
            for (file_idx, seed), group in model_data.groupby(['file_idx', 'seed']):
                if len(group) < 2:
                    continue

                # Trier par FPR pour l'interpolation
                group = group.sort_values('fpr')

                # Extraire les valeurs
                fpr = group['fpr'].values
                tpr = group['tpr'].values

                # Ajouter l'AUC à la liste
                if 'auc' in group.columns and not group['auc'].isnull().all():
                    aucs.append(group['auc'].iloc[0])

                # Interpoler pour obtenir des TPR aux FPR standards
                interp_tpr = np.interp(mean_tpr, fpr, tpr)
                interp_tpr[0] = 0.0  # Forcer le point (0,0)
                tprs.append(interp_tpr)

            # Calculer la moyenne et l'écart-type des TPR interpolés
            if tprs:
                mean_tpr_values = np.mean(tprs, axis=0)
                std_tpr = np.std(tprs, axis=0)

                # Limites supérieure et inférieure pour l'intervalle de confiance
                tprs_upper = np.minimum(mean_tpr_values + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr_values - std_tpr, 0)

                # Calculer l'AUC moyen
                mean_auc = np.mean(aucs) if aucs else auc(mean_tpr, mean_tpr_values)
                std_auc = np.std(aucs) if aucs else 0

                # Tracer la courbe ROC moyenne
                plt.plot(
                    mean_tpr, mean_tpr_values,
                    label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
                    color=colors[i], lw=2, alpha=0.8
                )

                # Tracer l'intervalle de confiance
                plt.fill_between(
                    mean_tpr, tprs_lower, tprs_upper,
                    color=colors[i], alpha=0.2,
                    label=f'±1 std. dev.'
                )

        # Ajouter la diagonale (ligne du hasard)
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Aléatoire')

        # Paramètres du graphique
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('Taux de faux positifs (1 - Spécificité)')
        plt.ylabel('Taux de vrais positifs (Sensibilité)')
        plt.title('Courbe ROC Moyenne pour Tous les Modèles (Tous fichiers, Toutes seeds)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)

        # Sauvegarder la figure
        global_roc_path = os.path.join(output_dir, "global_roc_curve.png")
        plt.savefig(global_roc_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Courbe ROC globale sauvegardée dans {global_roc_path}")

    except Exception as e:
        print(f"Erreur lors de la création de la courbe ROC globale: {e}")

def main():
    print("====== ANALYSE ROBUSTE DE DÉTECTION D'INTRUSION SUR CICIDS2018 AVEC HYPEROPT ======")
    print("Évaluation avec 10 fichiers de datasets et 5 seeds par fichier")

    # Nombre de seeds par fichier
    N_SEEDS = 5
    N_FILES = 10
    MAX_HYPEROPT_EVALS = 20  # Nombre d'évaluations pour l'optimisation d'hyperparamètres

    # Identifier tous les fichiers CSV dans le répertoire de données
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if len(all_files) < N_FILES:
        print(f"ATTENTION: Seulement {len(all_files)} fichiers CSV trouvés. L'analyse sera limitée à ce nombre.")
        file_count = len(all_files)
    else:
        file_count = N_FILES
        all_files = all_files[:N_FILES]  # Limiter aux premiers fichiers

    print(f"Fichiers de données disponibles ({file_count}):")
    for i, file in enumerate(all_files):
        print(f"  {i+1}. {file}")
    print(f"Chaque fichier sera évalué avec {N_SEEDS} seeds différentes")
    print(f"Optimisation d'hyperparamètres avec {MAX_HYPEROPT_EVALS} évaluations par modèle et par seed")

    # Dictionnaire pour stocker tous les résultats
    all_results = {
        'Logistic Regression': [],
        'Random Forest': [],
        'XGBoost': []
    }

    # DataFrame pour collecter toutes les données ROC
    all_roc_data = pd.DataFrame()

    best_models_per_file = {
        'Logistic Regression': {},
        'Random Forest': {},
        'XGBoost': {}
    }

    # Pour chaque fichier
    for file_idx, file in enumerate(all_files):
        print(f"\n\n===== FICHIER {file_idx+1}/{file_count}: {file} =====")

        # Création d'un sous-répertoire pour ce fichier
        file_output_dir = os.path.join(output_dir, f"file_{file_idx+1}")
        os.makedirs(file_output_dir, exist_ok=True)

        # Chemin du fichier actuel
        file_path = os.path.join(data_dir, file)

        # Créer un dossier temporaire pour ce fichier
        temp_dir = os.path.join(output_dir, f"temp_dir_{file_idx}")
        os.makedirs(temp_dir, exist_ok=True)

        # Copier le fichier dans le dossier temporaire
        temp_file_path = os.path.join(temp_dir, "current_file.csv")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        try:
            # Copier le fichier
            shutil.copy(file_path, temp_file_path)

            # Pour chaque seed
            for seed_idx in range(N_SEEDS):
                seed = 42 + (file_idx * N_SEEDS) + seed_idx
                print(f"\n--- Seed {seed_idx+1}/{N_SEEDS} (valeur: {seed}) ---")

                try:
                    # --- Étape 1: Prétraitement des données ---
                    print(f"[1/3] Prétraitement des données avec seed {seed}...")

                    # Utilisation de la classe PreprocessTabularData qui existe déjà
                    preprocessor = PreprocessTabularData(
                        data_folder_path=temp_dir,
                        max_rows=10000,  # Limite à 10 000 lignes
                        target_column='Label',
                        scaler_type='robust',
                        reduce_dim=True,
                        n_components=0.95,
                        variance_threshold=0.01,
                        random_state=seed,
                        use_gpu=True
                    )

                    # Chargement et prétraitement des données
                    X_processed, y = preprocessor.process()

                    print(f"Données prétraitées: {X_processed.shape[0]} échantillons, {X_processed.shape[1]} caractéristiques")

                    # --- Étape 2: Division train/test ---
                    print("[2/3] Division des données en ensembles d'entraînement et de test...")

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y, test_size=0.3, random_state=seed, stratify=y
                    )

                    print(f"Ensemble d'entraînement: {X_train.shape[0]} échantillons")
                    print(f"Ensemble de test: {X_test.shape[0]} échantillons")

                    # Créer un sous-répertoire pour cette seed
                    seed_output_dir = os.path.join(file_output_dir, f"seed_{seed}")
                    os.makedirs(seed_output_dir, exist_ok=True)

                    # --- Étape 3: Évaluation des modèles avec Hyperopt ---
                    print("[3/3] Optimisation des hyperparamètres et évaluation des modèles...")

                    for model_name in all_results.keys():
                        try:
                            # Vérifier si la distribution des classes est très déséquilibrée
                            class_distribution = np.bincount(y_train.astype(int))
                            imbalance_ratio = max(class_distribution) / min(class_distribution) if min(class_distribution) > 0 else float('inf')

                            if imbalance_ratio > 10 and model_name == 'Logistic Regression':
                                print(f"ATTENTION: Forte imbalance des classes (ratio {imbalance_ratio:.2f}). Estimation manuelle pour Logistic Regression.")

                                # Paramètres fixes pour éviter les problèmes numériques
                                lr_model = LogisticRegression(
                                    C=0.1,
                                    class_weight='balanced',
                                    solver='liblinear',
                                    penalty='l2',
                                    dual=False,
                                    max_iter=2000,
                                    random_state=seed,
                                    n_jobs=-1
                                )

                                # Entraînement
                                start_time = time.time()
                                lr_model.fit(X_train, y_train)
                                train_time = time.time() - start_time

                                # Prédiction
                                start_time = time.time()
                                lr_pred = lr_model.predict(X_test)
                                predict_time = time.time() - start_time

                                # Récupération des probabilités pour ROC
                                lr_proba = None
                                try:
                                    if hasattr(lr_model, "predict_proba"):
                                        lr_proba = lr_model.predict_proba(X_test)
                                except Exception as e:
                                    print(f"Impossible de calculer les probabilités pour ROC: {e}")

                                # Métriques
                                metrics = get_metrics(y_test, lr_pred, lr_proba)
                                metrics['train_time'] = train_time
                                metrics['predict_time'] = predict_time

                                # Paramètres manuels
                                best_params = {
                                    'C': 0.1,
                                    'class_weight': 'balanced',
                                    'solver': 'liblinear',
                                    'penalty': 'l2',
                                    'dual': False,
                                    'max_iter': 2000
                                }

                                # Ajouter aux résultats
                                metrics['file'] = file
                                metrics['file_idx'] = file_idx
                                metrics['seed'] = seed
                                metrics['seed_idx'] = seed_idx
                                all_results[model_name].append(metrics)

                                # Exporter les données ROC
                                if 'fpr' in metrics and 'tpr' in metrics:
                                    roc_file_path = export_roc_data(model_name, metrics, file_idx, seed, output_dir)
                                    print(f"  Données ROC exportées dans {roc_file_path}")

                                    # Ajouter aux données ROC globales
                                    temp_roc_df = pd.DataFrame({
                                        'model': model_name,
                                        'file_idx': file_idx,
                                        'seed': seed,
                                        'fpr': metrics['fpr'],
                                        'tpr': metrics['tpr'],
                                        'auc': metrics['roc_auc']
                                    })
                                    all_roc_data = pd.concat([all_roc_data, temp_roc_df])

                                # Sauvegarder les informations
                                with open(os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_best_params.json"), 'w') as f:
                                    json.dump(best_params, f, indent=2)

                                with open(os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_classification_report.txt"), 'w') as f:
                                    f.write(classification_report(y_test, lr_pred))

                                # Matrice de confusion
                                plt.figure(figsize=(10, 8))
                                cm = confusion_matrix(y_test, lr_pred)
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                                plt.title(f'Matrice de confusion - {model_name}')
                                plt.ylabel('Vraie classe')
                                plt.xlabel('Classe prédite')
                                plt.tight_layout()
                                plt.savefig(os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.png"))
                                plt.close()

                                # Courbe ROC pour problème binaire
                                if 'fpr' in metrics and 'tpr' in metrics:
                                    plt.figure(figsize=(8, 8))
                                    plt.plot(metrics['fpr'], metrics['tpr'],
                                             label=f'AUC = {metrics["roc_auc"]:.3f}')
                                    plt.plot([0, 1], [0, 1], 'k--')
                                    plt.xlabel('Taux de faux positifs')
                                    plt.ylabel('Taux de vrais positifs')
                                    plt.title(f'Courbe ROC - {model_name}')
                                    plt.legend(loc='lower right')
                                    plt.grid(True, linestyle='--', alpha=0.5)
                                    plt.savefig(os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_roc_curve.png"))
                                    plt.close()

                                # Stocker dans les meilleurs modèles
                                file_key = f"file_{file_idx}"
                                if file_key not in best_models_per_file[model_name]:
                                    best_models_per_file[model_name][file_key] = []

                                best_models_per_file[model_name][file_key].append({
                                    'seed': seed,
                                    'f1': metrics['f1'],
                                    'manual_config': True,
                                    'params': best_params
                                })

                                # Sauvegarder le modèle
                                model_path = os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_best_model.pkl")
                                with open(model_path, 'wb') as f:
                                    pickle.dump(lr_model, f)

                                # Libérer mémoire
                                del lr_model, lr_pred
                                if lr_proba is not None:
                                    del lr_proba
                                gc.collect()

                                print(f"  Configuration manuelle pour {model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                                if 'roc_auc' in metrics:
                                    print(f"  AUC-ROC: {metrics['roc_auc']:.4f}")

                            elif model_name in ['Random Forest', 'XGBoost'] or imbalance_ratio <= 10:
                                # Optimisation normale avec Hyperopt
                                best_model, best_params, best_metrics, best_predictions, best_probabilities, trials = optimize_hyperparameters(
                                    model_name, X_train, X_test, y_train, y_test,
                                    max_evals=MAX_HYPEROPT_EVALS, seed=seed
                                )

                                # Ajouter des infos sur le fichier et la seed
                                best_metrics['file'] = file
                                best_metrics['file_idx'] = file_idx
                                best_metrics['seed'] = seed
                                best_metrics['seed_idx'] = seed_idx

                                # Stocker les résultats
                                all_results[model_name].append(best_metrics)

                                # Exporter les données ROC
                                if 'fpr' in best_metrics and 'tpr' in best_metrics:
                                    roc_file_path = export_roc_data(model_name, best_metrics, file_idx, seed, output_dir)
                                    print(f"  Données ROC exportées dans {roc_file_path}")

                                    # Ajouter aux données ROC globales
                                    temp_roc_df = pd.DataFrame({
                                        'model': model_name,
                                        'file_idx': file_idx,
                                        'seed': seed,
                                        'fpr': best_metrics['fpr'],
                                        'tpr': best_metrics['tpr'],
                                        'auc': best_metrics['roc_auc']
                                    })
                                    all_roc_data = pd.concat([all_roc_data, temp_roc_df])

                                # Stocker les meilleurs modèles par fichier
                                file_key = f"file_{file_idx}"
                                if file_key not in best_models_per_file[model_name]:
                                    best_models_per_file[model_name][file_key] = []

                                best_models_per_file[model_name][file_key].append({
                                    'seed': seed,
                                    'f1': best_metrics['f1'],
                                    'params': best_params
                                })

                                # Sauvegarder le rapport de classification
                                with open(os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_classification_report.txt"), 'w') as f:
                                    f.write(classification_report(y_test, best_predictions))

                                # Sauvegarder les meilleurs paramètres
                                with open(os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_best_params.json"), 'w') as f:
                                    json.dump(best_params, f, indent=2, default=str)

                                # Sauvegarder le meilleur modèle
                                model_path = os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_best_model.pkl")
                                with open(model_path, 'wb') as f:
                                    pickle.dump(best_model, f)

                                # Tracer la matrice de confusion
                                plt.figure(figsize=(10, 8))
                                cm = confusion_matrix(y_test, best_predictions)
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                                plt.title(f'Matrice de confusion - {model_name}')
                                plt.ylabel('Vraie classe')
                                plt.xlabel('Classe prédite')
                                plt.tight_layout()
                                plt.savefig(os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.png"))
                                plt.close()

                                # Courbe ROC pour problème binaire
                                if 'fpr' in best_metrics and 'tpr' in best_metrics:
                                    plt.figure(figsize=(8, 8))
                                    plt.plot(best_metrics['fpr'], best_metrics['tpr'],
                                             label=f'AUC = {best_metrics["roc_auc"]:.3f}')
                                    plt.plot([0, 1], [0, 1], 'k--')
                                    plt.xlabel('Taux de faux positifs')
                                    plt.ylabel('Taux de vrais positifs')
                                    plt.title(f'Courbe ROC - {model_name}')
                                    plt.legend(loc='lower right')
                                    plt.grid(True, linestyle='--', alpha=0.5)
                                    plt.savefig(os.path.join(seed_output_dir, f"{model_name.replace(' ', '_')}_roc_curve.png"))
                                    plt.close()

                                # Libérer de la mémoire
                                del best_model, best_predictions, trials
                                if best_probabilities is not None:
                                    del best_probabilities
                                gc.collect()

                        except Exception as e:
                            print(f"ERREUR lors de l'évaluation de {model_name}: {e}")
                            continue

                    # Libérer de la mémoire
                    del X_processed, X_train, X_test, y_train, y_test, y
                    gc.collect()

                except Exception as e:
                    print(f"ERREUR avec la seed {seed}: {e}")
                    continue

        except Exception as e:
            print(f"ERREUR lors du traitement du fichier {file}: {e}")
        finally:
            # Nettoyer le dossier temporaire
            try:
                shutil.rmtree(temp_dir)
            except:
                print(f"Impossible de supprimer le dossier temporaire {temp_dir}")

    # --- Analyse des résultats ---
    print("\n===== ANALYSE DES RÉSULTATS =====")

    # Sauvegarder toutes les données ROC collectées
    if not all_roc_data.empty:
        all_roc_data_path = os.path.join(output_dir, "all_roc_data.csv")
        all_roc_data.to_csv(all_roc_data_path, index=False)
        print(f"Toutes les données ROC sauvegardées dans {all_roc_data_path}")

        # Générer la courbe ROC globale
        plot_global_roc_curves(all_roc_data, output_dir)

    # Sauvegarder les meilleurs modèles par fichier
    with open(os.path.join(output_dir, "best_models_per_file.json"), 'w') as f:
        json.dump(best_models_per_file, f, indent=2, default=str)

    # Calculer les statistiques pour chaque modèle
    summary_stats = {}
    per_file_stats = {}

    for model_name, results_list in all_results.items():
        if not results_list:
            print(f"Aucun résultat pour {model_name}, ignoré dans les statistiques")
            continue

        # Convertir en DataFrame pour faciliter les calculs
        model_df = pd.DataFrame(results_list)

        # Résultats globaux (toutes seeds, tous fichiers)
        stats = {
            'accuracy_mean': model_df['accuracy'].mean(),
            'accuracy_std': model_df['accuracy'].std(),
            'precision_mean': model_df['precision'].mean(),
            'precision_std': model_df['precision'].std(),
            'recall_mean': model_df['recall'].mean(),
            'recall_std': model_df['recall'].std(),
            'f1_mean': model_df['f1'].mean(),
            'f1_std': model_df['f1'].std(),
            'train_time_mean': model_df['train_time'].mean(),
            'predict_time_mean': model_df['predict_time'].mean()
        }

        # Ajouter les statistiques ROC si disponibles
        roc_available = 'roc_auc' in model_df.columns
        if roc_available:
            stats['roc_auc_mean'] = model_df['roc_auc'].mean()
            stats['roc_auc_std'] = model_df['roc_auc'].std()

        summary_stats[model_name] = stats

        # Statistiques par fichier
        agg_dict = {
            'accuracy': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'train_time': ['mean', 'std'],
            'file': 'first'  # Pour conserver le nom du fichier
        }

        # Ajouter ROC AUC aux agrégations si disponible
        if roc_available:
            agg_dict['roc_auc'] = ['mean', 'std']

        file_stats = model_df.groupby('file_idx').agg(agg_dict)

        # Renommer les colonnes pour éviter les indices multiples
        file_stats.columns = ['_'.join(col).strip() for col in file_stats.columns]

        per_file_stats[model_name] = file_stats

        # Sauvegarder les résultats détaillés par fichier/seed
        model_df.to_csv(os.path.join(output_dir, f"{model_name.replace(' ', '_')}_detailed_results.csv"), index=False)

    # Créer un DataFrame pour le CSV final (résultats globaux)
    final_df_columns = [
        'model', 'accuracy', 'accuracy_std', 'precision', 'precision_std',
        'recall', 'recall_std', 'f1', 'f1_std', 'train_time', 'predict_time'
    ]

    # Ajouter ROC AUC aux colonnes si disponible pour au moins un modèle
    roc_in_any_model = any('roc_auc_mean' in stats for stats in summary_stats.values())
    if roc_in_any_model:
        final_df_columns.extend(['roc_auc', 'roc_auc_std'])

    final_df = pd.DataFrame(columns=final_df_columns)

    for model_name, stats in summary_stats.items():
        row_data = {
            'model': model_name,
            'accuracy': stats['accuracy_mean'],
            'accuracy_std': stats['accuracy_std'],
            'precision': stats['precision_mean'],
            'precision_std': stats['precision_std'],
            'recall': stats['recall_mean'],
            'recall_std': stats['recall_std'],
            'f1': stats['f1_mean'],
            'f1_std': stats['f1_std'],
            'train_time': stats['train_time_mean'],
            'predict_time': stats['predict_time_mean']
        }

        # Ajouter ROC AUC si disponible
        if 'roc_auc_mean' in stats:
            row_data['roc_auc'] = stats['roc_auc_mean']
            row_data['roc_auc_std'] = stats['roc_auc_std']

        final_df = pd.concat([final_df, pd.DataFrame([row_data])], ignore_index=True)

    # Sauvegarder les résultats finaux
    results_path = os.path.join(output_dir, "model_comparison_global.csv")
    final_df.to_csv(results_path, index=False)
    print(f"Résultats globaux sauvegardés dans {results_path}")

    # Afficher le tableau de résultats
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\nRésultats globaux (10 fichiers × 5 seeds avec Hyperopt):")
    print(final_df.to_string(index=False))

    # Créer des visualisations
    try:
        # Visualisation de l'accuracy avec écart-type
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='model', y='accuracy', data=final_df, palette='viridis')

        # Ajouter les barres d'erreur pour l'écart-type
        for i, row in final_df.iterrows():
            ax.errorbar(i, row['accuracy'], yerr=row['accuracy_std'], fmt='o', color='black')

        # Ajouter les valeurs sur les barres
        for i, v in enumerate(final_df['accuracy']):
            ax.text(i, v + 0.01, f"{v:.4f} ± {final_df['accuracy_std'][i]:.4f}", ha='center')

        plt.title('Comparaison de la précision des modèles avec Hyperopt (10 fichiers × 5 seeds)')
        plt.ylabel('Accuracy moyenne')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_comparison_global.png"))

        # Visualisation du F1-score avec écart-type
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='model', y='f1', data=final_df, palette='magma')

        # Ajouter les barres d'erreur pour l'écart-type
        for i, row in final_df.iterrows():
            ax.errorbar(i, row['f1'], yerr=row['f1_std'], fmt='o', color='black')

        # Ajouter les valeurs sur les barres
        for i, v in enumerate(final_df['f1']):
            ax.text(i, v + 0.01, f"{v:.4f} ± {final_df['f1_std'][i]:.4f}", ha='center')

        plt.title('Comparaison du F1-score des modèles avec Hyperopt (10 fichiers × 5 seeds)')
        plt.ylabel('F1-score moyen')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "f1_comparison_global.png"))

        # Visualisation ROC AUC avec écart-type (si disponible)
        if roc_in_any_model and 'roc_auc' in final_df.columns:
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='model', y='roc_auc', data=final_df, palette='coolwarm')

            # Ajouter les barres d'erreur pour l'écart-type
            for i, row in final_df.iterrows():
                if 'roc_auc' in row and not pd.isna(row['roc_auc']):
                    ax.errorbar(i, row['roc_auc'], yerr=row['roc_auc_std'], fmt='o', color='black')

            # Ajouter les valeurs sur les barres
            for i, row in final_df.iterrows():
                if 'roc_auc' in row and not pd.isna(row['roc_auc']):
                    ax.text(i, row['roc_auc'] + 0.01,
                           f"{row['roc_auc']:.4f} ± {row['roc_auc_std']:.4f}",
                           ha='center')

            plt.title('Comparaison de l\'AUC ROC des modèles avec Hyperopt (10 fichiers × 5 seeds)')
            plt.ylabel('AUC ROC moyen')
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "roc_auc_comparison_global.png"))

        # Visualisation de la stabilité par fichier (écart-type moyen)
        stability_data = []
        for model, file_stats in per_file_stats.items():
            avg_std = file_stats['accuracy_std'].mean()
            stability_data.append({
                'model': model,
                'avg_std': avg_std
            })

        stability_df = pd.DataFrame(stability_data)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='model', y='avg_std', data=stability_df, palette='Set2')

        plt.title('Stabilité des modèles par fichier avec Hyperopt (écart-type moyen)')
        plt.ylabel('Écart-type moyen de l\'accuracy')
        plt.ylim(0, stability_df['avg_std'].max() * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_stability.png"))

        # Comparaison des temps d'exécution
        plt.figure(figsize=(12, 6))
        time_df = final_df[['model', 'train_time', 'predict_time']]
        time_df_melted = pd.melt(time_df, id_vars=['model'], var_name='time_type', value_name='seconds')
        sns.barplot(x='model', y='seconds', hue='time_type', data=time_df_melted, palette='Set2')
        plt.title('Comparaison des temps d\'exécution avec Hyperopt (10 fichiers × 5 seeds)')
        plt.ylabel('Temps moyen (secondes)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_comparison_global.png"))

        # Visualisation plus détaillée des temps d'entraînement pour chaque modèle
        # Créer un DataFrame avec les temps d'entraînement par modèle et par fichier
        train_times_data = []
        for model_name, results_list in all_results.items():
            if not results_list:
                continue

            for result in results_list:
                train_times_data.append({
                    'model': model_name,
                    'file_idx': result['file_idx'],
                    'seed': result['seed'],
                    'train_time': result['train_time']
                })

        if train_times_data:
            train_times_df = pd.DataFrame(train_times_data)

            # Temps d'entraînement par modèle et par fichier
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='file_idx', y='train_time', hue='model', data=train_times_df, palette='viridis')
            plt.title('Temps d\'entraînement par fichier et par modèle')
            plt.xlabel('Indice du fichier')
            plt.ylabel('Temps d\'entraînement (secondes)')
            plt.legend(title='Modèle')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "train_time_by_file.png"))

            # Exporter les données de temps d'entraînement
            train_times_df.to_csv(os.path.join(output_dir, "train_times_detailed.csv"), index=False)

        print(f"Visualisations sauvegardées dans {output_dir}")

    except Exception as e:
        print(f"Erreur lors de la création des visualisations: {e}")

    print("\n====== ANALYSE TERMINÉE ======")
    print(f"Tous les résultats ont été sauvegardés dans {output_dir}")


if __name__ == "__main__":
    try:
        # Mesurer l'utilisation de la mémoire avant l'exécution
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # en MB
        print(f"Mémoire utilisée au démarrage: {mem_before:.2f} MB")

        # Exécuter l'analyse
        main()

        # Mesurer l'utilisation de la mémoire après l'exécution
        mem_after = process.memory_info().rss / (1024 * 1024)  # en MB
        print(f"Mémoire utilisée à la fin: {mem_after:.2f} MB")
        print(f"Différence de mémoire: {mem_after - mem_before:.2f} MB")
    except Exception as e:
        print(f"Erreur dans le programme principal: {e}")