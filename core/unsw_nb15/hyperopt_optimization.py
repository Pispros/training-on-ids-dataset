import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from google.colab import drive

drive.mount('/content/drive')

# Définir le chemin du dossier contenant les fichiers CSV
DATA_PATH = "/content/drive/MyDrive/Datasets/UNSW"  # À adapter selon votre environnement

# Définir les seeds pour la reproductibilité
SEEDS = [42, 123, 456, 789, 1010]

# Paramètres pour l'optimisation des hyperparamètres
CV_FOLDS = 3  # Nombre de plis pour la validation croisée

# Définition des grilles de paramètres pour chaque modèle
def get_param_grids():
    """
    Définit les grilles de paramètres pour l'optimisation des hyperparamètres.

    Returns:
        dict: Dictionnaire des grilles de paramètres par modèle
    """
    param_grids = {
        "Logistic Regression": {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        "XGBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    }
    return param_grids

# Fonction pour optimiser les hyperparamètres avec GridSearchCV
def optimize_hyperparams(X_train, y_train, model_name, cv=CV_FOLDS, seed=42):
    """
    Optimise les hyperparamètres d'un modèle avec GridSearchCV.

    Args:
        X_train: Données d'entraînement
        y_train: Étiquettes d'entraînement
        model_name: Nom du modèle
        cv: Nombre de plis pour la validation croisée
        seed: Seed pour la reproductibilité

    Returns:
        dict: Meilleurs paramètres trouvés
    """
    print(f"\nOptimisation des hyperparamètres pour {model_name}...")

    # Obtenir la grille de paramètres
    param_grids = get_param_grids()
    param_grid = param_grids[model_name]

    # Créer le modèle de base
    if model_name == "Logistic Regression":
        model = LogisticRegression(random_state=seed)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=seed, n_jobs=-1)
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(random_state=seed, seed=seed, use_label_encoder=False,
                                 eval_metric='logloss', n_jobs=-1)

    # Créer un petit échantillon pour l'optimisation si le dataset est grand
    if X_train.shape[0] > 5000:
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=5000, random_state=seed, stratify=y_train
        )
    else:
        X_sample, y_sample = X_train, y_train

    # Pour éviter une grille trop grande, on peut réduire la taille de la grille pour XGBoost
    if model_name == "XGBoost" and X_sample.shape[0] > 2000:
        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [3, 7],
            'min_child_weight': [1, 5],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }

    # Optimisation avec GridSearchCV
    start_time = time.time()
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1
    )

    grid_search.fit(X_sample, y_sample)

    # Afficher les résultats
    end_time = time.time()
    print(f"Optimisation terminée en {end_time - start_time:.2f} secondes.")
    print(f"Meilleur score de validation: {grid_search.best_score_:.4f}")
    print(f"Meilleurs paramètres pour {model_name}:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    return grid_search.best_params_

# Fonction pour calculer et stocker les données ROC
def calculate_roc_data(y_true, y_proba, model_name, seed):
    """
    Calcule les points de la courbe ROC.

    Args:
        y_true: Vraies étiquettes
        y_proba: Probabilités prédites
        model_name: Nom du modèle
        seed: Seed utilisé

    Returns:
        dict: Données ROC (fpr, tpr, thresholds)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    # Stocker les données pour tracer la courbe ROC
    roc_data = {
        'model': model_name,
        'seed': seed,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': auc
    }

    return roc_data

# Fonction pour évaluer un modèle avec plusieurs seeds et hyperparamètres optimisés
def evaluate_model_with_seeds(model_class, model_name, X_train, y_train, X_test, y_test, seeds, params=None):
    """
    Évalue un modèle avec plusieurs seeds et retourne les résultats moyens.

    Args:
        model_class: Classe du modèle (LogisticRegression, RandomForestClassifier, etc.)
        model_name: Nom du modèle pour l'affichage
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        seeds: Liste des seeds à utiliser
        params: Dictionnaire des paramètres du modèle (optionnel)

    Returns:
        dict: Dictionnaire contenant les métriques de performance moyennes
    """
    results = []
    model_instances = []  # Stocke séparément les instances de modèles
    roc_results = []      # Stockage des données ROC

    for seed in seeds:
        print(f"\nÉvaluation de {model_name} avec seed={seed}")

        # Création du modèle avec les paramètres spécifiés et le seed courant
        if params:
            # Pour XGBoost, nous devons traiter le random_state différemment
            if model_name == 'XGBoost':
                model = model_class(**params, random_state=seed, seed=seed)
            else:
                model = model_class(**params, random_state=seed)
        else:
            if model_name == 'XGBoost':
                model = model_class(random_state=seed, seed=seed)
            else:
                model = model_class(random_state=seed)

        # Mesurer le temps d'entraînement
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Prédire sur les données de test
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time

        # Calculer les métriques détaillées
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Extraire les métriques du rapport
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        # Calculer la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        # Calculer l'AUC et les données ROC
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # Binary classification
                    y_proba = y_proba[:, 1]
                else:  # Multi-class - use one-vs-rest approach
                    # Convert to binary problem for ROC calculation
                    y_test_binary = (y_test == 1).astype(int)
                    y_proba = y_proba[:, 1]  # Use class index 1 probability
            else:
                # XGBoost peut utiliser predict directement pour les probs
                if model_name == "XGBoost":
                    y_proba = model.predict(X_test, output_margin=False)
                else:
                    y_proba = y_pred  # Fallback to predictions if no probabilities

            # Calcul de l'AUC
            auc = roc_auc_score(y_test, y_proba)

            # Stocker les données ROC pour tracer les courbes plus tard
            roc_data = calculate_roc_data(y_test, y_proba, model_name, seed)
            roc_results.append(roc_data)

        except Exception as e:
            print(f"Erreur lors du calcul de l'AUC: {e}")
            auc = 0.0

        # Afficher les résultats pour ce seed
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Temps d'entraînement: {train_time:.2f} s")
        print(f"  Temps de prédiction: {predict_time:.4f} s")

        # Stocker les résultats numériques seulement
        results.append({
            'seed': seed,
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'train_time': train_time,
            'predict_time': predict_time
        })

        # Stocker l'instance du modèle séparément
        model_instances.append(model)

    # Calculer les moyennes et écarts-types des métriques numériques uniquement
    df_results = pd.DataFrame(results)
    avg_results = df_results.mean(numeric_only=True)
    std_results = df_results.std(numeric_only=True)

    print(f"\nRésultats moyens pour {model_name} sur {len(seeds)} seeds:")
    print(f"  Accuracy: {avg_results['accuracy']:.4f} (±{std_results['accuracy']:.4f})")
    print(f"  Precision: {avg_results['precision']:.4f} (±{std_results['precision']:.4f})")
    print(f"  Recall: {avg_results['recall']:.4f} (±{std_results['recall']:.4f})")
    print(f"  F1-score: {avg_results['f1']:.4f} (±{std_results['f1']:.4f})")
    print(f"  AUC: {avg_results['auc']:.4f} (±{std_results['auc']:.4f})")
    print(f"  Temps d'entraînement: {avg_results['train_time']:.2f} s (±{std_results['train_time']:.2f})")
    print(f"  Temps de prédiction: {avg_results['predict_time']:.4f} s (±{std_results['predict_time']:.4f})")

    # Trouver l'index du meilleur modèle (basé sur F1-score)
    best_model_idx = df_results['f1'].idxmax()
    best_model = model_instances[best_model_idx]

    return {
        'model_name': model_name,
        'results': df_results,
        'avg_accuracy': avg_results['accuracy'],
        'std_accuracy': std_results['accuracy'],
        'avg_precision': avg_results['precision'],
        'std_precision': std_results['precision'],
        'avg_recall': avg_results['recall'],
        'std_recall': std_results['recall'],
        'avg_f1': avg_results['f1'],
        'std_f1': std_results['f1'],
        'avg_auc': avg_results['auc'],
        'std_auc': std_results['auc'],
        'avg_train_time': avg_results['train_time'],
        'std_train_time': std_results['train_time'],
        'avg_predict_time': avg_results['predict_time'],
        'std_predict_time': std_results['predict_time'],
        'best_model': best_model,
        'roc_data': roc_results
    }

# Exporter les données ROC en CSV pour pouvoir tracer les courbes plus tard
def export_roc_data(all_results, output_dir="/content/output/hyperopt"):
    """
    Exporte les données ROC pour chaque modèle et chaque seed en CSV,
    ainsi qu'un fichier global avec toutes les données ROC combinées.

    Args:
        all_results: Dictionnaire contenant les résultats pour chaque fichier
        output_dir: Répertoire de sortie
    """
    os.makedirs(output_dir, exist_ok=True)

    # Préparer un dictionnaire pour stocker les données ROC globales
    global_roc_data = {}

    # Pour chaque fichier
    for file_name, file_results in all_results.items():
        file_base = file_name.replace('.csv', '')

        # Pour chaque modèle
        for model_name, model_result in file_results['model_results'].items():
            if 'roc_data' in model_result:
                roc_data_list = model_result['roc_data']

                # Pour chaque seed
                for roc_data in roc_data_list:
                    seed = roc_data['seed']

                    # Créer un DataFrame pour les données ROC
                    roc_df = pd.DataFrame({
                        'fpr': roc_data['fpr'],
                        'tpr': roc_data['tpr'],
                        'thresholds': roc_data['thresholds']
                    })

                    # Exporter en CSV
                    output_file = f"{output_dir}/{file_base}_{model_name.replace(' ', '_').lower()}_seed{seed}_roc.csv"
                    roc_df.to_csv(output_file, index=False)

                    # Ajouter aux données globales
                    if model_name not in global_roc_data:
                        global_roc_data[model_name] = []

                    # Ajouter ces informations aux données globales
                    global_roc_data[model_name].append({
                        'file': file_name,
                        'seed': seed,
                        'fpr': roc_data['fpr'],
                        'tpr': roc_data['tpr'],
                        'auc': roc_data['auc']
                    })

                # Créer aussi un fichier résumé avec l'AUC par seed
                auc_summary = pd.DataFrame([
                    {'file': file_name, 'model': model_name, 'seed': data['seed'], 'auc': data['auc']}
                    for data in roc_data_list
                ])

                output_file = f"{output_dir}/{file_base}_{model_name.replace(' ', '_').lower()}_auc_summary.csv"
                auc_summary.to_csv(output_file, index=False)

    # Exporter les données ROC globales
    # Créer un grand DataFrame avec toutes les données ROC
    all_roc_rows = []

    for model_name, model_data_list in global_roc_data.items():
        for model_data in model_data_list:
            # Interpoler les données sur une grille FPR commune pour faciliter l'analyse
            common_fpr = np.linspace(0, 1, 100)
            interp_tpr = np.interp(common_fpr, model_data['fpr'], model_data['tpr'])

            for i, (fpr, tpr) in enumerate(zip(common_fpr, interp_tpr)):
                all_roc_rows.append({
                    'model': model_name,
                    'file': model_data['file'],
                    'seed': model_data['seed'],
                    'auc': model_data['auc'],
                    'fpr': fpr,
                    'tpr': tpr,
                    'point_index': i  # Pour faciliter le regroupement par point de la courbe
                })

    # Créer le DataFrame global
    global_roc_df = pd.DataFrame(all_roc_rows)

    # Exporter le DataFrame global
    output_file = f"{output_dir}/global_roc_data.csv"
    global_roc_df.to_csv(output_file, index=False)

    # Créer également un résumé AUC global par modèle et fichier
    global_auc_summary = global_roc_df[['model', 'file', 'seed', 'auc']].drop_duplicates()

    # Calculer les statistiques AUC par modèle
    model_auc_stats = global_auc_summary.groupby('model')['auc'].agg(['mean', 'std', 'min', 'max']).reset_index()
    model_auc_stats.columns = ['model', 'mean_auc', 'std_auc', 'min_auc', 'max_auc']

    # Exporter le résumé AUC global
    output_file = f"{output_dir}/global_auc_summary.csv"
    model_auc_stats.to_csv(output_file, index=False)

    print(f"\nDonnées ROC individuelles et globales exportées dans {output_dir}")
    print(f"Fichier global créé: global_roc_data.csv")
    print(f"Résumé AUC global créé: global_auc_summary.csv")

# Fonction modifiée pour évaluer les modèles pour chaque fichier CSV avec optimisation des hyperparamètres
def evaluate_models_for_csv_files(directory, seeds, max_samples=None):
    """
    Évalue les modèles pour chaque fichier CSV dans un répertoire avec optimisation des hyperparamètres.

    Args:
        directory: Chemin du répertoire contenant les fichiers CSV
        seeds: Liste des seeds à utiliser
        max_samples: Nombre maximal d'échantillons à utiliser (None = tous)

    Returns:
        dict: Dictionnaire contenant les résultats pour chaque fichier
    """
    # Récupérer la liste des fichiers CSV
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    if not csv_files:
        raise ValueError(f"Aucun fichier CSV trouvé dans {directory}")

    all_results = {}
    all_best_params = {}
    all_training_times = []  # Pour collecter les temps d'entraînement

    # Pour chaque fichier CSV
    for file in csv_files:
        file_path = os.path.join(directory, file)
        print(f"\n{'='*50}")
        print(f"Évaluation des modèles pour {file}")
        print(f"{'='*50}")

        # Initialiser le préprocesseur
        preprocessor = PreprocessTabularData(max_samples=max_samples)

        # Charger les données
        print(f"Chargement des données de {file}...")
        try:
            df = preprocessor.load_data(file_path)
            print(f"Dimensions du dataset: {df.shape}")
        except Exception as e:
            print(f"Erreur lors du chargement de {file}: {e}")
            continue

        # Vérifier la distribution de la variable cible
        print("\nDistribution de la variable cible:")
        print(df['label'].value_counts())

        # Diviser les données en ensembles d'entraînement et de test
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        print(f"Taille de l'ensemble d'entraînement: {train_data.shape}")
        print(f"Taille de l'ensemble de test: {test_data.shape}")

        # Prétraiter les données
        print("\nPrétraitement des données...")
        try:
            X_train, y_train = preprocessor.preprocess(train_data, fit=True)
            X_test, y_test = preprocessor.preprocess(test_data, fit=False)
            print(f"Dimensions après prétraitement: X_train {X_train.shape}, X_test {X_test.shape}")
        except Exception as e:
            print(f"Erreur lors du prétraitement: {e}")
            continue

        # Définir les modèles à évaluer
        models = {
            "Logistic Regression": {
                'class': LogisticRegression,
            },
            "Random Forest": {
                'class': RandomForestClassifier,
            },
            "XGBoost": {
                'class': xgb.XGBClassifier,
            }
        }

        # Optimiser les hyperparamètres pour chaque modèle
        best_params = {}
        for model_name, model_info in models.items():
            opt_params = optimize_hyperparams(X_train, y_train, model_name, cv=CV_FOLDS)
            best_params[model_name] = opt_params

        # Stocker les meilleurs hyperparamètres
        all_best_params[file] = best_params

        # Évaluer chaque modèle avec les hyperparamètres optimisés
        file_results = {}
        file_training_times = []

        for model_name, model_info in models.items():
            result = evaluate_model_with_seeds(
                model_info['class'],
                model_name,
                X_train, y_train,
                X_test, y_test,
                seeds,
                best_params[model_name]
            )
            file_results[model_name] = result

            # Collecter les temps d'entraînement
            file_training_times.append({
                'file': file,
                'model': model_name,
                'avg_train_time': result['avg_train_time'],
                'std_train_time': result['std_train_time'],
                'avg_predict_time': result['avg_predict_time'],
                'std_predict_time': result['std_predict_time']
            })

        all_training_times.extend(file_training_times)

        # Stocker les résultats pour ce fichier
        all_results[file] = {
            'file_name': file,
            'model_results': file_results,
            'preprocessor': preprocessor
        }

        # Créer un tableau comparatif
        comparison = []
        for model_name, result in file_results.items():
            comparison.append({
                'Model': model_name,
                'Avg Accuracy': f"{result['avg_accuracy']:.4f} (±{result['std_accuracy']:.4f})",
                'Avg Precision': f"{result['avg_precision']:.4f} (±{result['std_precision']:.4f})",
                'Avg Recall': f"{result['avg_recall']:.4f} (±{result['std_recall']:.4f})",
                'Avg F1': f"{result['avg_f1']:.4f} (±{result['std_f1']:.4f})",
                'Avg AUC': f"{result['avg_auc']:.4f} (±{result['std_auc']:.4f})",
                'Avg Train Time (s)': f"{result['avg_train_time']:.2f} (±{result['std_train_time']:.2f})"
            })

        comparison_df = pd.DataFrame(comparison)
        print("\nComparaison des modèles:")
        print(comparison_df)

        # Visualiser les résultats
        plt.figure(figsize=(14, 8))

        # Accuracy par seed et modèle
        plt.subplot(1, 2, 1)
        for model_name, result in file_results.items():
            plt.plot(result['results']['seed'], result['results']['accuracy'], 'o-', label=model_name)
        plt.xlabel('Seed')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy par seed pour {file}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # F1-score par seed et modèle
        plt.subplot(1, 2, 2)
        for model_name, result in file_results.items():
            plt.plot(result['results']['seed'], result['results']['f1'], 'o-', label=model_name)
        plt.xlabel('Seed')
        plt.ylabel('F1-Score')
        plt.title(f'F1-Score par seed pour {file}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

        # Visualisation des temps d'entraînement par modèle
        plt.figure(figsize=(10, 6))
        times_df = pd.DataFrame(file_training_times)

        plt.bar(times_df['model'], times_df['avg_train_time'],
                yerr=times_df['std_train_time'], capsize=5)
        plt.xlabel('Modèle')
        plt.ylabel('Temps d\'entraînement (s)')
        plt.title(f'Temps d\'entraînement par modèle pour {file}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()

        # Trouver le meilleur modèle pour ce fichier (basé sur F1-score)
        best_model_name = max(file_results.items(), key=lambda x: x[1]['avg_f1'])[0]
        best_model = file_results[best_model_name]['best_model']

        print(f"\nLe meilleur modèle pour {file} est {best_model_name}")

        # Sauvegarder le meilleur modèle
        model_filename = f'/content/output/hyperopt/unsw_nb15_{file.replace(".csv", "")}_{best_model_name.replace(" ", "_").lower()}.pkl'
        preprocessor_filename = f'/content/output/hyperopt/unsw_nb15_{file.replace(".csv", "")}_preprocessor.pkl'
        params_filename = f'/content/output/hyperopt/unsw_nb15_{file.replace(".csv", "")}_best_params.pkl'

        print(f"Sauvegarde du meilleur modèle dans {model_filename}...")
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Sauvegarde du préprocesseur dans {preprocessor_filename}...")
        with open(preprocessor_filename, 'wb') as f:
            pickle.dump(preprocessor, f)

        print(f"Sauvegarde des meilleurs hyperparamètres dans {params_filename}...")
        with open(params_filename, 'wb') as f:
            pickle.dump(best_params, f)

    # Exporter les temps d'entraînement en CSV
    times_df = pd.DataFrame(all_training_times)
    times_df.to_csv('/content/output/hyperopt/unsw_nb15_training_times.csv', index=False)
    print(f"Temps d'entraînement exportés dans unsw_nb15_training_times.csv")

    # Visualiser la comparaison globale des temps d'entraînement
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='avg_train_time', hue='file', data=times_df)
    plt.xlabel('Modèle')
    plt.ylabel('Temps d\'entraînement moyen (s)')
    plt.title('Comparaison des temps d\'entraînement par modèle et fichier')
    plt.xticks(rotation=15)
    plt.legend(title='Fichier', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('/content/output/hyperopt/unsw_nb15_training_times_comparison.png', dpi=300)
    plt.show()

    return all_results, all_best_params

# Fonction pour exporter les métriques en CSV
def export_metrics_to_csv(results, output_file="/content/output/hyperopt/unsw_nb15_model_metrics.csv"):
    """
    Exporte les métriques des modèles dans un fichier CSV.

    Args:
        results: Dictionnaire contenant les résultats pour chaque fichier
        output_file: Nom du fichier CSV de sortie
    """
    # Préparer les données pour le CSV
    csv_rows = []

    for file, file_results in results.items():
        for model_name, model_result in file_results['model_results'].items():
            csv_rows.append({
                'File': file,
                'Model': model_name,
                'Accuracy': model_result['avg_accuracy'],
                'Accuracy_Std': model_result['std_accuracy'],
                'Precision': model_result['avg_precision'],
                'Precision_Std': model_result['std_precision'],
                'Recall': model_result['avg_recall'],
                'Recall_Std': model_result['std_recall'],
                'F1_Score': model_result['avg_f1'],
                'F1_Score_Std': model_result['std_f1'],
                'AUC': model_result['avg_auc'],
                'AUC_Std': model_result['std_auc'],
                'Train_Time': model_result['avg_train_time'],
                'Train_Time_Std': model_result['std_train_time'],
                'Predict_Time': model_result['avg_predict_time'],
                'Predict_Time_Std': model_result['std_predict_time']
            })

    # Créer un DataFrame et exporter en CSV
    metrics_df = pd.DataFrame(csv_rows)
    metrics_df.to_csv(output_file, index=False)
    print(f"\nMétriques exportées dans {output_file}")

    return metrics_df

# Exporter les hyperparamètres optimisés en CSV
def export_hyperparams_to_csv(hyperparams, output_file="/content/output/hyperopt/unsw_nb15_hyperparams.csv"):
    """
    Exporte les hyperparamètres optimisés dans un fichier CSV.

    Args:
        hyperparams: Dictionnaire des hyperparamètres optimisés
        output_file: Nom du fichier CSV de sortie
    """
    # Préparer les données pour le CSV
    csv_rows = []

    for file, file_params in hyperparams.items():
        for model_name, params in file_params.items():
            # Créer une ligne avec le fichier et le modèle
            row = {
                'File': file,
                'Model': model_name
            }

            # Ajouter chaque hyperparamètre comme une colonne
            for param_name, param_value in params.items():
                row[param_name] = param_value

            csv_rows.append(row)

    # Créer un DataFrame et exporter en CSV
    params_df = pd.DataFrame(csv_rows)
    params_df.to_csv(output_file, index=False)
    print(f"\nHyperparamètres exportés dans {output_file}")

    return params_df

# Visualiser et comparer les courbes ROC
def plot_roc_curves(all_results, output_dir="/content/output/hyperopt"):
    """
    Trace les courbes ROC moyennes pour chaque modèle sur chaque fichier,
    et une courbe ROC globale combinant tous les fichiers.

    Args:
        all_results: Dictionnaire contenant les résultats pour chaque fichier
        output_dir: Répertoire de sortie pour les graphiques
    """
    os.makedirs(output_dir, exist_ok=True)

    # Créer un dictionnaire pour stocker les données ROC globales par modèle
    global_roc_data = {}

    # Pour chaque fichier
    for file_name, file_results in all_results.items():
        file_base = file_name.replace('.csv', '')

        plt.figure(figsize=(10, 8))

        # Pour chaque modèle, tracer la courbe ROC moyenne
        for model_name, model_result in file_results['model_results'].items():
            if 'roc_data' in model_result:
                roc_data_list = model_result['roc_data']

                # Calculer la courbe ROC moyenne
                # Interpoler les courbes sur une grille de FPR commune
                common_fpr = np.linspace(0, 1, 100)
                tpr_list = []

                for roc_data in roc_data_list:
                    fpr = roc_data['fpr']
                    tpr = roc_data['tpr']
                    # Interpoler pour obtenir les TPR correspondant aux FPR communs
                    interp_tpr = np.interp(common_fpr, fpr, tpr)
                    tpr_list.append(interp_tpr)

                # Calculer la moyenne et l'écart-type
                mean_tpr = np.mean(tpr_list, axis=0)
                std_tpr = np.std(tpr_list, axis=0)

                # Ajouter aux données globales
                if model_name not in global_roc_data:
                    global_roc_data[model_name] = {
                        'tpr_curves': [],
                        'auc_values': []
                    }
                global_roc_data[model_name]['tpr_curves'].append(mean_tpr)
                global_roc_data[model_name]['auc_values'].append(model_result["avg_auc"])

                # Tracer la courbe ROC moyenne
                plt.plot(common_fpr, mean_tpr, label=f'{model_name} (AUC = {model_result["avg_auc"]:.4f})')

                # Ajouter l'intervalle de confiance
                plt.fill_between(
                    common_fpr,
                    np.maximum(0, mean_tpr - std_tpr),
                    np.minimum(1, mean_tpr + std_tpr),
                    alpha=0.2
                )

        # Ajouter la ligne de référence (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')

        plt.xlabel('Taux de faux positifs (FPR)')
        plt.ylabel('Taux de vrais positifs (TPR)')
        plt.title(f'Courbes ROC pour {file_name}')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Enregistrer la figure
        output_file = f"{output_dir}/{file_base}_roc_curves.png"
        plt.savefig(output_file, dpi=300)
        plt.close()

    # Créer la courbe ROC globale
    plt.figure(figsize=(12, 9))

    # Pour chaque modèle, tracer la courbe ROC globale
    for model_name, model_data in global_roc_data.items():
        tpr_curves = np.array(model_data['tpr_curves'])
        auc_values = np.array(model_data['auc_values'])

        # Calculer la courbe ROC moyenne globale
        global_mean_tpr = np.mean(tpr_curves, axis=0)
        global_std_tpr = np.std(tpr_curves, axis=0)
        global_mean_auc = np.mean(auc_values)
        global_std_auc = np.std(auc_values)

        # Tracer la courbe ROC globale
        plt.plot(
            common_fpr,
            global_mean_tpr,
            label=f'{model_name} (AUC = {global_mean_auc:.4f}±{global_std_auc:.4f})',
            linewidth=2
        )

        # Ajouter l'intervalle de confiance
        plt.fill_between(
            common_fpr,
            np.maximum(0, global_mean_tpr - global_std_tpr),
            np.minimum(1, global_mean_tpr + global_std_tpr),
            alpha=0.2
        )

        # Exporter les données de la courbe ROC globale en CSV
        global_roc_df = pd.DataFrame({
            'fpr': common_fpr,
            'tpr': global_mean_tpr,
            'tpr_std': global_std_tpr
        })
        global_roc_df.to_csv(f"{output_dir}/global_{model_name.replace(' ', '_').lower()}_roc.csv", index=False)

    # Ajouter la ligne de référence (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)

    plt.xlabel('Taux de faux positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de vrais positifs (TPR)', fontsize=12)
    plt.title('Courbes ROC globales pour tous les jeux de données', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Enregistrer la figure globale
    output_file = f"{output_dir}/global_roc_curves.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Courbes ROC individuelles et globale enregistrées dans {output_dir}")

# Programme principal modifié
def main():
    print(f"Évaluation des modèles sur les données UNSW-NB15 avec {len(SEEDS)} seeds et optimisation des hyperparamètres...")
    print(f"Chemin des données: {DATA_PATH}")
    print(f"Seeds utilisés: {SEEDS}")
    print(f"Nombre de plis pour la validation croisée: {CV_FOLDS}")

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs("/content/output/hyperopt", exist_ok=True)

    # Exécuter l'évaluation pour tous les fichiers CSV
    results, best_params = evaluate_models_for_csv_files(DATA_PATH, SEEDS, max_samples=10000)

    # Exporter les données ROC
    export_roc_data(results)

    # Visualiser les courbes ROC
    plot_roc_curves(results)

    # Exporter les métriques en CSV
    metrics_df = export_metrics_to_csv(results)
    print("\nAperçu du fichier CSV exporté:")
    print(metrics_df.head())

    # Exporter les hyperparamètres optimisés
    params_df = export_hyperparams_to_csv(best_params)
    print("\nAperçu des hyperparamètres exportés:")
    print(params_df.head())

    # Créer un tableau récapitulatif par modèle (toutes données confondues)
    model_summary = metrics_df.groupby('Model').agg({
        'Accuracy': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'F1_Score': ['mean', 'std'],
        'AUC': ['mean', 'std'],
        'Train_Time': 'mean',
        'Predict_Time': 'mean'
    }).reset_index()

    # Reformater les colonnes pour plus de clarté
    model_summary.columns = [
        'Model',
        'Global_Accuracy', 'Global_Accuracy_Std',
        'Global_Precision', 'Global_Precision_Std',
        'Global_Recall', 'Global_Recall_Std',
        'Global_F1', 'Global_F1_Std',
        'Global_AUC', 'Global_AUC_Std',
        'Global_Train_Time',
        'Global_Predict_Time'
    ]

    # Sauvegarder le récapitulatif global
    model_summary.to_csv("/content/output/hyperopt/unsw_nb15_global_metrics.csv", index=False)
    print("\nRécapitulatif global exporté dans unsw_nb15_global_metrics.csv")
    print("\nAperçu du récapitulatif global:")
    print(model_summary)

    # Visualiser les métriques globales avec barres d'erreur
    plt.figure(figsize=(15, 10))

    # Accuracy
    plt.subplot(2, 2, 1)
    plt.errorbar(
        model_summary['Model'],
        model_summary['Global_Accuracy'],
        yerr=model_summary['Global_Accuracy_Std'],
        fmt='o', capsize=5, elinewidth=2, markeredgewidth=2
    )
    plt.title('Accuracy Globale avec Écart-Type')
    plt.ylim(0.8, 1.0)  # Ajuster selon vos résultats
    plt.grid(True, linestyle='--', alpha=0.7)

    # Precision
    plt.subplot(2, 2, 2)
    plt.errorbar(
        model_summary['Model'],
        model_summary['Global_Precision'],
        yerr=model_summary['Global_Precision_Std'],
        fmt='o', capsize=5, elinewidth=2, markeredgewidth=2
    )
    plt.title('Precision Globale avec Écart-Type')
    plt.ylim(0.8, 1.0)  # Ajuster selon vos résultats
    plt.grid(True, linestyle='--', alpha=0.7)

    # Recall
    plt.subplot(2, 2, 3)
    plt.errorbar(
        model_summary['Model'],
        model_summary['Global_Recall'],
        yerr=model_summary['Global_Recall_Std'],
        fmt='o', capsize=5, elinewidth=2, markeredgewidth=2
    )
    plt.title('Recall Global avec Écart-Type')
    plt.ylim(0.8, 1.0)  # Ajuster selon vos résultats
    plt.grid(True, linestyle='--', alpha=0.7)

    # F1-Score
    plt.subplot(2, 2, 4)
    plt.errorbar(
        model_summary['Model'],
        model_summary['Global_F1'],
        yerr=model_summary['Global_F1_Std'],
        fmt='o', capsize=5, elinewidth=2, markeredgewidth=2
    )
    plt.title('F1-Score Global avec Écart-Type')
    plt.ylim(0.8, 1.0)  # Ajuster selon vos résultats
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('/content/output/hyperopt/unsw_nb15_global_metrics.png', dpi=300)
    plt.show()

    # Visualiser la comparaison des temps d'entraînement et de prédiction
    plt.figure(figsize=(12, 6))

    # Temps d'entraînement
    plt.subplot(1, 2, 1)
    bar_positions = np.arange(len(model_summary))
    plt.bar(bar_positions, model_summary['Global_Train_Time'])
    plt.xticks(bar_positions, model_summary['Model'])
    plt.title('Temps d\'entraînement global par modèle')
    plt.ylabel('Temps (s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Temps de prédiction
    plt.subplot(1, 2, 2)
    plt.bar(bar_positions, model_summary['Global_Predict_Time'])
    plt.xticks(bar_positions, model_summary['Model'])
    plt.title('Temps de prédiction global par modèle')
    plt.ylabel('Temps (s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('/content/output/hyperopt/unsw_nb15_timing_comparison.png', dpi=300)
    plt.show()

    # Identifier le meilleur modèle global basé sur le F1-score
    best_model_name = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Model']
    best_f1 = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_F1']
    best_f1_std = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_F1_Std']
    best_accuracy = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_Accuracy']
    best_accuracy_std = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_Accuracy_Std']

    print(f"\nLe meilleur modèle global est {best_model_name} avec:")
    print(f"  - F1-score: {best_f1:.4f} (±{best_f1_std:.4f})")
    print(f"  - Accuracy: {best_accuracy:.4f} (±{best_accuracy_std:.4f})")

    # Créer une matrice de comparaison des métriques par modèle
    comparison_table = metrics_df.pivot_table(
        index='File',
        columns='Model',
        values=['Accuracy', 'F1_Score'],
        aggfunc='mean'
    )

    # Créer une heatmap pour visualiser les performances par fichier et modèle
    plt.figure(figsize=(12, 8))

    # Heatmap pour l'accuracy
    plt.subplot(1, 2, 1)
    sns.heatmap(comparison_table['Accuracy'], annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title('Accuracy par fichier et modèle')

    # Heatmap pour le F1-score
    plt.subplot(1, 2, 2)
    sns.heatmap(comparison_table['F1_Score'], annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title('F1-Score par fichier et modèle')

    plt.tight_layout()
    plt.savefig('/content/output/hyperopt/unsw_nb15_performance_heatmap.png', dpi=300)
    plt.show()

    # Visualisation de comparaison des hyperparamètres optimisés
    plt.figure(figsize=(15, 10))

    # Pour chaque modèle, visualiser les paramètres les plus importants
    model_key_params = {
        "Logistic Regression": ['C', 'penalty'],
        "Random Forest": ['n_estimators', 'max_depth'],
        "XGBoost": ['learning_rate', 'max_depth']
    }

    plot_idx = 1
    for model_name, key_params in model_key_params.items():
        for param in key_params:
            if plot_idx <= 6:  # Limiter à 6 graphiques
                plt.subplot(2, 3, plot_idx)

                param_values = []
                file_labels = []

                for file, file_params in best_params.items():
                    if model_name in file_params and param in file_params[model_name]:
                        param_values.append(file_params[model_name][param])
                        file_labels.append(file.replace('.csv', ''))

                if param_values:
                    plt.bar(range(len(param_values)), param_values)
                    plt.xticks(range(len(param_values)), file_labels, rotation=45)
                    plt.title(f'{model_name}: {param}')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)

                plot_idx += 1

    plt.tight_layout()
    plt.savefig('/content/output/hyperopt/unsw_nb15_hyperparams_comparison.png', dpi=300)
    plt.show()

    print("\nAnalyse terminée avec succès!")
    print(f"Récapitulatif des fichiers générés:")
    print(f"  - unsw_nb15_model_metrics.csv: Métriques détaillées pour chaque fichier et modèle")
    print(f"  - unsw_nb15_global_metrics.csv: Métriques globales moyennes par modèle")
    print(f"  - unsw_nb15_hyperparams.csv: Hyperparamètres optimisés pour chaque modèle et fichier")
    print(f"  - unsw_nb15_training_times.csv: Temps d'entraînement et de prédiction par modèle et fichier")
    print(f"  - unsw_nb15_global_metrics.png: Visualisation des métriques globales")
    print(f"  - unsw_nb15_performance_heatmap.png: Heatmap des performances par fichier et modèle")
    print(f"  - unsw_nb15_hyperparams_comparison.png: Comparaison des hyperparamètres optimisés")
    print(f"  - unsw_nb15_timing_comparison.png: Comparaison des temps d'entraînement et de prédiction")
    print(f"  - unsw_nb15_training_times_comparison.png: Visualisation des temps d'entraînement par modèle et fichier")
    print(f"  - *_roc_curves.png: Courbes ROC pour chaque fichier")
    print(f"  - *_roc.csv: Données des courbes ROC pour chaque modèle et seed")
    print(f"  - *_auc_summary.csv: Résumé des AUC par modèle et seed")
    print(f"  - Fichiers .pkl: Modèles, préprocesseurs et hyperparamètres sauvegardés pour chaque fichier")

# Exécuter le programme principal si ce script est exécuté directement
if __name__ == "__main__":
    main()