import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, auc)
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
import time
import gc
import warnings
import json
from tqdm import tqdm
import psutil
import shutil
from google.colab import drive
drive.mount('/content/drive')

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# Définition des chemins
data_dir = "/content/drive/MyDrive/Datasets"
output_dir = "/content/output/tabs"

# Créer le répertoire de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

def get_metrics(y_true, y_pred, y_scores=None):
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

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Si les scores de probabilité sont fournis et c'est un problème binaire, calculer les données ROC
    if y_scores is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        metrics['fpr'] = fpr.tolist()
        metrics['tpr'] = tpr.tolist()
        metrics['roc_auc'] = roc_auc

    return metrics

def evaluate_model(model_name, clf, X_train, X_test, y_train, y_test):
    """Évalue un modèle et retourne les métriques et les temps d'exécution"""
    print(f"Évaluation de {model_name}...")

    # Entraînement
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Prédiction
    start_time = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - start_time

    # Récupérer les scores de probabilité si disponibles
    y_scores = None
    if hasattr(clf, "predict_proba"):
        try:
            # Pour le cas binaire, nous prenons les probabilités de la classe positive (=1)
            if len(np.unique(y_test)) == 2:
                y_scores = clf.predict_proba(X_test)[:, 1]
            else:
                # Pour les cas multiclasses, on ne calcule pas les données ROC
                y_scores = None
        except:
            y_scores = None

    # Calcul des métriques
    metrics = get_metrics(y_test, y_pred, y_scores)

    # Ajouter les temps d'exécution
    metrics['train_time'] = train_time
    metrics['predict_time'] = predict_time

    print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"  Temps d'entraînement: {train_time:.2f}s, Temps de prédiction: {predict_time:.2f}s")

    if 'roc_auc' in metrics:
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")

    return metrics, y_pred, y_scores

def main():
    print("====== ANALYSE ROBUSTE DE DÉTECTION D'INTRUSION SUR CICIDS2018 ======")
    print("Évaluation avec 10 fichiers de datasets et 5 seeds par fichier")

    # Nombre de seeds par fichier
    N_SEEDS = 5
    N_FILES = 10

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

    # Dictionnaire pour stocker tous les résultats
    all_results = {
        'TabPFN': [],
        'TabICL': []
    }

    # Dictionnaire pour stocker les données ROC
    roc_data = {
        'TabPFN': {
            'fpr_avg': None,
            'tpr_avg': None,
            'auc_avg': 0,
            'count': 0
        },
        'TabICL': {
            'fpr_avg': None,
            'tpr_avg': None,
            'auc_avg': 0,
            'count': 0
        }
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

                    # Initialisation du préprocesseur avec la seed actuelle
                    preprocessor = PreprocessTabularData(
                        data_folder_path=temp_dir,
                        max_rows=10000,  # Limite stricte à 10 000 lignes
                        target_column='Label',
                        scaler_type='robust',
                        reduce_dim=True,
                        n_components=0.95,
                        variance_threshold=0.01,
                        random_state=seed
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

                    # --- Étape 3: Évaluation des modèles ---
                    print("[3/3] Entraînement et évaluation des modèles...")

                    # Déterminer si c'est un problème binaire ou multiclasse
                    n_classes = len(np.unique(y))
                    is_binary = (n_classes == 2)

                    # 1. TabPFN Classifier
                    # Note: TabPFN est limité à 1000 échantillons et 100 caractéristiques
                    # Si les données sont plus grandes, nous devons les sous-échantillonner
                    X_train_tabpfn = X_train
                    y_train_tabpfn = y_train

                    # Si trop d'échantillons, sous-échantillonnage stratifié
                    if X_train.shape[0] > 1000:
                        print("  Sous-échantillonnage pour TabPFN (max 1000 échantillons)")
                        X_train_tabpfn, _, y_train_tabpfn, _ = train_test_split(
                            X_train, y_train, train_size=1000, random_state=seed, stratify=y_train
                        )

                    # Si trop de caractéristiques, réduction
                    if X_train.shape[1] > 100:
                        print("  Réduction de caractéristiques pour TabPFN (max 100 caractéristiques)")
                        from sklearn.feature_selection import SelectKBest, f_classif
                        selector = SelectKBest(f_classif, k=100)
                        X_train_tabpfn = selector.fit_transform(X_train_tabpfn, y_train_tabpfn)
                        X_test_tabpfn = selector.transform(X_test)
                    else:
                        X_test_tabpfn = X_test

                    tabpfn_model = TabPFNClassifier(
                        device='cuda',
                        random_state=seed
                    )

                    tabpfn_metrics, tabpfn_pred, tabpfn_scores = evaluate_model('TabPFN',
                                                                tabpfn_model,
                                                                X_train_tabpfn,
                                                                X_test_tabpfn,
                                                                y_train_tabpfn,
                                                                y_test)

                    # Ajouter des infos sur le fichier et la seed
                    tabpfn_metrics['file'] = file
                    tabpfn_metrics['file_idx'] = file_idx
                    tabpfn_metrics['seed'] = seed
                    tabpfn_metrics['seed_idx'] = seed_idx

                    all_results['TabPFN'].append(tabpfn_metrics)

                    # Ajouter les données ROC si disponibles et si c'est un problème binaire
                    if is_binary and 'fpr' in tabpfn_metrics and 'tpr' in tabpfn_metrics:
                        # Si c'est la première fois, initialiser les moyennes
                        if roc_data['TabPFN']['fpr_avg'] is None:
                            roc_data['TabPFN']['fpr_avg'] = np.linspace(0, 1, 100)
                            roc_data['TabPFN']['tpr_avg'] = np.zeros(100)

                        # Interpoler les points TPR pour les FPR standard
                        interp_tpr = np.interp(roc_data['TabPFN']['fpr_avg'],
                                              tabpfn_metrics['fpr'],
                                              tabpfn_metrics['tpr'])
                        interp_tpr[0] = 0.0  # Forcer le premier point à (0,0)

                        # Accumuler les TPR interpolés
                        roc_data['TabPFN']['tpr_avg'] += interp_tpr
                        roc_data['TabPFN']['auc_avg'] += tabpfn_metrics['roc_auc']
                        roc_data['TabPFN']['count'] += 1

                    # Sauvegarder le rapport de classification
                    with open(os.path.join(seed_output_dir, "tabpfn_classification_report.txt"), 'w') as f:
                        f.write(classification_report(y_test, tabpfn_pred))

                    # Si binaire, sauvegarder les données ROC pour cette exécution
                    if is_binary and tabpfn_scores is not None:
                        roc_data_df = pd.DataFrame({
                            'fpr': tabpfn_metrics['fpr'],
                            'tpr': tabpfn_metrics['tpr'],
                            'model': 'TabPFN',
                            'file': file,
                            'seed': seed
                        })
                        roc_data_df.to_csv(os.path.join(seed_output_dir, "tabpfn_roc_data.csv"), index=False)

                    # Libérer de la mémoire
                    del tabpfn_model, tabpfn_pred, tabpfn_scores, X_train_tabpfn, X_test_tabpfn, y_train_tabpfn
                    gc.collect()

                    # 2. TabICL Classifier
                    # TabICL a également des limites similaires, mais peut être plus flexible
                    X_train_tabicl = X_train
                    y_train_tabicl = y_train

                    # Si trop d'échantillons, sous-échantillonnage stratifié
                    if X_train.shape[0] > 1000:
                        print("  Sous-échantillonnage pour TabICL (max 1000 échantillons)")
                        X_train_tabicl, _, y_train_tabicl, _ = train_test_split(
                            X_train, y_train, train_size=1000, random_state=seed, stratify=y_train
                        )

                    # Si trop de caractéristiques, réduction
                    if X_train.shape[1] > 100:
                        print("  Réduction de caractéristiques pour TabICL (max 100 caractéristiques)")
                        from sklearn.feature_selection import SelectKBest, f_classif
                        selector = SelectKBest(f_classif, k=100)
                        X_train_tabicl = selector.fit_transform(X_train_tabicl, y_train_tabicl)
                        X_test_tabicl = selector.transform(X_test)
                    else:
                        X_test_tabicl = X_test

                    tabicl_model = TabICLClassifier(
                        random_state=seed
                    )

                    tabicl_metrics, tabicl_pred, tabicl_scores = evaluate_model('TabICL',
                                                                tabicl_model,
                                                                X_train_tabicl,
                                                                X_test_tabicl,
                                                                y_train_tabicl,
                                                                y_test)

                    # Ajouter des infos sur le fichier et la seed
                    tabicl_metrics['file'] = file
                    tabicl_metrics['file_idx'] = file_idx
                    tabicl_metrics['seed'] = seed
                    tabicl_metrics['seed_idx'] = seed_idx

                    all_results['TabICL'].append(tabicl_metrics)

                    # Ajouter les données ROC si disponibles et si c'est un problème binaire
                    if is_binary and 'fpr' in tabicl_metrics and 'tpr' in tabicl_metrics:
                        # Si c'est la première fois, initialiser les moyennes
                        if roc_data['TabICL']['fpr_avg'] is None:
                            roc_data['TabICL']['fpr_avg'] = np.linspace(0, 1, 100)
                            roc_data['TabICL']['tpr_avg'] = np.zeros(100)

                        # Interpoler les points TPR pour les FPR standard
                        interp_tpr = np.interp(roc_data['TabICL']['fpr_avg'],
                                              tabicl_metrics['fpr'],
                                              tabicl_metrics['tpr'])
                        interp_tpr[0] = 0.0  # Forcer le premier point à (0,0)

                        # Accumuler les TPR interpolés
                        roc_data['TabICL']['tpr_avg'] += interp_tpr
                        roc_data['TabICL']['auc_avg'] += tabicl_metrics['roc_auc']
                        roc_data['TabICL']['count'] += 1

                    # Sauvegarder le rapport de classification
                    with open(os.path.join(seed_output_dir, "tabicl_classification_report.txt"), 'w') as f:
                        f.write(classification_report(y_test, tabicl_pred))

                    # Si binaire, sauvegarder les données ROC pour cette exécution
                    if is_binary and tabicl_scores is not None:
                        roc_data_df = pd.DataFrame({
                            'fpr': tabicl_metrics['fpr'],
                            'tpr': tabicl_metrics['tpr'],
                            'model': 'TabICL',
                            'file': file,
                            'seed': seed
                        })
                        roc_data_df.to_csv(os.path.join(seed_output_dir, "tabicl_roc_data.csv"), index=False)

                    # Libérer de la mémoire
                    del tabicl_model, tabicl_pred, tabicl_scores, X_train_tabicl, X_test_tabicl, y_train_tabicl
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

    # --- Créer les données ROC moyennes et les sauvegarder ---
    global_roc_data = []

    for model_name, data in roc_data.items():
        if data['count'] > 0:
            # Calculer les moyennes
            data['tpr_avg'] /= data['count']
            data['auc_avg'] /= data['count']

            # Ajouter aux données globales pour le graphique
            for i in range(len(data['fpr_avg'])):
                global_roc_data.append({
                    'model': model_name,
                    'fpr': data['fpr_avg'][i],
                    'tpr': data['tpr_avg'][i]
                })

            # Sauvegarder les données moyennes dans un CSV
            model_roc_df = pd.DataFrame({
                'fpr': data['fpr_avg'],
                'tpr': data['tpr_avg'],
                'model': model_name
            })
            model_roc_df.to_csv(os.path.join(output_dir, f"{model_name}_avg_roc_data.csv"), index=False)

    # Créer un CSV global avec toutes les données ROC moyennes
    if global_roc_data:
        global_roc_df = pd.DataFrame(global_roc_data)
        global_roc_df.to_csv(os.path.join(output_dir, "global_roc_data.csv"), index=False)

    # --- Analyse des résultats ---
    print("\n===== ANALYSE DES RÉSULTATS =====")

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

        # Ajouter ROC AUC moyen si disponible
        if 'roc_auc' in model_df.columns:
            stats['roc_auc_mean'] = model_df['roc_auc'].mean()
            stats['roc_auc_std'] = model_df['roc_auc'].std()

        summary_stats[model_name] = stats

        # Statistiques par fichier
        file_stats = model_df.groupby('file_idx').agg({
            'accuracy': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'train_time': ['mean', 'std'],  # Ajout des statistiques de temps d'entraînement
            'predict_time': ['mean', 'std'],  # Ajout des statistiques de temps de prédiction
            'file': 'first'  # Pour conserver le nom du fichier
        })

        # Renommer les colonnes pour éviter les indices multiples
        file_stats.columns = ['_'.join(col).strip() for col in file_stats.columns]

        per_file_stats[model_name] = file_stats

        # Sauvegarder les résultats détaillés par fichier/seed
        model_df.to_csv(os.path.join(output_dir, f"{model_name.replace(' ', '_')}_detailed_results.csv"), index=False)

    # Créer un DataFrame pour le CSV final (résultats globaux)
    final_df = pd.DataFrame({
        'model': [],
        'accuracy': [],
        'accuracy_std': [],
        'precision': [],
        'precision_std': [],
        'recall': [],
        'recall_std': [],
        'f1': [],
        'f1_std': [],
        'train_time': [],
        'predict_time': []
    })

    for model_name, stats in summary_stats.items():
        row_data = {
            'model': [model_name],
            'accuracy': [stats['accuracy_mean']],
            'accuracy_std': [stats['accuracy_std']],
            'precision': [stats['precision_mean']],
            'precision_std': [stats['precision_std']],
            'recall': [stats['recall_mean']],
            'recall_std': [stats['recall_std']],
            'f1': [stats['f1_mean']],
            'f1_std': [stats['f1_std']],
            'train_time': [stats['train_time_mean']],
            'predict_time': [stats['predict_time_mean']]
        }

        # Ajouter ROC AUC si disponible
        if 'roc_auc_mean' in stats:
            row_data['roc_auc'] = [stats['roc_auc_mean']]
            row_data['roc_auc_std'] = [stats['roc_auc_std']]

        final_df = pd.concat([final_df, pd.DataFrame(row_data)], ignore_index=True)

    # Sauvegarder les résultats finaux
    results_path = os.path.join(output_dir, "model_comparison_global.csv")
    final_df.to_csv(results_path, index=False)
    print(f"Résultats globaux sauvegardés dans {results_path}")

    # Créer un tableau comparatif des temps d'entraînement
    time_comparison_df = pd.DataFrame({
        'model': [],
        'train_time_mean': [],
        'train_time_std': [],
        'predict_time_mean': [],
        'predict_time_std': []
    })

    for model_name, stats in summary_stats.items():
        time_comparison_df = pd.concat([time_comparison_df, pd.DataFrame({
            'model': [model_name],
            'train_time_mean': [stats['train_time_mean']],
            'train_time_std': [stats.get('train_time_std', 0)],
            'predict_time_mean': [stats['predict_time_mean']],
            'predict_time_std': [stats.get('predict_time_std', 0)]
        })], ignore_index=True)

    # Sauvegarder les résultats de comparaison des temps
    time_results_path = os.path.join(output_dir, "time_comparison_global.csv")
    time_comparison_df.to_csv(time_results_path, index=False)
    print(f"Comparaison des temps sauvegardée dans {time_results_path}")

    # Afficher le tableau de résultats
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\nRésultats globaux (10 fichiers × 5 seeds):")
    print(final_df.to_string(index=False))

    # Afficher la comparaison des temps
    print("\nComparaison des temps d'exécution (10 fichiers × 5 seeds):")
    print(time_comparison_df.to_string(index=False))

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

        plt.title('Comparaison de la précision des modèles (10 fichiers × 5 seeds)')
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

        plt.title('Comparaison du F1-score des modèles (10 fichiers × 5 seeds)')
        plt.ylabel('F1-score moyen')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "f1_comparison_global.png"))

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

        plt.title('Stabilité des modèles par fichier (écart-type moyen)')
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
        plt.title('Comparaison des temps d\'exécution (10 fichiers × 5 seeds)')
        plt.ylabel('Temps moyen (secondes)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_comparison_global.png"))

        # Visualisation de la courbe ROC moyenne
        if global_roc_data:
            plt.figure(figsize=(10, 8))
            global_roc_df = pd.DataFrame(global_roc_data)

            # Tracer la courbe ROC pour chaque modèle
            for model in global_roc_df['model'].unique():
                model_data = global_roc_df[global_roc_df['model'] == model]
                auc_value = roc_data[model]['auc_avg'] if model in roc_data and roc_data[model]['count'] > 0 else 0
                plt.plot(model_data['fpr'], model_data['tpr'],
                         label=f'{model} (AUC = {auc_value:.4f})',
                         linewidth=2)

            # Ajouter la ligne de référence (classifieur aléatoire)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

            plt.xlabel('Taux de faux positifs')
            plt.ylabel('Taux de vrais positifs')
            plt.title('Courbe ROC moyenne pour tous les fichiers et seeds')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "global_roc_curve.png"))

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