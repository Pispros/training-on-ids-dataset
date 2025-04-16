import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from google.colab import drive

drive.mount('/content/drive')

# Créer le répertoire de sortie s'il n'existe pas déjà
os.makedirs('/content/output/training', exist_ok=True)
os.makedirs('/content/output/roc_data', exist_ok=True)

# Définir le chemin du dossier contenant les fichiers CSV
DATA_PATH = "/content/drive/MyDrive/Datasets/UNSW"  # À adapter selon votre environnement

# Définir les seeds pour la reproductibilité
SEEDS = [42, 123, 456, 789, 1010]

# Fonction pour calculer et sauvegarder les données ROC
def save_roc_data(model, model_name, X_test, y_test, file_name, seed):
    """
    Calcule et sauvegarde les données ROC pour un modèle.

    Args:
        model: Modèle entraîné
        model_name: Nom du modèle
        X_test: Données de test
        y_test: Étiquettes de test
        file_name: Nom du fichier de données
        seed: Seed utilisé

    Returns:
        dict: Dictionnaire contenant les données ROC
    """
    try:
        # Obtenir les probabilités de prédiction
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif model_name == "XGBoost":
            y_proba = model.predict(X_test, output_margin=False)
        else:
            print(f"Le modèle {model_name} ne prend pas en charge predict_proba.")
            return None

        # Calculer les taux de faux positifs et vrais positifs
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        # Calculer l'AUC
        auc = roc_auc_score(y_test, y_proba)

        # Créer un DataFrame pour stocker les résultats
        roc_df = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'model': model_name,
            'file': file_name,
            'seed': seed,
            'auc': [auc] * len(fpr)
        })

        # Sauvegarder le DataFrame
        output_file = f'/content/output/roc_data/roc_data_{file_name.replace(".csv", "")}_{model_name.replace(" ", "_").lower()}_seed{seed}.csv'
        roc_df.to_csv(output_file, index=False)
        print(f"Données ROC sauvegardées dans {output_file}")

        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        }

    except Exception as e:
        print(f"Erreur lors du calcul des données ROC: {e}")
        return None

# Fonction pour évaluer un modèle avec plusieurs seeds
def evaluate_model_with_seeds(model_class, model_name, X_train, y_train, X_test, y_test, seeds, file_name, params=None):
    """
    Évalue un modèle avec plusieurs seeds et retourne les résultats moyens.

    Args:
        model_class: Classe du modèle (LogisticRegression, RandomForestClassifier, etc.)
        model_name: Nom du modèle pour l'affichage
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        seeds: Liste des seeds à utiliser
        file_name: Nom du fichier de données
        params: Dictionnaire des paramètres du modèle (optionnel)

    Returns:
        dict: Dictionnaire contenant les métriques de performance moyennes
    """
    results = []
    model_instances = []  # Stocke séparément les instances de modèles
    roc_data_collection = []  # Pour stocker les données ROC

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

        # Calculer l'AUC et sauvegarder les données ROC
        try:
            roc_data = save_roc_data(model, model_name, X_test, y_test, file_name, seed)
            if roc_data:
                auc = roc_data['auc']
                roc_data_collection.append(roc_data)
            else:
                auc = 0.0
        except Exception as e:
            print(f"Erreur lors du calcul de l'AUC: {e}")
            auc = 0.0

        # Calculer la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        # Extraire les TP, FP, TN, FN de la matrice de confusion
        if cm.shape == (2, 2):  # Pour un problème binaire
            tn, fp, fn, tp = cm.ravel()
        else:  # Pour un problème multiclasse (on calcule pour chaque classe)
            tn, fp, fn, tp = None, None, None, None
            print("Note: Matrice de confusion non binaire - calcul TP, FP, TN, FN non applicable")

        # Afficher les résultats pour ce seed
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Temps d'entraînement: {train_time:.2f} s")
        print(f"  Temps de prédiction: {predict_time:.2f} s")

        if tn is not None:
            print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            print(f"  TPR (sensibilité): {tp/(tp+fn):.4f}")
            print(f"  FPR: {fp/(fp+tn):.4f}")
            print(f"  TNR (spécificité): {tn/(tn+fp):.4f}")

        # Stocker les résultats numériques
        results.append({
            'seed': seed,
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'train_time': train_time,
            'predict_time': predict_time,
            'TP': tp if tn is not None else None,
            'FP': fp if tn is not None else None,
            'TN': tn if tn is not None else None,
            'FN': fn if tn is not None else None,
            'TPR': tp/(tp+fn) if tn is not None else None,
            'FPR': fp/(fp+tn) if tn is not None else None,
            'TNR': tn/(tn+fp) if tn is not None else None
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

    if 'TPR' in avg_results:
        print(f"  TPR moyen: {avg_results['TPR']:.4f} (±{std_results['TPR']:.4f})")
        print(f"  FPR moyen: {avg_results['FPR']:.4f} (±{std_results['FPR']:.4f})")
        print(f"  TNR moyen: {avg_results['TNR']:.4f} (±{std_results['TNR']:.4f})")

    # Trouver l'index du meilleur modèle (basé sur accuracy)
    best_model_idx = df_results['accuracy'].idxmax()
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
        'avg_tpr': avg_results.get('TPR'),
        'std_tpr': std_results.get('TPR'),
        'avg_fpr': avg_results.get('FPR'),
        'std_fpr': std_results.get('FPR'),
        'avg_tnr': avg_results.get('TNR'),
        'std_tnr': std_results.get('TNR'),
        'best_model': best_model,
        'roc_data': roc_data_collection
    }

# Fonction pour évaluer les modèles pour chaque fichier CSV
def evaluate_models_for_csv_files(directory, seeds, max_samples=None):
    """
    Évalue les modèles pour chaque fichier CSV dans un répertoire.

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
    all_training_times = []  # Pour collecter les temps d'entraînement de tous les modèles

    # Pour chaque fichier CSV
    for file in csv_files:
        file_path = os.path.join(directory, file)
        print(f"\n{'='*50}")
        print(f"Évaluation des modèles pour {file}")
        print(f"{'='*50}")

        # Initialiser le préprocesseur avec notre nouvelle classe
        preprocessor = PreprocessTabularData(max_samples=max_samples)

        # Charger les données en utilisant notre méthode améliorée
        print(f"Chargement des données de {file}...")
        try:
            # Utiliser la méthode load_data améliorée avec dtype='object' et low_memory=False
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

        # Prétraiter les données avec notre méthode améliorée
        print("\nPrétraitement des données...")
        try:
            X_train, y_train = preprocessor.preprocess(train_data, fit=True)
            X_test, y_test = preprocessor.preprocess(test_data, fit=False)
            print(f"Dimensions après prétraitement: X_train {X_train.shape}, X_test {X_test.shape}")
        except Exception as e:
            print(f"Erreur lors du prétraitement: {e}")
            continue

        # Définir les modèles à évaluer avec leurs paramètres
        models = {
            "Logistic Regression": {
                'class': LogisticRegression,
                'params': {'max_iter': 1000, 'C': 1.0, 'solver': 'liblinear'}
            },
            "Random Forest": {
                'class': RandomForestClassifier,
                'params': {'n_estimators': 100, 'n_jobs': -1}
            },
            "XGBoost": {
                'class': xgb.XGBClassifier,
                'params': {'n_estimators': 100, 'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1}
            }
        }

        # Évaluer chaque modèle
        file_results = {}
        file_training_times = []  # Pour collecter les temps d'entraînement par fichier

        for model_name, model_info in models.items():
            result = evaluate_model_with_seeds(
                model_info['class'],
                model_name,
                X_train, y_train,
                X_test, y_test,
                seeds,
                file,
                model_info['params']
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

            all_training_times.append({
                'file': file,
                'model': model_name,
                'avg_train_time': result['avg_train_time'],
                'std_train_time': result['std_train_time'],
                'avg_predict_time': result['avg_predict_time'],
                'std_predict_time': result['std_predict_time']
            })

        # Stocker les résultats pour ce fichier
        all_results[file] = {
            'file_name': file,
            'model_results': file_results,
            'preprocessor': preprocessor,
            'training_times': file_training_times
        }

        # Exporter les temps d'entraînement pour ce fichier
        times_df = pd.DataFrame(file_training_times)
        times_df.to_csv(f'/content/output/training/train_times_{file.replace(".csv", "")}.csv', index=False)

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
        plt.savefig(f'/content/output/training/accuracy_f1_scores_{file.replace(".csv", "")}.png', dpi=300)
        plt.show()

        # Visualiser les temps d'entraînement
        plt.figure(figsize=(10, 6))
        for model_name, result in file_results.items():
            plt.bar(model_name, result['avg_train_time'],
                    yerr=result['std_train_time'],
                    capsize=5, alpha=0.7)
        plt.ylabel('Temps d\'entraînement (s)')
        plt.title(f'Temps d\'entraînement moyen par modèle pour {file}')
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'/content/output/training/train_times_{file.replace(".csv", "")}.png', dpi=300)
        plt.show()

        # Trouver le meilleur modèle pour ce fichier (basé sur F1-score)
        best_model_name = max(file_results.items(), key=lambda x: x[1]['avg_f1'])[0]
        best_model = file_results[best_model_name]['best_model']

        print(f"\nLe meilleur modèle pour {file} est {best_model_name}")

        # Sauvegarder le meilleur modèle
        model_filename = f'/content/output/training/unsw_nb15_{file.replace(".csv", "")}_{best_model_name.replace(" ", "_").lower()}.pkl'
        preprocessor_filename = f'/content/output/training/unsw_nb15_{file.replace(".csv", "")}_preprocessor.pkl'

        print(f"Sauvegarde du meilleur modèle dans {model_filename}...")
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Sauvegarde du préprocesseur dans {preprocessor_filename}...")
        with open(preprocessor_filename, 'wb') as f:
            pickle.dump(preprocessor, f)

    # Exporter les temps d'entraînement globaux
    all_times_df = pd.DataFrame(all_training_times)
    all_times_df.to_csv('/content/output/training/unsw_nb15_all_training_times.csv', index=False)

    return all_results

# Fonction pour exporter les métriques en CSV
def export_metrics_to_csv(results, output_file="/content/output/training/unsw_nb15_model_metrics.csv"):
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
                'Predict_Time_Std': model_result['std_predict_time'],
                'TPR': model_result.get('avg_tpr'),
                'TPR_Std': model_result.get('std_tpr'),
                'FPR': model_result.get('avg_fpr'),
                'FPR_Std': model_result.get('std_fpr'),
                'TNR': model_result.get('avg_tnr'),
                'TNR_Std': model_result.get('std_tnr')
            })

    # Créer un DataFrame et exporter en CSV
    metrics_df = pd.DataFrame(csv_rows)
    metrics_df.to_csv(output_file, index=False)
    print(f"\nMétriques exportées dans {output_file}")

    return metrics_df

# Programme principal
def main():
    print(f"Évaluation des modèles sur les données UNSW-NB15 avec {len(SEEDS)} seeds...")
    print(f"Chemin des données: {DATA_PATH}")
    print(f"Seeds utilisés: {SEEDS}")

    # Exécuter l'évaluation pour tous les fichiers CSV
    results = evaluate_models_for_csv_files(DATA_PATH, SEEDS, max_samples=10000)

    # Exporter les métriques en CSV
    metrics_df = export_metrics_to_csv(results)
    print("\nAperçu du fichier CSV exporté:")
    print(metrics_df.head())

    # Créer un tableau récapitulatif par modèle (toutes données confondues)
    model_summary = metrics_df.groupby('Model').agg({
        'Accuracy': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'F1_Score': ['mean', 'std'],
        'AUC': ['mean', 'std'],
        'Train_Time': 'mean',
        'Train_Time_Std': 'mean',
        'Predict_Time': 'mean',
        'TPR': ['mean', 'std'],
        'FPR': ['mean', 'std'],
        'TNR': ['mean', 'std']
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
        'Global_Train_Time_Std',
        'Global_Predict_Time',
        'Global_TPR', 'Global_TPR_Std',
        'Global_FPR', 'Global_FPR_Std',
        'Global_TNR', 'Global_TNR_Std'
    ]

    # Sauvegarder le récapitulatif global
    model_summary.to_csv("/content/output/training/unsw_nb15_global_metrics.csv", index=False)
    print("\nRécapitulatif global exporté dans /content/output/training/unsw_nb15_global_metrics.csv")
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
    plt.savefig('unsw_nb15_global_metrics.png', dpi=300)
    plt.show()

    # Visualiser les temps d'entraînement globaux
    plt.figure(figsize=(12, 6))

    # Créer un DataFrame pour faciliter la visualisation
    train_times_plot = pd.DataFrame({
        'Model': model_summary['Model'],
        'Train Time': model_summary['Global_Train_Time'],
        'Error': model_summary['Global_Train_Time_Std']
    })

    # Trier par temps d'entraînement (du plus rapide au plus lent)
    train_times_plot = train_times_plot.sort_values('Train Time')

    # Créer le graphique à barres
    plt.bar(range(len(train_times_plot)), train_times_plot['Train Time'],
            yerr=train_times_plot['Error'], capsize=10, alpha=0.7)
    plt.xticks(range(len(train_times_plot)), train_times_plot['Model'])
    plt.ylabel('Temps d\'entraînement moyen (s)')
    plt.title('Comparaison des temps d\'entraînement par modèle')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('unsw_nb15_training_times_comparison.png', dpi=300)
    plt.show()

    # Visualiser le rapport performance (F1) vs temps d'entraînement
    plt.figure(figsize=(10, 8))

    # Créer un DataFrame pour la visualisation
    perf_vs_time = pd.DataFrame({
        'Model': model_summary['Model'],
        'F1 Score': model_summary['Global_F1'],
        'Train Time': model_summary['Global_Train_Time']
    })

    # Créer un scatter plot
    for i, model in enumerate(perf_vs_time['Model']):
        plt.scatter(perf_vs_time.loc[i, 'Train Time'],
                   perf_vs_time.loc[i, 'F1 Score'],
                   s=100, label=model)

    # Ajouter des étiquettes pour chaque point
    for i, model in enumerate(perf_vs_time['Model']):
        plt.annotate(model,
                    (perf_vs_time.loc[i, 'Train Time'], perf_vs_time.loc[i, 'F1 Score']),
                    xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Temps d\'entraînement (s)')
    plt.ylabel('F1 Score')
    plt.title('Performance (F1) vs Temps d\'entraînement')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('unsw_nb15_performance_vs_time.png', dpi=300)
    plt.show()

    # Visualiser les courbes ROC moyennes pour chaque modèle
    plt.figure(figsize=(12, 8))

    # Charger et combiner toutes les données ROC
    all_roc_files = [f for f in os.listdir('/content/output/roc_data') if f.startswith('roc_data_')]

    if all_roc_files:
        all_roc_data = pd.DataFrame()
        for roc_file in all_roc_files:
            file_path = os.path.join('/content/output/roc_data', roc_file)
            df = pd.read_csv(file_path)
            all_roc_data = pd.concat([all_roc_data, df])

        # Calculer la moyenne des courbes ROC par modèle
        for model in all_roc_data['model'].unique():
            model_data = all_roc_data[all_roc_data['model'] == model]
            avg_auc = model_data['auc'].mean()

            # Tracer la courbe ROC moyenne
            plt.plot(model_data['fpr'], model_data['tpr'],
                    label=f'{model} (AUC = {avg_auc:.3f})', alpha=0.7)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Aléatoire')
        plt.xlabel('Taux de faux positifs (FPR)')
        plt.ylabel('Taux de vrais positifs (TPR)')
        plt.title('Courbes ROC moyennes par modèle')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('unsw_nb15_roc_curves.png', dpi=300)
        plt.show()
    else:
        print("Aucune donnée ROC trouvée pour générer les courbes")

    # Identifier le meilleur modèle global basé sur le F1-score
    best_model_name = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Model']
    best_f1 = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_F1']
    best_f1_std = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_F1_Std']
    best_accuracy = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_Accuracy']
    best_accuracy_std = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_Accuracy_Std']
    best_train_time = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_Train_Time']

    print(f"\nLe meilleur modèle global est {best_model_name} avec:")
    print(f"  - F1-score: {best_f1:.4f} (±{best_f1_std:.4f})")
    print(f"  - Accuracy: {best_accuracy:.4f} (±{best_accuracy_std:.4f})")
    print(f"  - Temps d'entraînement moyen: {best_train_time:.2f} s")

    # Créer une matrice de comparaison des métriques par modèle
    comparison_table = metrics_df.pivot_table(
        index='File',
        columns='Model',
        values=['Accuracy', 'F1_Score', 'Train_Time'],
        aggfunc='mean'
    )

    # Créer des heatmaps pour visualiser les performances par fichier et modèle
    plt.figure(figsize=(18, 12))

    # Heatmap pour l'accuracy
    plt.subplot(2, 2, 1)
    sns.heatmap(comparison_table['Accuracy'], annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title('Accuracy par fichier et modèle')

    # Heatmap pour le F1-score
    plt.subplot(2, 2, 2)
    sns.heatmap(comparison_table['F1_Score'], annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title('F1-Score par fichier et modèle')

    # Heatmap pour le temps d'entraînement
    plt.subplot(2, 2, 3)
    sns.heatmap(comparison_table['Train_Time'], annot=True, cmap='YlOrRd', fmt='.2f', linewidths=.5)
    plt.title('Temps d\'entraînement (s) par fichier et modèle')

    plt.tight_layout()
    plt.savefig('unsw_nb15_performance_heatmap.png', dpi=300)
    plt.show()

    # Créer un tableau de comparaison des TPR/FPR moyens par modèle
    if 'TPR' in metrics_df.columns and not metrics_df['TPR'].isna().all():
        plt.figure(figsize=(10, 6))
        tpr_fpr_df = metrics_df.groupby('Model')[['TPR', 'FPR']].mean().reset_index()

        # Créer un scatter plot des TPR vs FPR
        for i, model in enumerate(tpr_fpr_df['Model']):
            plt.scatter(tpr_fpr_df.loc[i, 'FPR'],
                       tpr_fpr_df.loc[i, 'TPR'],
                       s=100, label=model)
            plt.annotate(model,
                        (tpr_fpr_df.loc[i, 'FPR'], tpr_fpr_df.loc[i, 'TPR']),
                        xytext=(5, 5), textcoords='offset points')

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Taux de faux positifs (FPR)')
        plt.ylabel('Taux de vrais positifs (TPR)')
        plt.title('TPR vs FPR par modèle')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('unsw_nb15_tpr_fpr_comparison.png', dpi=300)
        plt.show()

        # Exporter les données TPR/FPR en CSV
        tpr_fpr_df.to_csv('/content/output/training/unsw_nb15_tpr_fpr_comparison.csv', index=False)

    print("\nAnalyse terminée avec succès!")
    print(f"Récapitulatif des fichiers générés:")
    print(f"  - unsw_nb15_model_metrics.csv: Métriques détaillées pour chaque fichier et modèle")
    print(f"  - unsw_nb15_global_metrics.csv: Métriques globales moyennes par modèle")
    print(f"  - unsw_nb15_global_metrics.png: Visualisation des métriques globales")
    print(f"  - unsw_nb15_training_times_comparison.png: Comparaison des temps d'entraînement")
    print(f"  - unsw_nb15_performance_vs_time.png: Rapport performance vs temps")
    print(f"  - unsw_nb15_performance_heatmap.png: Heatmap des performances par fichier et modèle")
    print(f"  - unsw_nb15_tpr_fpr_comparison.png: Comparaison TPR vs FPR")
    print(f"  - unsw_nb15_roc_curves.png: Courbes ROC moyennes")
    print(f"  - /content/output/roc_data/: Fichiers CSV contenant les données ROC détaillées")
    print(f"  - unsw_nb15_all_training_times.csv: Temps d'entraînement pour tous les modèles")
    print(f"  - Fichiers .pkl: Modèles et préprocesseurs sauvegardés pour chaque fichier")


main()