import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from google.colab import drive

# Importation des classificateurs de Prior Labs uniquement
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier

drive.mount('/content/drive')

# Définir le chemin du dossier contenant les fichiers CSV
DATA_PATH = "/content/drive/MyDrive/Datasets/UNSW"  # À adapter selon votre environnement

# Définir les seeds pour la reproductibilité
SEEDS = [42, 123, 456, 789, 1010]

# Fonction pour évaluer un modèle avec plusieurs seeds
def evaluate_model_with_seeds(model_class, model_name, X_train, y_train, X_test, y_test, seeds, params=None):
    """
    Évalue un modèle avec plusieurs seeds et retourne les résultats moyens.

    Args:
        model_class: Classe du modèle (TabICLClassifier ou TabPFNClassifier)
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
    roc_data = {}  # Pour stocker les données ROC par seed

    for seed in seeds:
        print(f"\nÉvaluation de {model_name} avec seed={seed}")

        # Création du modèle avec les paramètres spécifiés et le seed courant
        if params:
            if model_name == 'TabPFN':
                model = model_class(**params, device='cuda' if torch.cuda.is_available() else 'cpu', random_state=seed)
            else:  # TabICL
                model = model_class(**params, random_state=seed)
        else:
            if model_name == 'TabPFN':
                model = model_class(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=seed)
            else:  # TabICL
                model = model_class(random_state=seed)

        # Mesurer le temps d'entraînement
        start_time = time.time()

        # Ces modèles prennent les données sous forme de numpy arrays
        model.fit(X_train.values if hasattr(X_train, 'values') else X_train,
                  y_train.values if hasattr(y_train, 'values') else y_train)

        train_time = time.time() - start_time

        # Prédire sur les données de test
        start_time = time.time()
        y_pred = model.predict(X_test.values if hasattr(X_test, 'values') else X_test)
        predict_time = time.time() - start_time

        # Calculer les métriques détaillées
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Extraire les métriques du rapport
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        # Calculer l'AUC et données ROC si possible
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test.values if hasattr(X_test, 'values') else X_test)

                # Stocker les données ROC pour chaque classe
                classes = np.unique(y_test)
                roc_seed_data = {}

                # Vérifier si binaire ou multiclasse
                if y_proba.shape[1] == 2:  # Binaire
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
                    roc_seed_data['binary'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist(),
                        'auc': auc
                    }
                else:  # Multiclasse
                    # Convertir les étiquettes en format one-hot
                    y_test_onehot = pd.get_dummies(y_test)
                    auc = roc_auc_score(y_test_onehot, y_proba, multi_class='ovr')

                    # Calculer ROC pour chaque classe
                    for i, class_name in enumerate(classes):
                        if len(classes) <= 10:  # Limiter aux 10 premières classes pour éviter trop de données
                            fpr, tpr, thresholds = roc_curve(
                                (y_test == class_name).astype(int),
                                y_proba[:, i]
                            )
                            roc_seed_data[str(class_name)] = {
                                'fpr': fpr.tolist(),
                                'tpr': tpr.tolist(),
                                'thresholds': thresholds.tolist(),
                                'auc': roc_auc_score((y_test == class_name).astype(int), y_proba[:, i])
                            }
            else:
                auc = 0.0
                roc_seed_data = {}
        except Exception as e:
            print(f"Erreur lors du calcul de l'AUC et des données ROC: {e}")
            auc = 0.0
            roc_seed_data = {}

        # Stocker les données ROC pour ce seed
        roc_data[seed] = roc_seed_data

        # Afficher les résultats pour ce seed
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Temps d'entraînement: {train_time:.2f} s")
        print(f"  Temps de prédiction: {predict_time:.2f} s")

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
    print(f"  Temps de prédiction: {avg_results['predict_time']:.2f} s (±{std_results['predict_time']:.2f})")

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
        'best_model': best_model,
        'roc_data': roc_data
    }

# Fonction pour exporter les données ROC en CSV
def export_roc_data(roc_data, file_name, model_name, output_dir="/content/output/tabs/roc_data"):
    """
    Exporte les données ROC pour un modèle dans un fichier CSV.

    Args:
        roc_data: Dictionnaire contenant les données ROC par seed
        file_name: Nom du fichier CSV original
        model_name: Nom du modèle
        output_dir: Répertoire de sortie
    """
    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Préparer le nom du fichier de sortie
    output_file = os.path.join(output_dir, f"{file_name.replace('.csv', '')}_{model_name}_roc_data.csv")

    # Pour chaque seed et chaque classe, sauvegarder les données ROC
    all_roc_data = []

    for seed, seed_data in roc_data.items():
        for class_name, class_data in seed_data.items():
            for i in range(len(class_data['fpr'])):
                all_roc_data.append({
                    'seed': seed,
                    'class': class_name,
                    'fpr': class_data['fpr'][i],
                    'tpr': class_data['tpr'][i],
                    'threshold': class_data['thresholds'][i] if i < len(class_data['thresholds']) else None,
                    'auc': class_data['auc']
                })

    # Créer un DataFrame et exporter en CSV
    if all_roc_data:
        roc_df = pd.DataFrame(all_roc_data)
        roc_df.to_csv(output_file, index=False)
        print(f"Données ROC exportées dans {output_file}")
    else:
        print(f"Pas de données ROC disponibles pour {model_name} sur {file_name}")

# Fonction pour créer un comparatif des temps d'entraînement
def plot_training_times(results, file_name, output_dir="/content/output/tabs"):
    """
    Crée et sauvegarde un graphique comparatif des temps d'entraînement et de prédiction.

    Args:
        results: Dictionnaire contenant les résultats pour chaque modèle
        file_name: Nom du fichier CSV original
        output_dir: Répertoire de sortie
    """
    # Extraire les données de temps pour chaque modèle
    models = []
    train_times = []
    train_stds = []
    predict_times = []
    predict_stds = []

    for model_name, model_result in results.items():
        models.append(model_name)
        train_times.append(model_result['avg_train_time'])
        train_stds.append(model_result['std_train_time'])
        predict_times.append(model_result['avg_predict_time'])
        predict_stds.append(model_result['std_predict_time'])

    # Créer le graphique
    plt.figure(figsize=(12, 6))

    # Créer un graphique à barres avec deux séries
    x = np.arange(len(models))
    width = 0.35

    # Barres pour les temps d'entraînement
    plt.bar(x - width/2, train_times, width, label='Temps d\'entraînement', color='royalblue',
            yerr=train_stds, capsize=5, alpha=0.8)

    # Barres pour les temps de prédiction
    plt.bar(x + width/2, predict_times, width, label='Temps de prédiction', color='orange',
            yerr=predict_stds, capsize=5, alpha=0.8)

    # Ajouter les étiquettes et la légende
    plt.xlabel('Modèles')
    plt.ylabel('Temps (secondes)')
    plt.title(f'Comparaison des temps d\'entraînement et de prédiction - {file_name}')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Ajouter les valeurs au-dessus des barres
    for i, v in enumerate(train_times):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontsize=8)

    for i, v in enumerate(predict_times):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontsize=8)

    # Sauvegarder le graphique
    output_file = os.path.join(output_dir, f"{file_name.replace('.csv', '')}_time_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    print(f"Graphique des temps sauvegardé dans {output_file}")
    return plt

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

    # Pour chaque fichier CSV
    for file in csv_files:
        file_path = os.path.join(directory, file)
        print(f"\n{'='*50}")
        print(f"Évaluation des modèles pour {file}")
        print(f"{'='*50}")

        # Initialiser le préprocesseur avec la classe existante
        preprocessor = PreprocessTabularData(max_samples=max_samples)

        # Charger les données en utilisant notre méthode améliorée
        print(f"Chargement des données de {file}...")
        try:
            # Utiliser la méthode load_data avec dtype='object' et low_memory=False
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

        # Définir les modèles Prior Labs à évaluer avec leurs paramètres
        models = {
            "TabICL": {
                'class': TabICLClassifier,
                'params': {}
            },
            "TabPFN": {
                'class': TabPFNClassifier,
                'params': {}
            }
        }

        # Évaluer chaque modèle
        file_results = {}
        for model_name, model_info in models.items():
            result = evaluate_model_with_seeds(
                model_info['class'],
                model_name,
                X_train, y_train,
                X_test, y_test,
                seeds,
                model_info['params']
            )
            file_results[model_name] = result

            # Exporter les données ROC pour ce modèle
            export_roc_data(result['roc_data'], file, model_name)

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
                'Avg Train Time (s)': f"{result['avg_train_time']:.2f} (±{result['std_train_time']:.2f})",
                'Avg Predict Time (s)': f"{result['avg_predict_time']:.2f} (±{result['std_predict_time']:.2f})"
            })

        comparison_df = pd.DataFrame(comparison)
        print("\nComparaison des modèles:")
        print(comparison_df)

        # Créer le comparatif des temps d'entraînement
        plot_training_times(file_results, file)

        # Visualiser les résultats de performance
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
        plt.savefig(f"/content/output/tabs/{file.replace('.csv', '')}_performance_by_seed.png", dpi=300)
        plt.show()

        # Trouver le meilleur modèle pour ce fichier (basé sur F1-score)
        best_model_name = max(file_results.items(), key=lambda x: x[1]['avg_f1'])[0]
        best_model = file_results[best_model_name]['best_model']

        print(f"\nLe meilleur modèle pour {file} est {best_model_name}")

        # Sauvegarder le meilleur modèle
        model_filename = f'/content/output/tabs/unsw_nb15_{file.replace(".csv", "")}_{best_model_name.replace(" ", "_").lower()}.pkl'
        preprocessor_filename = f'/content/output/tabs/unsw_nb15_{file.replace(".csv", "")}_preprocessor.pkl'

        print(f"Sauvegarde du meilleur modèle dans {model_filename}...")
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Sauvegarde du préprocesseur dans {preprocessor_filename}...")
        with open(preprocessor_filename, 'wb') as f:
            pickle.dump(preprocessor, f)

    return all_results

# Fonction pour exporter les métriques en CSV
def export_metrics_to_csv(results, output_file="/content/output/tabs/unsw_nb15_prior_models_metrics.csv"):
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
                'Train_Time_per_Sample': model_result['avg_train_time'] / 1000 if 1000 else None,  # Basé sur max_samples=1000
                'Predict_Time_per_Sample': model_result['avg_predict_time'] / 200 if 200 else None,  # Basé sur test_size=0.2
            })

    # Créer un DataFrame et exporter en CSV
    metrics_df = pd.DataFrame(csv_rows)
    metrics_df.to_csv(output_file, index=False)
    print(f"\nMétriques exportées dans {output_file}")

    # Calculer et exporter le ratio performance/temps
    metrics_df['Efficiency_Score'] = metrics_df['F1_Score'] / metrics_df['Train_Time']
    metrics_df['Efficiency_Score'] = metrics_df['Efficiency_Score'] / metrics_df['Efficiency_Score'].max()  # Normaliser

    efficiency_file = output_file.replace('.csv', '_efficiency.csv')
    metrics_df[['File', 'Model', 'F1_Score', 'Train_Time', 'Efficiency_Score']].to_csv(efficiency_file, index=False)
    print(f"Métriques d'efficacité exportées dans {efficiency_file}")

    return metrics_df

# Fonction pour créer une visualisation des courbes ROC moyennes
def plot_average_roc_curves(results, output_dir="/content/output/tabs"):
    """
    Crée et sauvegarde des graphiques des courbes ROC moyennes pour chaque modèle et fichier.

    Args:
        results: Dictionnaire contenant les résultats pour chaque fichier
        output_dir: Répertoire de sortie
    """
    # Pour chaque fichier individuel
    for file_name, file_results in results.items():
        plt.figure(figsize=(10, 8))

        for model_name, model_result in file_results['model_results'].items():
            roc_data = model_result['roc_data']

            # Si nous avons des données ROC binaires, les tracer
            has_binary_data = False
            mean_tpr = []
            mean_fpr = np.linspace(0, 1, 100)

            for seed, seed_data in roc_data.items():
                if 'binary' in seed_data:
                    has_binary_data = True
                    fpr = seed_data['binary']['fpr']
                    tpr = seed_data['binary']['tpr']

                    # Interpoler pour avoir des fpr uniformes
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    mean_tpr.append(interp_tpr)

            if has_binary_data and mean_tpr:
                mean_tpr = np.array(mean_tpr)
                mean_tpr_avg = mean_tpr.mean(axis=0)
                mean_tpr_std = mean_tpr.std(axis=0)

                # Tracer la courbe ROC moyenne
                plt.plot(mean_fpr, mean_tpr_avg,
                         label=f'{model_name} (AUC = {model_result["avg_auc"]:.4f})',
                         lw=2, alpha=0.8)

                # Tracer l'intervalle de confiance
                plt.fill_between(mean_fpr,
                                mean_tpr_avg - mean_tpr_std,
                                mean_tpr_avg + mean_tpr_std,
                                alpha=0.2)

        # Finaliser le graphique
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title(f'Courbes ROC moyennes - {file_name}')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.3)

        # Sauvegarder le graphique
        output_file = os.path.join(output_dir, f"{file_name.replace('.csv', '')}_average_roc_curves.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)

        print(f"Courbes ROC moyennes sauvegardées dans {output_file}")

    # Création d'une courbe ROC globale pour tous les fichiers combinés
    plt.figure(figsize=(12, 10))

    # Organiser les données par modèle à travers tous les fichiers
    global_roc_data = {}

    for model_name in ['TabICL', 'TabPFN']:
        global_roc_data[model_name] = {
            'mean_tpr_lists': [],
            'auc_values': []
        }

    # Collecter les courbes TPR pour chaque modèle à travers tous les fichiers
    for file_name, file_results in results.items():
        for model_name, model_result in file_results['model_results'].items():
            roc_data = model_result['roc_data']

            # Si nous avons des données ROC binaires pour ce fichier, les collecter
            file_mean_tpr = []
            mean_fpr = np.linspace(0, 1, 100)

            for seed, seed_data in roc_data.items():
                if 'binary' in seed_data:
                    fpr = seed_data['binary']['fpr']
                    tpr = seed_data['binary']['tpr']

                    # Interpoler pour avoir des fpr uniformes
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    file_mean_tpr.append(interp_tpr)

            if file_mean_tpr:
                file_mean_tpr = np.array(file_mean_tpr)
                file_mean_tpr_avg = file_mean_tpr.mean(axis=0)

                # Ajouter cette TPR moyenne au modèle correspondant
                global_roc_data[model_name]['mean_tpr_lists'].append(file_mean_tpr_avg)
                global_roc_data[model_name]['auc_values'].append(model_result['avg_auc'])

    # Tracer les courbes ROC globales pour chaque modèle
    mean_fpr = np.linspace(0, 1, 100)

    for model_name, model_data in global_roc_data.items():
        if model_data['mean_tpr_lists']:
            # Calculer la TPR moyenne globale
            all_tpr = np.array(model_data['mean_tpr_lists'])
            global_mean_tpr = all_tpr.mean(axis=0)
            global_std_tpr = all_tpr.std(axis=0)

            # Calculer l'AUC moyen global
            global_auc = np.mean(model_data['auc_values'])
            global_auc_std = np.std(model_data['auc_values'])

            # Tracer la courbe ROC globale
            plt.plot(mean_fpr, global_mean_tpr,
                     label=f'{model_name} (AUC = {global_auc:.4f} ± {global_auc_std:.4f})',
                     lw=2.5, alpha=0.8)

            # Tracer l'intervalle de confiance
            plt.fill_between(mean_fpr,
                            global_mean_tpr - global_std_tpr,
                            global_mean_tpr + global_std_tpr,
                            alpha=0.2)

    # Finaliser le graphique global
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de vrais positifs (TPR)', fontsize=12)
    plt.title('Courbes ROC Globales - Tous Fichiers Combinés', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Sauvegarder le graphique global
    global_output_file = os.path.join(output_dir, "unsw_nb15_global_roc_curves.png")
    plt.tight_layout()
    plt.savefig(global_output_file, dpi=300)

    print(f"Courbe ROC globale sauvegardée dans {global_output_file}")

    # Exporter les données ROC globales en CSV
    global_roc_csv = []

    for model_name, model_data in global_roc_data.items():
        if model_data['mean_tpr_lists']:
            global_mean_tpr = np.array(model_data['mean_tpr_lists']).mean(axis=0)
            global_std_tpr = np.array(model_data['mean_tpr_lists']).std(axis=0)

            for i in range(len(mean_fpr)):
                global_roc_csv.append({
                    'model': model_name,
                    'fpr': mean_fpr[i],
                    'tpr': global_mean_tpr[i],
                    'tpr_std': global_std_tpr[i]
                })

    global_roc_df = pd.DataFrame(global_roc_csv)
    global_roc_csv_file = os.path.join(output_dir, "unsw_nb15_global_roc_data.csv")
    global_roc_df.to_csv(global_roc_csv_file, index=False)

    print(f"Données ROC globales exportées dans {global_roc_csv_file}")


# Programme principal
def main():
    print(f"Évaluation des modèles Prior Labs sur les données UNSW-NB15 avec {len(SEEDS)} seeds...")
    print(f"Chemin des données: {DATA_PATH}")
    print(f"Seeds utilisés: {SEEDS}")

    # Créer les répertoires de sortie
    os.makedirs("/content/output/tabs/roc_data", exist_ok=True)

    # Exécuter l'évaluation pour tous les fichiers CSV
    results = evaluate_models_for_csv_files(DATA_PATH, SEEDS, max_samples=1000)

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
        'Train_Time': ['mean', 'std'],
        'Predict_Time': ['mean', 'std']
    }).reset_index()

    # Reformater les colonnes pour plus de clarté
    model_summary.columns = [
        'Model',
        'Global_Accuracy', 'Global_Accuracy_Std',
        'Global_Precision', 'Global_Precision_Std',
        'Global_Recall', 'Global_Recall_Std',
        'Global_F1', 'Global_F1_Std',
        'Global_AUC', 'Global_AUC_Std',
        'Global_Train_Time', 'Global_Train_Time_Std',
        'Global_Predict_Time', 'Global_Predict_Time_Std'
    ]

    # Sauvegarder le récapitulatif global
    model_summary.to_csv("/content/output/tabs/unsw_nb15_prior_global_metrics.csv", index=False)
    print("\nRécapitulatif global exporté dans unsw_nb15_prior_global_metrics.csv")
    print("\nAperçu du récapitulatif global:")
    print(model_summary)

    # Visualiser les métriques globales avec barres d'erreur
    plt.figure(figsize=(15, 10))

    # Accuracy
    plt.subplot(2, 3, 1)
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
    plt.subplot(2, 3, 2)
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
    plt.subplot(2, 3, 3)
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
    plt.subplot(2, 3, 4)
    plt.errorbar(
        model_summary['Model'],
        model_summary['Global_F1'],
        yerr=model_summary['Global_F1_Std'],
        fmt='o', capsize=5, elinewidth=2, markeredgewidth=2
    )
    plt.title('F1-Score Global avec Écart-Type')
    plt.ylim(0.8, 1.0)  # Ajuster selon vos résultats
    plt.grid(True, linestyle='--', alpha=0.7)

    # AUC
    plt.subplot(2, 3, 5)
    plt.errorbar(
        model_summary['Model'],
        model_summary['Global_AUC'],
        yerr=model_summary['Global_AUC_Std'],
        fmt='o', capsize=5, elinewidth=2, markeredgewidth=2
    )
    plt.title('AUC Global avec Écart-Type')
    plt.ylim(0.8, 1.0)  # Ajuster selon vos résultats
    plt.grid(True, linestyle='--', alpha=0.7)

    # Score d'efficacité (F1/Temps)
    # Calculer le score d'efficacité normalisé
    efficiency_scores = []
    for i, model in enumerate(model_summary['Model']):
        efficiency = model_summary['Global_F1'][i] / model_summary['Global_Train_Time'][i]
        efficiency_scores.append(efficiency)

    # Normaliser les scores d'efficacité
    max_efficiency = max(efficiency_scores)
    efficiency_scores = [e/max_efficiency for e in efficiency_scores]

    plt.subplot(2, 3, 6)
    plt.bar(model_summary['Model'], efficiency_scores, alpha=0.7)
    plt.title('Score d\'Efficacité (F1/Temps)')
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Ajouter les valeurs au-dessus des barres
    for i, v in enumerate(efficiency_scores):
        plt.text(i, v + 0.05, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig('unsw_nb15_prior_global_metrics.png', dpi=300)
    plt.show()

    # Créer un graphique comparatif global des temps d'entraînement et de prédiction
    plt.figure(figsize=(14, 10))

    # Préparation des données
    models = model_summary['Model']
    x = np.arange(len(models))
    width = 0.35

    # Graphique des temps absolus
    plt.subplot(2, 1, 1)

    # Barres pour les temps d'entraînement
    plt.bar(x - width/2, model_summary['Global_Train_Time'], width,
            label='Temps d\'entraînement', color='royalblue',
            yerr=model_summary['Global_Train_Time_Std'], capsize=5, alpha=0.8)

    # Barres pour les temps de prédiction
    plt.bar(x + width/2, model_summary['Global_Predict_Time'], width,
            label='Temps de prédiction', color='orange',
            yerr=model_summary['Global_Predict_Time_Std'], capsize=5, alpha=0.8)

    # Ajouter les étiquettes et la légende
    plt.xlabel('Modèles')
    plt.ylabel('Temps (secondes)')
    plt.title('Comparaison Globale des Temps d\'Entraînement et de Prédiction')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Ajouter les valeurs au-dessus des barres
    for i, v in enumerate(model_summary['Global_Train_Time']):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)

    for i, v in enumerate(model_summary['Global_Predict_Time']):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)

    # Graphique des temps par échantillon
    plt.subplot(2, 1, 2)

    # Calculer les temps par échantillon (en millisecondes)
    train_time_per_sample = model_summary['Global_Train_Time'] * 1000 / 1000  # Temps par échantillon en ms
    predict_time_per_sample = model_summary['Global_Predict_Time'] * 1000 / 200  # Temps par échantillon en ms

    # Barres pour les temps d'entraînement par échantillon
    plt.bar(x - width/2, train_time_per_sample, width,
            label='Temps d\'entraînement par échantillon', color='royalblue',
            alpha=0.8)

    # Barres pour les temps de prédiction par échantillon
    plt.bar(x + width/2, predict_time_per_sample, width,
            label='Temps de prédiction par échantillon', color='orange',
            alpha=0.8)

    # Ajouter les étiquettes et la légende
    plt.xlabel('Modèles')
    plt.ylabel('Temps par échantillon (ms)')
    plt.title('Comparaison des Temps par Échantillon')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Ajouter les valeurs au-dessus des barres
    for i, v in enumerate(train_time_per_sample):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}ms', ha='center', va='bottom', fontsize=9)

    for i, v in enumerate(predict_time_per_sample):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}ms', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('unsw_nb15_prior_global_time_comparison.png', dpi=300)
    plt.show()

    # Tracer les courbes ROC moyennes
    plot_average_roc_curves(results)

    # Créer le radar chart pour comparer les modèles
    plot_radar_chart(model_summary)

    # Identifier le meilleur modèle global basé sur le F1-score
    best_model_name = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Model']
    best_f1 = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_F1']
    best_f1_std = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_F1_Std']
    best_accuracy = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_Accuracy']
    best_accuracy_std = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_Accuracy_Std']
    best_train_time = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_Train_Time']
    best_train_time_std = model_summary.loc[model_summary['Global_F1'].idxmax(), 'Global_Train_Time_Std']

    print(f"\nLe meilleur modèle global est {best_model_name} avec:")
    print(f"  - F1-score: {best_f1:.4f} (±{best_f1_std:.4f})")
    print(f"  - Accuracy: {best_accuracy:.4f} (±{best_accuracy_std:.4f})")
    print(f"  - Temps d'entraînement: {best_train_time:.2f}s (±{best_train_time_std:.2f})")

    # Créer une matrice de comparaison des métriques par modèle
    comparison_table = metrics_df.pivot_table(
        index='File',
        columns='Model',
        values=['Accuracy', 'F1_Score', 'Train_Time', 'Efficiency_Score'],
        aggfunc='mean'
    )

    # Créer une heatmap pour visualiser les performances par fichier et modèle
    plt.figure(figsize=(20, 12))

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
    plt.title('Temps d\'entraînement par fichier et modèle (s)')

    # Heatmap pour le score d'efficacité
    plt.subplot(2, 2, 4)
    sns.heatmap(comparison_table['Efficiency_Score'], annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
    plt.title('Score d\'efficacité (F1/Temps) par fichier et modèle')

    plt.tight_layout()
    plt.savefig('unsw_nb15_prior_performance_heatmap.png', dpi=300)
    plt.show()

    print("\nAnalyse terminée avec succès!")
    print(f"Récapitulatif des fichiers générés:")
    print(f"  - unsw_nb15_prior_models_metrics.csv: Métriques détaillées pour chaque fichier et modèle")
    print(f"  - unsw_nb15_prior_global_metrics.csv: Métriques globales moyennes par modèle")
    print(f"  - unsw_nb15_prior_global_metrics.png: Visualisation des métriques globales")
    print(f"  - unsw_nb15_prior_global_time_comparison.png: Visualisation des temps d'entraînement et de prédiction")
    print(f"  - unsw_nb15_prior_performance_heatmap.png: Heatmap des performances par fichier et modèle")
    print(f"  - unsw_nb15_radar_comparison.png: Graphique radar comparant les modèles sur plusieurs métriques")
    print(f"  - unsw_nb15_prior_models_metrics_efficiency.csv: Métriques d'efficacité (score F1/temps)")
    print(f"  - unsw_nb15_global_roc_curves.png: Courbe ROC globale combinant tous les fichiers")
    print(f"  - unsw_nb15_global_roc_data.csv: Données ROC globales pour analyse ultérieure")
    print(f"  - Dossier roc_data/: Fichiers CSV contenant les données ROC pour traçage")
    print(f"  - *_average_roc_curves.png: Courbes ROC moyennes par fichier")
    print(f"  - *_performance_by_seed.png: Graphiques de performance par seed")
    print(f"  - *_time_comparison.png: Comparaison des temps par fichier")
    print(f"  - Fichiers .pkl: Modèles et préprocesseurs sauvegardés pour chaque fichier")

# Exécuter le programme principal si ce script est exécuté directement
if __name__ == "__main__":
    main()