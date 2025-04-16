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

from google.colab import drive
drive.mount('/content/drive')

# La classe PreprocessTabularData est supposée être disponible dans votre environnement

# Définition des chemins exactement comme dans l'exemple original
input_file = "/content/drive/MyDrive/Datasets/KDD/file.csv"
temp_dir = "/content/temp"
output_dir = "/content/output/training"

# Création des répertoires s'ils n'existent pas
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "seeds"), exist_ok=True)

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

# Liste des seeds à utiliser
SEEDS = [42, 123, 256, 789, 1024, 2048, 4096, 8192, 16384, 32768]

# Structure pour stocker les résultats
results_by_model = {
    'Logistic Regression': [],
    'Random Forest': [],
    'XGBoost': []
}

# Fonction pour évaluer un modèle (reprise exacte de l'exemple)
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

    # Affichage des résultats
    print(f"\n----- {model_name} (Seed {seed}) -----")
    print(f"Temps d'entraînement: {train_time:.2f} secondes")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

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
        'seed': seed
    }

# Boucle principale pour chaque seed
for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\n{'='*80}")
    print(f"EXPÉRIENCE AVEC SEED {seed} ({seed_idx}/{len(SEEDS)})")
    print(f"{'='*80}")

    try:
        # Initialisation du préprocesseur avec le seed actuel (exactement comme dans l'exemple)
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

        # Ajustement des hyperparamètres pour les grands jeux de données
        n_features = X.shape[1]
        n_samples = X.shape[0]

        if n_features > 1000 or n_samples > 100000:
            print("\nGrand jeu de données détecté, ajustement des hyperparamètres...")
            n_estimators = 50
            max_depth = 10
            max_iter = 500
        else:
            n_estimators = 100
            max_depth = None
            max_iter = 1000

        print(f"Paramètres: n_estimators={n_estimators}, max_depth={max_depth}, max_iter={max_iter}")

        # Configuration des modèles (exactement comme dans l'exemple)
        models = [
            {
                'name': 'Logistic Regression',
                'model': LogisticRegression(
                    C=1.0,
                    max_iter=max_iter,
                    class_weight='balanced',
                    random_state=seed,
                    n_jobs=-1,
                    solver='saga' if n_features > 1000 else 'lbfgs',
                    multi_class='auto'
                )
            },
            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight='balanced',
                    random_state=seed,
                    n_jobs=-1
                )
            },
            {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=0.1,
                    max_depth=10 if max_depth is None else max_depth,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method='hist' if n_samples > 10000 else 'auto',
                    objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
                    random_state=seed,
                    n_jobs=-1
                )
            }
        ]

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

for model_name, results in results_by_model.items():
    if not results:
        print(f"Aucun résultat pour {model_name}")
        continue

    # Calculer les moyennes et écarts types
    accuracies = [r['accuracy'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]

    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)

    std_accuracy = np.std(accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1_scores)

    print(f"\n----- {model_name} (sur {len(results)} seeds) -----")
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")

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
        'Num_Seeds': len(results)
    })

# Exporter les statistiques en CSV
stats_df = pd.DataFrame(stats_data)
stats_csv_path = os.path.join(output_dir, 'model_performance_stats.csv')
stats_df.to_csv(stats_csv_path, index=False)
print(f"\nStatistiques exportées vers {stats_csv_path}")

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

# Visualisation des résultats
print("\nCréation des visualisations...")

# Comparaison des métriques moyennes avec écart type
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)

    means = [d[f'{metric}_Mean'] for d in stats_data]
    stds = [d[f'{metric}_Std'] for d in stats_data]
    models = [d['Model'] for d in stats_data]

    bars = plt.bar(models, means, yerr=stds, capsize=5, color=colors[i % len(colors)])

    # Ajouter les valeurs sur les barres
    for j, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + stds[j] + 0.01,
                f'{means[j]:.4f}', ha='center', va='bottom', fontsize=9)

    plt.title(f'{metric} moyenne avec écart type')
    plt.ylabel(metric)
    plt.ylim(0, 1.1)  # Limiter l'axe y entre 0 et 1.1
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_metrics_comparison.png'))

# Visualisation de la stabilité (écart type des métriques)
plt.figure(figsize=(12, 6))
bar_width = 0.2
x = np.arange(len(stats_data))

for i, metric in enumerate(metrics):
    plt.bar(x + i * bar_width,
            [d[f'{metric}_Std'] for d in stats_data],
            bar_width,
            label=metric,
            color=colors[i % len(colors)])

plt.xlabel('Modèle')
plt.ylabel('Écart type')
plt.title('Stabilité des modèles à travers différents seeds')
plt.xticks(x + bar_width * (len(metrics) - 1) / 2, [d['Model'] for d in stats_data])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_stability.png'))

print(f"Visualisations sauvegardées dans {output_dir}")
print("\nExpérience terminée !")