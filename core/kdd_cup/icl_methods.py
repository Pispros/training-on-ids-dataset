import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
import time
import joblib
from google.colab import drive
drive.mount('/content/drive')


# Importer les classifieurs TabPFN et TabICL
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier

# Définir les chemins
DATA_PATH = "/content/drive/MyDrive/Datasets/KDD"  # Chemin direct vers le fichier CSV
OUTPUT_DIR = "/content/output/tabs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurer l'affichage des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Définir les seeds pour les 10 exécutions
SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]

print("=" * 80)
print("PRÉTRAITEMENT DES DONNÉES AVEC PREPROCESTABUILARDATA")
print("=" * 80)

# Le fichier n'a pas d'en-tête, nous utiliserons la dernière colonne comme cible
target_column = 'target'  # Nous renommerons la dernière colonne en 'target'
print(f"Utilisation de la dernière colonne comme cible avec le nom '{target_column}'")

# Initialiser et exécuter le prétraitement avec la colonne cible détectée
preprocessor = PreprocessTabularData(
    data_path=DATA_PATH,
    target_column=target_column,  # Utiliser la colonne cible détectée
    max_rows=50000,  # Limiter le nombre de lignes pour l'exemple
    apply_pca=True,
    pca_components=50,  # Assurer compatibilité avec TabPFN (max 100 features)
    apply_feature_selection=False,  # Désactivé car on utilise déjà PCA
    random_state=42
)

# Exécuter le prétraitement complet
X, y, feature_names = preprocessor.preprocess()
print(f"Dimensions après prétraitement: {X.shape}")
print(f"Nombre de caractéristiques: {len(feature_names)}")

# Vérifier la distribution des classes
unique_classes, class_counts = np.unique(y, return_counts=True)
print(f"Nombre de classes: {len(unique_classes)}")
print(f"Distribution: min={class_counts.min()}, max={class_counts.max()}")

# Sauvegarder le préprocesseur pour une utilisation ultérieure
preprocessor.save_preprocessor(os.path.join(OUTPUT_DIR, "preprocessor.joblib"))

print("\n" + "=" * 80)
print("ÉVALUATION DES MODÈLES SUR 10 SEEDS")
print("=" * 80)

# Dictionnaire pour stocker les résultats de tous les runs
all_results = {
    "TabPFNClassifier": {
        "accuracy": [], "precision": [], "recall": [], "f1": [], "train_time": [],
        "fpr": [], "tpr": [], "roc_auc": [], "inference_time": []
    },
    "TabICLClassifier": {
        "accuracy": [], "precision": [], "recall": [], "f1": [], "train_time": [],
        "fpr": [], "tpr": [], "roc_auc": [], "inference_time": []
    }
}

# Fonction pour évaluer un modèle
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, seed):
    """Évalue un modèle et retourne ses performances"""

    # Entraînement avec mesure du temps
    print(f"\nEntraînement du modèle {model_name} avec seed {seed}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Temps d'entraînement: {train_time:.2f} secondes")

    # Prédiction avec mesure du temps d'inférence
    print("Prédiction sur l'ensemble de test...")
    inference_start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - inference_start_time
    print(f"Temps d'inférence: {inference_time:.2f} secondes")

    # Calculer les probabilités pour les courbes ROC si le modèle le supporte
    try:
        y_prob = model.predict_proba(X_test)
    except (AttributeError, NotImplementedError):
        # Si le modèle ne supporte pas predict_proba, on crée une matrice binaire
        n_classes = len(np.unique(y))
        y_prob = np.zeros((y_test.shape[0], n_classes))
        for i, pred in enumerate(y_pred):
            y_prob[i, pred] = 1

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)

    # Adapter le calcul des métriques selon le type de problème
    is_binary = len(np.unique(y)) == 2

    if is_binary:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Calcul des métriques ROC pour les problèmes binaires
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
        except (IndexError, ValueError):
            # Cas où y_prob n'a pas la bonne forme pour les problèmes binaires
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 0])
            roc_auc = auc(fpr, tpr)
    else:  # Multi-classe
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Pour les problèmes multi-classes, on calcule ROC one-vs-rest
        n_classes = len(np.unique(y))

        # Initialiser les listes pour stocker les résultats de chaque classe
        all_fpr = []
        all_tpr = []

        try:
            # Calculer ROC pour chaque classe
            for i in range(n_classes):
                fpr_class, tpr_class, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
                all_fpr.append(fpr_class)
                all_tpr.append(tpr_class)

            # Calculer la moyenne des AUC pour multiclasse
            roc_auc = roc_auc_score(pd.get_dummies(y_test), y_prob, average='weighted', multi_class='ovr')

            # Utiliser le premier ensemble de fpr/tpr pour l'export (ou faire une moyenne)
            fpr, tpr = all_fpr[0], all_tpr[0]
        except (IndexError, ValueError):
            # Fallback si le calcul ROC échoue
            fpr, tpr = np.array([0, 1]), np.array([0, 1])
            roc_auc = 0.5

    # Afficher les résultats
    print(f"\nRésultats pour {model_name} avec seed {seed}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Temps d'inférence: {inference_time:.4f} secondes")

    # Sauvegarder le modèle (uniquement pour le premier seed)
    if seed == SEEDS[0]:
        model_path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
        print(f"Modèle sauvegardé: {model_path}")

        # Exporter les données ROC pour ce seed
        roc_data = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'model': model_name
        })
        roc_csv_path = os.path.join(OUTPUT_DIR, f'{model_name.lower().replace(" ", "_")}_roc_data.csv')
        roc_data.to_csv(roc_csv_path, index=False)
        print(f"Données ROC sauvegardées dans: {roc_csv_path}")

    # Ajouter les résultats au dictionnaire global
    all_results[model_name]["accuracy"].append(accuracy)
    all_results[model_name]["precision"].append(precision)
    all_results[model_name]["recall"].append(recall)
    all_results[model_name]["f1"].append(f1)
    all_results[model_name]["train_time"].append(train_time)
    all_results[model_name]["inference_time"].append(inference_time)
    all_results[model_name]["fpr"].append(fpr)
    all_results[model_name]["tpr"].append(tpr)
    all_results[model_name]["roc_auc"].append(roc_auc)

    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'inference_time': inference_time,
        'roc_auc': roc_auc,
        'seed': seed
    }

# Boucler sur chaque seed
for seed in SEEDS:
    print(f"\n{'='*40}")
    print(f"EXÉCUTION AVEC SEED {seed}")
    print(f"{'='*40}")

    # Diviser les données en ensembles d'entraînement et de test avec le seed actuel
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y if len(unique_classes) < 50 else None
    )
    print(f"Ensemble d'entraînement: {X_train.shape}, Ensemble de test: {X_test.shape}")

    # Préparation pour TabPFN (limité à 100 features et 10 classes)
    if X_train.shape[1] > 100:
        X_train_pfn = X_train[:, :100]
        X_test_pfn = X_test[:, :100]
    else:
        X_train_pfn = X_train
        X_test_pfn = X_test

    # Vérifier le nombre de classes pour TabPFN
    n_classes = len(np.unique(y))
    if n_classes > 10:
        print(f"Adaptation des classes pour TabPFN (de {n_classes} à 10 max)...")

        # Garder les 9 classes les plus fréquentes et regrouper les autres
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        sorted_idx = np.argsort(class_counts)[::-1]
        top_classes = unique_classes[sorted_idx[:9]]

        # Créer des copies des labels
        y_train_pfn = y_train.copy()
        y_test_pfn = y_test.copy()

        # Remplacer les classes non-top
        for idx, label in enumerate(unique_classes):
            if label not in top_classes:
                y_train_pfn[y_train == label] = 999
                y_test_pfn[y_test == label] = 999

        # Ré-encoder les classes
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_pfn = le.fit_transform(y_train_pfn)
        y_test_pfn = le.transform(y_test_pfn)
    else:
        y_train_pfn = y_train
        y_test_pfn = y_test

    # Évaluer TabPFNClassifier
    try:
        print("\n" + "-" * 40)
        print(f"ÉVALUATION DE TABPFNCLASSIFIER (SEED {seed})")
        print("-" * 40)

        tab_pfn = TabPFNClassifier(
            device='cuda',
            ignore_pretraining_limits=True
        )

        evaluate_model(tab_pfn, X_train_pfn, X_test_pfn, y_train_pfn, y_test_pfn, "TabPFNClassifier", seed)
    except Exception as e:
        print(f"Erreur lors de l'évaluation de TabPFNClassifier avec seed {seed}: {str(e)}")

    # Évaluer TabICLClassifier
    try:
        print("\n" + "-" * 40)
        print(f"ÉVALUATION DE TABICLCLASSIFIER (SEED {seed})")
        print("-" * 40)

        tab_icl = TabICLClassifier()

        evaluate_model(tab_icl, X_train, X_test, y_train, y_test, "TabICLClassifier", seed)
    except Exception as e:
        print(f"Erreur lors de l'évaluation de TabICLClassifier avec seed {seed}: {str(e)}")

# Calculer les moyennes et écarts-types
summary_results = {}
for model_name, metrics in all_results.items():
    summary_results[model_name] = {}
    for metric_name, values in metrics.items():
        if metric_name not in ["fpr", "tpr"]:  # On ne calcule pas la moyenne pour fpr et tpr
            if values:  # Vérifier que la liste n'est pas vide
                summary_results[model_name][f"{metric_name}_mean"] = np.mean(values)
                summary_results[model_name][f"{metric_name}_std"] = np.std(values)
            else:
                summary_results[model_name][f"{metric_name}_mean"] = np.nan
                summary_results[model_name][f"{metric_name}_std"] = np.nan

# Créer un DataFrame pour le CSV de résumé des métriques
df_results = pd.DataFrame([
    {
        'model': model_name,
        'accuracy': summary_results[model_name]['accuracy_mean'],
        'accuracy_std': summary_results[model_name]['accuracy_std'],
        'precision': summary_results[model_name]['precision_mean'],
        'precision_std': summary_results[model_name]['precision_std'],
        'recall': summary_results[model_name]['recall_mean'],
        'recall_std': summary_results[model_name]['recall_std'],
        'f1': summary_results[model_name]['f1_mean'],
        'f1_std': summary_results[model_name]['f1_std'],
        'roc_auc': summary_results[model_name]['roc_auc_mean'],
        'roc_auc_std': summary_results[model_name]['roc_auc_std'],
        'train_time': summary_results[model_name]['train_time_mean'],
        'train_time_std': summary_results[model_name]['train_time_std'],
        'inference_time': summary_results[model_name]['inference_time_mean'],
        'inference_time_std': summary_results[model_name]['inference_time_std']
    }
    for model_name in all_results.keys()
])

# Sauvegarder les résultats en CSV
csv_path = os.path.join(OUTPUT_DIR, 'model_results_summary.csv')
df_results.to_csv(csv_path, index=False)
print(f"\nRésultats sauvegardés dans: {csv_path}")

# Créer un DataFrame pour le CSV comparatif des temps d'exécution
df_times = pd.DataFrame([
    {
        'model': model_name,
        'seed': seed,
        'train_time': all_results[model_name]['train_time'][i],
        'inference_time': all_results[model_name]['inference_time'][i],
        'total_time': all_results[model_name]['train_time'][i] + all_results[model_name]['inference_time'][i]
    }
    for i, seed in enumerate(SEEDS)
    for model_name in all_results.keys()
])

# Sauvegarder les temps d'exécution en CSV
times_csv_path = os.path.join(OUTPUT_DIR, 'execution_times.csv')
df_times.to_csv(times_csv_path, index=False)
print(f"Temps d'exécution sauvegardés dans: {times_csv_path}")

# Créer un DataFrame pour toutes les métriques par seed
df_all_metrics = pd.DataFrame([
    {
        'model': model_name,
        'seed': SEEDS[i],
        'accuracy': all_results[model_name]['accuracy'][i],
        'precision': all_results[model_name]['precision'][i],
        'recall': all_results[model_name]['recall'][i],
        'f1': all_results[model_name]['f1'][i],
        'roc_auc': all_results[model_name]['roc_auc'][i],
        'train_time': all_results[model_name]['train_time'][i],
        'inference_time': all_results[model_name]['inference_time'][i]
    }
    for i in range(len(SEEDS))
    for model_name in all_results.keys()
    if i < len(all_results[model_name]['accuracy'])  # Vérification pour éviter les erreurs d'index
])

# Sauvegarder toutes les métriques par seed
all_metrics_csv_path = os.path.join(OUTPUT_DIR, 'all_metrics_by_seed.csv')
df_all_metrics.to_csv(all_metrics_csv_path, index=False)
print(f"Toutes les métriques par seed sauvegardées dans: {all_metrics_csv_path}")

# Exporter les données ROC moyennes (utilise uniquement le premier seed pour simplifier)
roc_data_combined = pd.DataFrame()
for model_name in all_results.keys():
    if all_results[model_name]["fpr"] and all_results[model_name]["tpr"]:
        roc_data = pd.DataFrame({
            'fpr': all_results[model_name]["fpr"][0],  # Premier seed
            'tpr': all_results[model_name]["tpr"][0],  # Premier seed
            'model': model_name
        })
        roc_data_combined = pd.concat([roc_data_combined, roc_data], ignore_index=True)

roc_combined_path = os.path.join(OUTPUT_DIR, 'combined_roc_data.csv')
roc_data_combined.to_csv(roc_combined_path, index=False)
print(f"Données ROC combinées sauvegardées dans: {roc_combined_path}")

# Afficher le tableau récapitulatif
print("\n" + "=" * 80)
print("RÉCAPITULATIF DES PERFORMANCES AVEC MOYENNE ET ÉCART-TYPE")
print("=" * 80)

print("\nMoyenne et écart-type des performances sur 10 seeds:")
print(df_results.to_string(float_format=lambda x: f"{x:.4f}"))

# Visualisation des résultats avec barres d'erreur
print("\n" + "=" * 80)
print("VISUALISATION DES RÉSULTATS")
print("=" * 80)

# Graphique de comparaison des métriques avec barres d'erreur
plt.figure(figsize=(15, 8))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(all_results.keys()))
width = 0.15  # Ajusté pour accommoder 5 barres

# Créer des sous-graphiques pour chaque métrique avec barres d'erreur
for i, metric in enumerate(metrics):
    means = [summary_results[model][f"{metric}_mean"] for model in all_results.keys()]
    stds = [summary_results[model][f"{metric}_std"] for model in all_results.keys()]

    plt.bar(
        x + (i - len(metrics)/2 + 0.5) * width,
        means,
        width,
        label=metric.capitalize(),
        color=colors[i % len(colors)],
        yerr=stds,
        capsize=5
    )

plt.xlabel('Modèle')
plt.ylabel('Score')
plt.title('Comparaison des performances des modèles (moyenne et écart-type sur 10 seeds)')
plt.xticks(x, all_results.keys())
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_with_std.png'))
plt.show()

# Graphique des temps d'exécution avec barres d'erreur (séparé entre entraînement et inférence)
plt.figure(figsize=(12, 6))
x = np.arange(len(all_results.keys()))
width = 0.35

# Barres pour les temps d'entraînement
train_means = [summary_results[model]["train_time_mean"] for model in all_results.keys()]
train_stds = [summary_results[model]["train_time_std"] for model in all_results.keys()]
bar1 = plt.bar(
    x - width/2,
    train_means,
    width,
    label="Temps d'entraînement",
    color=colors[0],
    yerr=train_stds,
    capsize=5
)

# Barres pour les temps d'inférence
inference_means = [summary_results[model]["inference_time_mean"] for model in all_results.keys()]
inference_stds = [summary_results[model]["inference_time_std"] for model in all_results.keys()]
bar2 = plt.bar(
    x + width/2,
    inference_means,
    width,
    label="Temps d'inférence",
    color=colors[1],
    yerr=inference_stds,
    capsize=5
)

plt.xlabel('Modèle')
plt.ylabel('Temps (secondes)')
plt.title("Comparaison des temps d'exécution (moyenne et écart-type sur 10 seeds)")
plt.xticks(x, all_results.keys())
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'execution_times_comparison.png'))
plt.show()

# Tracer les courbes ROC pour le premier seed de chaque modèle
plt.figure(figsize=(10, 8))
for model_name in all_results.keys():
    if all_results[model_name]["fpr"] and all_results[model_name]["tpr"]:
        plt.plot(
            all_results[model_name]["fpr"][0],
            all_results[model_name]["tpr"][0],
            label=f'{model_name} (AUC = {all_results[model_name]["roc_auc"][0]:.4f})'
        )

plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC des différents modèles (premier seed)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'))
plt.show()

# Graphique des temps d'entraînement par modèle et par seed
plt.figure(figsize=(14, 7))
sns.boxplot(x='model', y='train_time', data=df_times)
plt.title("Distribution des temps d'entraînement par modèle sur 10 seeds")
plt.xlabel('Modèle')
plt.ylabel("Temps d'entraînement (secondes)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_times_boxplot.png'))
plt.show()

# Graphique des temps d'inférence par modèle et par seed
plt.figure(figsize=(14, 7))
sns.boxplot(x='model', y='inference_time', data=df_times)
plt.title("Distribution des temps d'inférence par modèle sur 10 seeds")
plt.xlabel('Modèle')
plt.ylabel("Temps d'inférence (secondes)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'inference_times_boxplot.png'))
plt.show()

print("\n" + "=" * 80)
print("FIN DE L'EXEMPLE AVEC MULTIPLE SEEDS")
print("=" * 80)