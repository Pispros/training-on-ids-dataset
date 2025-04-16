import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# Définissez ici le chemin vers votre dossier de sortie parent
# qui contient les dossiers file_1, file_2, etc.
base_dir = "/content/output/training"  # Modifiez ce chemin si nécessaire

# Définir les noms des modèles
all_results = {
    'Logistic Regression': [],
    'Random Forest': [],
    'XGBoost': []
}

# Fonction pour trouver tous les fichiers ROC de manière récursive
def find_roc_files(base_path):
    roc_files = []
    # Recherche récursive de tous les fichiers CSV de données ROC
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('_roc_data.csv'):
                full_path = os.path.join(root, file)
                roc_files.append(full_path)
    return roc_files

# Trouver tous les fichiers de données ROC
all_roc_files = find_roc_files(base_dir)
print(f"Trouvé {len(all_roc_files)} fichiers de données ROC au total")

# Organiser les fichiers par modèle
model_roc_files = {
    'Logistic Regression': [],
    'Random Forest': [],
    'XGBoost': []
}

for file_path in all_roc_files:
    file_name = os.path.basename(file_path)
    if file_name.startswith('Logistic_Regression'):
        model_roc_files['Logistic Regression'].append(file_path)
    elif file_name.startswith('Random_Forest'):
        model_roc_files['Random Forest'].append(file_path)
    elif file_name.startswith('XGBoost'):
        model_roc_files['XGBoost'].append(file_path)

# Création des courbes ROC par modèle
if all_roc_files:
    # Pour chaque modèle, créer une courbe ROC distincte
    for model_name, roc_files in model_roc_files.items():
        if not roc_files:
            print(f"Aucun fichier ROC trouvé pour {model_name}")
            continue

        print(f"Création des courbes ROC pour {model_name} ({len(roc_files)} fichiers)")

        # Limiter le nombre de fichiers pour la lisibilité (max 5 fichiers)
        sample_files = roc_files[:5] if len(roc_files) > 5 else roc_files

        plt.figure(figsize=(12, 9))

        # Tracer une ligne de référence (aléatoire)
        plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')

        # Pour chaque fichier de ce modèle
        for roc_file in sample_files:
            try:
                # Extraire les informations du chemin
                parts = roc_file.split(os.sep)
                file_info = None
                seed_info = None

                # Recherche du pattern file_X et seed_Y dans le chemin
                for part in parts:
                    if part.startswith('file_'):
                        file_info = part
                    elif part.startswith('seed_'):
                        seed_info = part

                file_seed_info = f"{file_info}_{seed_info}" if file_info and seed_info else os.path.basename(roc_file)

                print(f"  Traitement du fichier {file_seed_info}")
                roc_data = pd.read_csv(roc_file)

                # Vérifier si nous avons des données de classe
                if 'class' not in roc_data.columns:
                    print(f"  ERREUR: Colonne 'class' manquante dans {roc_file}")
                    continue

                # Obtenir les classes uniques
                classes = roc_data['class'].unique()

                for class_name in classes:
                    class_data = roc_data[roc_data['class'] == class_name]

                    # Vérifier si les colonnes requises sont présentes
                    required_cols = ['fpr', 'tpr', 'auc']
                    if not all(col in class_data.columns for col in required_cols):
                        missing = [col for col in required_cols if col not in class_data.columns]
                        print(f"  ERREUR: Colonnes manquantes {missing} pour {class_name}")
                        continue

                    # S'assurer que les données sont triées
                    class_data = class_data.sort_values('fpr')

                    # Prendre l'AUC de la première ligne
                    auc_value = class_data['auc'].iloc[0] if not class_data.empty else 0

                    # Tracer la courbe ROC
                    label = f"{class_name} - {file_seed_info} (AUC = {auc_value:.3f})"
                    plt.plot(class_data['fpr'], class_data['tpr'], label=label)

            except Exception as e:
                print(f"  ERREUR lors du traitement de {roc_file}: {e}")

        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title(f'Courbes ROC pour {model_name}')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Gérer la légende pour qu'elle reste lisible
        if len(sample_files) > 5:
            plt.legend(loc='lower right', fontsize='x-small', bbox_to_anchor=(1.1, 0))
        else:
            plt.legend(loc='lower right', fontsize='small')

        # Sauvegarder la figure
        save_path = os.path.join(base_dir, f"{model_name.replace(' ', '_')}_roc_curves.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  Courbes ROC pour {model_name} sauvegardées dans {save_path}")

    # Créer une courbe ROC moyenne par modèle pour comparaison
    plt.figure(figsize=(12, 9))
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')

    colors = plt.cm.tab10.colors  # Palette de couleurs

    # Pour chaque modèle
    for i, (model_name, roc_files) in enumerate(model_roc_files.items()):
        if not roc_files:
            continue

        print(f"Calcul de la ROC moyenne pour {model_name}")

        # Collecter toutes les données ROC pour ce modèle
        all_fpr_tpr_auc = []

        for roc_file in roc_files:
            try:
                roc_data = pd.read_csv(roc_file)

                # Vérifier les colonnes nécessaires
                if not all(col in roc_data.columns for col in ['class', 'fpr', 'tpr', 'auc']):
                    continue

                # Sélectionner une classe appropriée
                # (soit 'binary', 'positive', ou la première classe disponible)
                if 'binary' in roc_data['class'].values:
                    class_data = roc_data[roc_data['class'] == 'binary']
                elif 'positive' in roc_data['class'].values:
                    class_data = roc_data[roc_data['class'] == 'positive']
                else:
                    # Si pas de classe spécifique, prendre la première
                    first_class = roc_data['class'].unique()[0]
                    class_data = roc_data[roc_data['class'] == first_class]

                # S'assurer que les données sont triées
                class_data = class_data.sort_values('fpr')

                # S'assurer qu'il y a au moins deux points
                if len(class_data) >= 2:
                    fpr = class_data['fpr'].values
                    tpr = class_data['tpr'].values
                    auc_val = class_data['auc'].iloc[0]

                    # Vérifier les valeurs
                    if np.isfinite(fpr).all() and np.isfinite(tpr).all() and np.isfinite(auc_val):
                        all_fpr_tpr_auc.append((fpr, tpr, auc_val))

            except Exception as e:
                print(f"  Erreur avec {roc_file}: {e}")

        if all_fpr_tpr_auc:
            # Calculer la moyenne des AUC
            auc_values = [auc for _, _, auc in all_fpr_tpr_auc]
            mean_auc = np.mean(auc_values)

            # Pour la courbe moyenne, utiliser une grille FPR commune
            mean_fpr = np.linspace(0, 1, 100)

            # Interpoler tous les TPR pour cette grille FPR
            interp_tpr = []
            for fpr, tpr, _ in all_fpr_tpr_auc:
                if len(fpr) >= 2:  # Besoin d'au moins 2 points pour l'interpolation
                    interp_tpr.append(np.interp(mean_fpr, fpr, tpr))

            if interp_tpr:
                # Calculer le TPR moyen
                mean_tpr = np.mean(interp_tpr, axis=0)

                # S'assurer que la courbe commence à (0,0) et finit à (1,1)
                mean_tpr[0] = 0.0
                mean_tpr[-1] = 1.0

                # Tracer la courbe ROC moyenne
                plt.plot(mean_fpr, mean_tpr, color=colors[i % len(colors)],
                         label=f'{model_name} (AUC = {mean_auc:.3f})',
                         linewidth=2)

                # Tracer l'écart-type (zone ombrée)
                std_tpr = np.std(interp_tpr, axis=0)
                tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
                tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tpr_lower, tpr_upper,
                                 color=colors[i % len(colors)], alpha=0.2,
                                 label=f'±1 écart-type')
            else:
                print(f"  Pas assez de données valides pour interpoler {model_name}")
        else:
            print(f"  Pas de données ROC valides pour {model_name}")

    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Comparaison des courbes ROC moyennes par modèle')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(base_dir, "models_comparison_roc_curves.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Comparaison des courbes ROC sauvegardée dans {save_path}")

    print("Courbes ROC créées avec succès.")
else:
    print("Aucun fichier de données ROC trouvé. Impossible de créer les courbes ROC.")

# Rechercher et analyser les temps d'entraînement
training_times_file = glob.glob(os.path.join(base_dir, '**/training_times_comparison.csv'), recursive=True)

if training_times_file:
    try:
        training_times_path = training_times_file[0]
        training_times_df = pd.read_csv(training_times_path)

        print("\nCréation de la visualisation des temps d'entraînement...")

        # Préparer les données pour la visualisation
        plot_data = []

        for _, row in training_times_df.iterrows():
            # Extraire le nom du fichier (raccourci si nécessaire)
            file_name = row['file']
            if len(file_name) > 20:  # Tronquer les noms trop longs
                file_name = file_name[:17] + '...'

            seed = row['seed']

            for model in ['Logistic Regression', 'Random Forest', 'XGBoost']:
                if model in row:
                    plot_data.append({
                        'file': file_name,
                        'seed': seed,
                        'model': model,
                        'train_time': row[model]
                    })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)

            # 1. Graphique de comparaison des temps d'entraînement par modèle (boxplot)
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='model', y='train_time', data=plot_df, palette='viridis')
            plt.title('Distribution des temps d\'entraînement par modèle')
            plt.ylabel('Temps d\'entraînement (secondes)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            save_path = os.path.join(base_dir, "training_times_boxplot.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Boxplot des temps d'entraînement sauvegardé dans {save_path}")

            # 2. Graphique détaillé des temps d'entraînement par fichier et modèle
            plt.figure(figsize=(15, 8))

            # Nuage de points avec jitter pour voir tous les points
            ax = sns.stripplot(x='file', y='train_time', hue='model', data=plot_df,
                              dodge=True, jitter=True, alpha=0.6, size=7)

            # Ajouter les moyennes
            sns.pointplot(x='file', y='train_time', hue='model', data=plot_df,
                         dodge=0.5, join=False, ci=None, markers='d', scale=1.2,
                         palette='dark')

            plt.title('Temps d\'entraînement par fichier et par modèle')
            plt.ylabel('Temps (secondes)')
            plt.xlabel('Fichier de données')
            plt.xticks(rotation=45)

            # Gérer la légende (éviter la duplication)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[:3], labels[:3], title='Modèle')

            plt.tight_layout()

            save_path = os.path.join(base_dir, "detailed_training_times.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Visualisation détaillée des temps d'entraînement créée dans {save_path}")

            # 3. Heatmap des temps d'entraînement
            pivot_df = plot_df.pivot_table(
                values='train_time',
                index='file',
                columns='model',
                aggfunc='mean'
            )

            plt.figure(figsize=(12, len(pivot_df) * 0.7 + 2))
            sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
            plt.title('Heatmap des temps d\'entraînement moyens par fichier et modèle')
            plt.tight_layout()

            save_path = os.path.join(base_dir, "training_times_heatmap.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Heatmap des temps d'entraînement sauvegardée dans {save_path}")

        else:
            print("Pas suffisamment de données de temps d'entraînement pour créer des visualisations")
    except Exception as e:
        print(f"\nErreur lors de la création des graphiques de temps d'entraînement: {e}")
else:
    print("\nAucun fichier de temps d'entraînement trouvé.")

print("\nTraitement terminé.")