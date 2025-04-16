#!pip install cudf-cu11 cuml-cu11

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
import gc
from tqdm import tqdm

# Importations conditionnelles pour RAPIDS (GPU)
try:
    import cudf
    import cuml
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.preprocessing import RobustScaler as cuRobustScaler
    from cuml.decomposition import PCA as cuPCA
    from cuml.preprocessing import SimpleImputer as cuSimpleImputer
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

class PreprocessTabularData:
    """
    Classe pour le prétraitement des données tabulaires pour la détection d'intrusion,
    spécifiquement adaptée pour le dataset CICIDS2018.

    Cette classe est optimisée pour une utilisation minimale de la mémoire et
    limite strictement le nombre de lignes chargées.

    Supporte l'accélération GPU via RAPIDS (cuDF et cuML) lorsque disponible.
    """

    def __init__(self,
                 data_folder_path,
                 max_rows=10000,  # Strictement limité à ce nombre
                 target_column='Label',
                 scaler_type='robust',
                 reduce_dim=True,
                 n_components=0.95,
                 variance_threshold=0.01,
                 random_state=42,
                 use_gpu=True):
        """
        Initialisation de la classe de prétraitement.

        Args:
            data_folder_path (str): Chemin vers le dossier contenant les fichiers CSV
            max_rows (int): Nombre maximal de lignes à charger (strictement respecté)
            target_column (str): Nom de la colonne cible
            scaler_type (str): Type de normalisateur ('standard', 'robust')
            reduce_dim (bool): Appliquer ou non la réduction de dimensionnalité
            n_components (float or int): Nombre de composantes PCA à conserver
            variance_threshold (float): Seuil de variance pour la sélection de caractéristiques
            random_state (int): Graine aléatoire pour la reproductibilité
            use_gpu (bool): Utiliser le GPU si disponible
        """
        self.data_folder_path = data_folder_path
        self.max_rows = max_rows
        self.target_column = target_column
        self.scaler_type = scaler_type
        self.reduce_dim = reduce_dim
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.use_gpu = use_gpu

        # Vérifier si RAPIDS est disponible et si l'utilisateur veut utiliser le GPU
        self.gpu_available = RAPIDS_AVAILABLE and use_gpu
        if self.use_gpu and not RAPIDS_AVAILABLE:
            print("AVERTISSEMENT: RAPIDS (cuDF, cuML) n'est pas disponible. Utilisation du CPU.")
        elif self.use_gpu and RAPIDS_AVAILABLE:
            print("GPU activé: Utilisation de RAPIDS (cuDF, cuML) pour l'accélération GPU.")

        # Attributs qui seront définis ultérieurement
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.cat_columns = None
        self.num_columns = None
        self.preprocessor = None
        self.pca = None
        self.label_mapping = None
        self.on_gpu = False  # Indique si les données sont sur GPU

        print(f"Préprocesseur initialisé avec un maximum strict de {self.max_rows} lignes")

    def load_data(self):
        """
        Charge les données à partir des fichiers CSV dans le dossier spécifié.
        Limite strictement le nombre de lignes chargées selon max_rows.
        Utilise cuDF si GPU disponible.
        """
        all_files = [f for f in os.listdir(self.data_folder_path) if f.endswith('.csv')]

        if not all_files:
            raise ValueError(f"Aucun fichier CSV trouvé dans {self.data_folder_path}")

        print(f"Dossier contient {len(all_files)} fichiers CSV")

        # Sélectionner un seul fichier pour éviter de surcharger la mémoire
        selected_file = all_files[0]
        file_path = os.path.join(self.data_folder_path, selected_file)

        print(f"Chargement du fichier: {selected_file} (limité à {self.max_rows} lignes)")

        try:
            # Optimisation pour économiser la mémoire
            dtype_dict = {
                'Flow Bytes/s': 'float32',
                'Flow Packets/s': 'float32',
                # Plus de types peuvent être ajoutés ici si nécessaire
            }

            if self.gpu_available:
                try:
                    # Utiliser cuDF pour charger les données sur GPU
                    print("Utilisation de cuDF pour charger les données sur GPU...")
                    df = cudf.read_csv(
                        file_path,
                        nrows=self.max_rows,
                        dtype=dtype_dict
                    )
                    self.on_gpu = True
                    print("Données chargées sur GPU avec cuDF.")
                except Exception as e:
                    print(f"Erreur lors du chargement avec cuDF: {e}")
                    print("Revenir au chargement CPU avec pandas...")
                    self.on_gpu = False
                    df = pd.read_csv(
                        file_path,
                        nrows=self.max_rows,
                        low_memory=False,
                        na_values=['?', '', 'NA', 'null', 'NULL', 'nan', 'NaN'],
                        dtype=dtype_dict,
                        memory_map=True,
                        skipinitialspace=True
                    )
            else:
                # Utiliser pandas pour le CPU
                df = pd.read_csv(
                    file_path,
                    nrows=self.max_rows,
                    low_memory=False,
                    na_values=['?', '', 'NA', 'null', 'NULL', 'nan', 'NaN'],
                    dtype=dtype_dict,
                    memory_map=True,
                    skipinitialspace=True
                )

            # Vérifier si le fichier contient des données
            if len(df) == 0:
                raise ValueError("Le fichier est vide")

            # Recherche de la colonne cible
            if self.target_column not in df.columns:
                potential_targets = [col for col in df.columns if 'label' in str(col).lower() or 'attack' in str(col).lower() or 'class' in str(col).lower()]
                if potential_targets:
                    self.target_column = potential_targets[0]
                    print(f"Utilisation de '{self.target_column}' comme colonne cible")
                else:
                    raise ValueError(f"Aucune colonne cible trouvée dans le fichier")

            # Optimisation des types de données pour réduire la consommation mémoire
            # (cette étape est différente pour cuDF vs pandas)
            if not self.on_gpu:
                df = self._optimize_dtypes(df)

            # Assignation au DataFrame de la classe
            self.df = df
            print(f"Données chargées: {len(df)} lignes et {df.shape[1]} colonnes")

        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du fichier: {e}")

        # Force le garbage collector pour libérer la mémoire
        gc.collect()

        return self

    def _optimize_dtypes(self, df):
        """
        Optimise les types de données pour réduire l'utilisation mémoire.
        Note: Cette fonction est utilisée uniquement pour pandas (CPU).
        """
        print("Optimisation des types de données...")
        memory_before = df.memory_usage(deep=True).sum() / (1024 * 1024)  # en MB

        # Traiter les colonnes numériques
        for col in df.select_dtypes(include=['float64']).columns:
            # Convertir en float32 pour économiser la moitié de la mémoire
            df[col] = df[col].astype('float32')

        # Traiter les entiers
        for col in df.select_dtypes(include=['int64']).columns:
            if col == self.target_column:
                continue  # Laisser la colonne cible intacte

            # Trouver le type le plus petit qui convient
            col_min, col_max = df[col].min(), df[col].max()

            if col_min >= 0:  # Valeurs positives
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:  # Valeurs négatives
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')

        # Traiter les colonnes de type 'object' qui pourraient être catégorielles
        for col in df.select_dtypes(include=['object']).columns:
            if col == self.target_column:
                continue

            n_unique = df[col].nunique()
            if n_unique < 0.5 * len(df):  # Si moins de 50% des valeurs sont uniques
                df[col] = df[col].astype('category')

        memory_after = df.memory_usage(deep=True).sum() / (1024 * 1024)  # en MB
        print(f"Mémoire utilisée réduite de {memory_before:.2f}MB à {memory_after:.2f}MB")

        return df

    def clean_data(self):
        """
        Nettoie les données : suppression des doublons, gestion des valeurs
        manquantes, valeurs infinies, colonnes avec trop de valeurs manquantes.
        Adapté pour fonctionner avec cuDF ou pandas.
        """
        if self.df is None:
            raise ValueError("Les données n'ont pas été chargées. Appelez d'abord load_data().")

        print("Nettoyage des données...")

        # Sauvegarde du nombre de lignes initial
        initial_rows = len(self.df)

        # Suppression des doublons (compatible avec cuDF et pandas)
        self.df = self.df.drop_duplicates()
        print(f"Doublons supprimés: {initial_rows - len(self.df)} lignes")

        # Gestion des colonnes avec trop de valeurs manquantes (>50%)
        # Cette partie est différente entre cuDF et pandas
        if self.on_gpu:
            # Version cuDF
            missing_percentage = self.df.isna().mean()
            cols_to_drop = [col for col, perc in missing_percentage.items() if perc > 0.5]
        else:
            # Version pandas
            missing_percentage = self.df.isnull().mean()
            cols_to_drop = missing_percentage[missing_percentage > 0.5].index.tolist()

        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            print(f"Colonnes supprimées (>50% valeurs manquantes): {len(cols_to_drop)}")

        # Remplacement des valeurs infinies (différent pour cuDF)
        if self.on_gpu:
            # cuDF version - plus limité mais fonctionne
            for col in self.df.select_dtypes(include=['float32', 'float64']).columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        else:
            # pandas version
            self.df = self.df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)

        # Identifier les colonnes numériques et catégorielles
        if self.on_gpu:
            # Version cuDF
            self.cat_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
            self.num_columns = [col for col in self.df.columns if col not in self.cat_columns]
        else:
            # Version pandas
            self.cat_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            self.num_columns = self.df.select_dtypes(include=['number']).columns.tolist()

        # Assurer que la colonne cible est exclue des colonnes de caractéristiques
        if self.target_column in self.cat_columns:
            self.cat_columns.remove(self.target_column)
        if self.target_column in self.num_columns:
            self.num_columns.remove(self.target_column)

        print(f"Nettoyage terminé. {len(self.df)} lignes, {len(self.cat_columns)} colonnes catégorielles, {len(self.num_columns)} colonnes numériques")

        # Force le garbage collector pour libérer la mémoire
        gc.collect()

        return self

    def transform_target(self):
        """
        Transforme la colonne cible pour la détection d'intrusion.
        Pour CICIDS2018, convertit les types d'attaques en classification binaire ou multi-classes.
        Adapté pour fonctionner avec cuDF ou pandas.
        """
        if self.df is None:
            raise ValueError("Les données n'ont pas été nettoyées. Appelez d'abord clean_data().")

        print("Transformation de la variable cible...")

        # S'assurer que la colonne cible existe
        if self.target_column not in self.df.columns:
            raise ValueError(f"La colonne cible {self.target_column} n'existe pas dans le DataFrame.")

        # Pour CICIDS2018, la colonne cible peut contenir différents types d'attaques
        if self.on_gpu:
            # Version cuDF
            unique_values = self.df[self.target_column].unique().to_pandas()
        else:
            # Version pandas
            unique_values = self.df[self.target_column].unique()

        print(f"Nombre de valeurs uniques dans la colonne cible: {len(unique_values)}")

        # Limiter l'affichage si trop de valeurs uniques
        if len(unique_values) > 10:
            print(f"Premières valeurs: {unique_values[:10]}...")
        else:
            print(f"Valeurs uniques: {unique_values}")

        # Déterminer si la colonne cible est déjà encodée numériquement
        is_numeric = pd.api.types.is_numeric_dtype(self.df[self.target_column].dtype)

        if not is_numeric:
            # Créer une version binaire (normal vs attaque)
            normal_labels = [label for label in unique_values
                           if isinstance(label, str) and
                           ('normal' in label.lower() or 'benign' in label.lower())]

            if normal_labels:
                normal_label = normal_labels[0]
                if self.on_gpu:
                    # Version cuDF
                    self.df['attack_binary'] = self.df[self.target_column].eq(normal_label).astype('int8').map({1: 0, 0: 1})
                else:
                    # Version pandas
                    self.df['attack_binary'] = np.where(self.df[self.target_column] == normal_label, 0, 1)

                print(f"Colonne binaire 'attack_binary' créée (0: normal, 1: attaque)")

                # Statistiques de classes
                if self.on_gpu:
                    binary_counts = self.df['attack_binary'].value_counts().to_pandas().to_dict()
                else:
                    binary_counts = self.df['attack_binary'].value_counts().to_dict()
                print(f"Distribution des classes binaires: {binary_counts}")

            # Encoder la colonne cible pour la classification multi-classes
            if self.on_gpu:
                # Version cuDF - plus limité que pandas.factorize
                # Créer une table de mapping manuelle
                unique_vals = self.df[self.target_column].unique().to_pandas()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.df['attack_type'] = self.df[self.target_column].map(mapping).fillna(0).astype('int32')
                self.label_mapping = {v: k for k, v in mapping.items()}
            else:
                # Version pandas
                codes, uniques = pd.factorize(self.df[self.target_column])
                self.df['attack_type'] = codes
                self.label_mapping = dict(enumerate(uniques))

            print(f"Colonne 'attack_type' créée avec encodage numérique")

        # Force le garbage collector pour libérer la mémoire
        gc.collect()

        return self

    def prepare_features(self):
        """
        Prépare les caractéristiques pour le ML avec gestion optimisée de la mémoire.
        Adapté pour fonctionner avec cuDF ou pandas.
        """
        if self.df is None:
            raise ValueError("La cible n'a pas été transformée. Appelez d'abord transform_target().")

        print("Préparation des caractéristiques...")

        # Par défaut, utiliser la colonne target d'origine
        target_col = self.target_column

        # Si des colonnes encodées ont été créées, les utiliser de préférence
        if 'attack_binary' in self.df.columns:
            # Classification binaire (recommandée pour Logistic Regression)
            target_col = 'attack_binary'
        elif 'attack_type' in self.df.columns:
            # Classification multi-classes
            target_col = 'attack_type'

        # La stratégie d'encodage One-Hot dépend du backend (cuDF vs pandas)
        if self.cat_columns:
            print(f"Encodage de {len(self.cat_columns)} colonnes catégorielles...")

            if self.on_gpu:
                # Version cuDF - Utiliser cuDF get_dummies (similaire à pd.get_dummies)
                # Créer un DataFrame pour les caractéristiques numériques
                all_features = self.df[self.num_columns].copy()

                # Encoder chaque colonne catégorielle séparément pour économiser la mémoire
                for col in tqdm(self.cat_columns, desc="Encodage One-Hot"):
                    # Créer des variables indicatrices
                    dummies = cudf.get_dummies(self.df[col], prefix=col, prefix_sep='_')
                    # Concaténer avec les caractéristiques existantes
                    all_features = cudf.concat([all_features, dummies], axis=1)
                    # Libérer la mémoire
                    del dummies
                    gc.collect()
            else:
                # Version pandas - méthode originale
                all_features = self.df[self.num_columns].copy()

                # Encoder les colonnes catégorielles une par une
                for col in tqdm(self.cat_columns, desc="Encodage One-Hot"):
                    unique_values = self.df[col].unique()[1:]
                    for val in unique_values:
                        col_name = f"{col}_{val}".replace(" ", "_").replace("/", "_")
                        all_features[col_name] = (self.df[col] == val).astype('uint8')
                    gc.collect()

            # Extraction X et y
            self.X = all_features
            if self.on_gpu:
                # Pour cuDF, nous devons décider si nous voulons garder y sur GPU
                self.y = self.df[target_col].values
                self.feature_names = list(all_features.columns)
            else:
                # Version pandas
                self.y = self.df[target_col].values
                self.feature_names = all_features.columns.tolist()

        else:
            # Si pas de colonnes catégorielles, c'est plus simple
            self.X = self.df[self.num_columns].copy()
            self.y = self.df[target_col].values
            if self.on_gpu:
                self.feature_names = list(self.num_columns)
            else:
                self.feature_names = self.num_columns

        # Vérifier les données pour éviter les problèmes lors de l'entraînement
        if self.on_gpu:
            # Version cuDF
            if self.X.isna().any().any():
                print("ATTENTION: Il reste des valeurs NaN dans X. Imputation...")
                for col in self.X.columns:
                    if self.X[col].isna().any():
                        self.X[col] = self.X[col].fillna(self.X[col].mean())
        else:
            # Version pandas
            if self.X.isna().any().any():
                print("ATTENTION: Il reste des valeurs NaN dans X. Imputation avec la médiane...")
                for col in self.X.columns:
                    if self.X[col].isna().any():
                        self.X[col] = self.X[col].fillna(self.X[col].median())

        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
        print(f"Target utilisée: {target_col}")

        # Libérer le DataFrame original pour économiser la mémoire
        del self.df
        self.df = None  # Marquer explicitement comme supprimé
        gc.collect()

        return self

    def create_preprocessor(self):
        """
        Crée un pipeline de prétraitement pour les caractéristiques numériques.
        Utilise soit des composants cuML (GPU) soit scikit-learn (CPU).
        """
        if self.X is None:
            raise ValueError("Les caractéristiques n'ont pas été préparées. Appelez d'abord prepare_features().")

        print("Création du pipeline de prétraitement...")

        # Création du pipeline selon le backend (GPU ou CPU)
        if self.on_gpu and self.gpu_available:
            print("Utilisation de cuML pour le prétraitement sur GPU...")

            # Définir le scaler GPU en fonction du type choisi
            if self.scaler_type == 'standard':
                scaler = cuStandardScaler()
            elif self.scaler_type == 'robust':
                scaler = cuRobustScaler()
            else:
                print(f"Type de scaler inconnu: {self.scaler_type}, utilisation de RobustScaler")
                scaler = cuRobustScaler()

            # Imputation des valeurs manquantes (GPU)
            imputer = cuSimpleImputer(strategy='median')

            # Appliquer l'imputation et la normalisation
            X_imputed = imputer.fit_transform(self.X)
            X_scaled = scaler.fit_transform(X_imputed)

            # PCA si demandé (GPU)
            if self.reduce_dim:
                if isinstance(self.n_components, float) and self.n_components < 1.0:
                    n_components = self.n_components
                else:
                    n_components = min(int(self.n_components), min(self.X.shape[1] // 2, 50))

                self.pca = cuPCA(n_components=n_components)
                X_transformed = self.pca.fit_transform(X_scaled)

                cum_var = self.pca.explained_variance_ratio_.cumsum()
                n_components = len(self.pca.explained_variance_ratio_)
                print(f"PCA (GPU): {n_components} composantes conservées, expliquant {cum_var[-1]*100:.2f}% de la variance")

                # Mise à jour de X
                self.X = X_transformed
            else:
                # Sans PCA
                self.X = X_scaled

            # Pas de pipeline formel avec cuML, on garde les transformateurs séparés
            self.preprocessor = {'imputer': imputer, 'scaler': scaler}
            if self.reduce_dim:
                self.preprocessor['pca'] = self.pca

            print(f"Prétraitement GPU terminé. X shape: {self.X.shape}")

        else:
            # Version CPU classique (scikit-learn)
            if self.on_gpu:
                # Si nous étions sur GPU mais cuML n'est pas disponible, revenir à pandas
                print("Conversion des données de cuDF à pandas pour le prétraitement CPU...")
                self.X = self.X.to_pandas()
                self.on_gpu = False

            # Définir le scaler en fonction du type choisi
            if self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                print(f"Type de scaler inconnu: {self.scaler_type}, utilisation de RobustScaler")
                scaler = RobustScaler()

            # Créer une liste d'étapes pour le pipeline
            steps = [
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', scaler)
            ]

            # Ajouter la sélection de caractéristiques basée sur la variance si demandé
            if self.variance_threshold > 0:
                steps.append(('variance_threshold', VarianceThreshold(threshold=self.variance_threshold)))

            # Ajouter PCA si la réduction de dimensionnalité est demandée
            if self.reduce_dim:
                if isinstance(self.n_components, float) and self.n_components < 1.0:
                    n_components = self.n_components
                else:
                    n_components = min(int(self.n_components), min(self.X.shape[1] // 2, 50))

                self.pca = PCA(n_components=n_components, random_state=self.random_state)
                steps.append(('pca', self.pca))

            # Créer le pipeline
            self.preprocessor = Pipeline(steps)
            print(f"Pipeline CPU créé avec les étapes: {[step[0] for step in steps]}")

        return self

    def fit_transform(self):
        """
        Ajuste le préprocesseur sur les données et les transforme.
        Compatible avec GPU (cuML) ou CPU (scikit-learn).
        """
        if self.preprocessor is None:
            raise ValueError("Le préprocesseur n'a pas été créé. Appelez d'abord create_preprocessor().")

        print("Ajustement et transformation des données...")

        # Si nous avons déjà transformé les données dans create_preprocessor() (cas GPU)
        if self.on_gpu and isinstance(self.preprocessor, dict):
            # Les données sont déjà transformées
            X_transformed = self.X
        else:
            # Version CPU standard avec pipeline scikit-learn
            X_transformed = self.preprocessor.fit_transform(self.X)

            # Si PCA a été appliquée, afficher les informations de variance expliquée
            if 'pca' in self.preprocessor.named_steps:
                cum_var = np.cumsum(self.pca.explained_variance_ratio_)
                n_components = len(self.pca.explained_variance_ratio_)
                print(f"PCA: {n_components} composantes conservées, expliquant {cum_var[-1]*100:.2f}% de la variance")

        print(f"Forme des données transformées: {X_transformed.shape}")

        # Libérer de la mémoire
        if not self.on_gpu:  # Pour GPU, X est déjà transformé
            del self.X
            self.X = X_transformed
        gc.collect()

        return X_transformed, self.y

    def transform(self, X_new):
        """
        Transforme de nouvelles données avec le préprocesseur déjà ajusté.
        Compatible avec GPU ou CPU.

        Args:
            X_new: Nouvelles données à transformer (cuDF DataFrame ou pandas DataFrame)

        Returns:
            array: Données transformées
        """
        if self.preprocessor is None:
            raise ValueError("Le préprocesseur n'a pas été ajusté. Appelez d'abord fit_transform().")

        # Vérifier si nous sommes en mode GPU avec dictionnaire d'étapes
        if self.on_gpu and isinstance(self.preprocessor, dict):
            # Appliquer chaque étape manuellement
            X_temp = self.preprocessor['imputer'].transform(X_new)
            X_temp = self.preprocessor['scaler'].transform(X_temp)
            if 'pca' in self.preprocessor:
                X_temp = self.preprocessor['pca'].transform(X_temp)
            return X_temp
        else:
            # Pipeline scikit-learn standard
            return self.preprocessor.transform(X_new)

    def process(self):
        """
        Exécute l'ensemble du pipeline de prétraitement.
        Compatible avec GPU ou CPU.

        Returns:
            tuple: (X_transformed, y) - Données prêtes pour l'entraînement
        """
        try:
            return (self
                   .load_data()
                   .clean_data()
                   .transform_target()
                   .prepare_features()
                   .create_preprocessor()
                   .fit_transform())
        except Exception as e:
            print(f"Erreur durant le traitement: {e}")
            # Si nous sommes arrivés jusqu'à self.X et self.y, les retourner malgré l'erreur
            if self.X is not None and self.y is not None:
                print("Retour des données partiellement traitées")
                return self.X, self.y
            raise

    def get_feature_names(self):
        """
        Retourne les noms des caractéristiques après prétraitement.
        Si PCA a été appliquée, renvoie les noms des composantes PCA.
        Compatible avec GPU ou CPU.

        Returns:
            list: Noms des caractéristiques
        """
        # Cas GPU avec dictionnaire d'étapes
        if self.on_gpu and isinstance(self.preprocessor, dict):
            if 'pca' in self.preprocessor:
                n_components = self.preprocessor['pca'].n_components
                return [f'PC{i+1}' for i in range(n_components)]
            else:
                return self.feature_names
        # Cas CPU avec pipeline scikit-learn
        else:
            if hasattr(self, 'pca') and self.pca is not None:
                return [f'PC{i+1}' for i in range(self.pca.n_components_)]
            elif 'variance_threshold' in self.preprocessor.named_steps:
                # Obtenir les indices des caractéristiques conservées
                support = self.preprocessor.named_steps['variance_threshold'].get_support()
                return [self.feature_names[i] for i, keep in enumerate(support) if keep]
            else:
                return self.feature_names