import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

class PreprocessTabularData:
    """
    Classe pour prétraiter les données tabulaires de détection d'intrusion (KDD Cup).

    Cette classe effectue les opérations suivantes:
    - Chargement des données CSV sans en-têtes
    - Gestion des types mixtes de données
    - Normalisation des variables numériques
    - Encodage des variables catégorielles
    - Réduction de dimensionnalité (optionnelle)
    - Préparation des données pour l'entraînement
    """

    def __init__(self, data_path, target_column='label', max_rows=10000,
                 apply_pca=True, pca_components=0.95, apply_feature_selection=True,
                 k_best_features=20, random_state=42):
        """
        Initialise la classe de prétraitement.

        Args:
            data_path (str): Chemin vers le dossier contenant les fichiers CSV
            target_column (str): Nom de la colonne cible
            max_rows (int): Nombre maximal de lignes à charger (default: 10000)
            apply_pca (bool): Appliquer PCA pour réduire la dimensionnalité
            pca_components (float/int): Nombre de composantes PCA ou ratio de variance expliquée
            apply_feature_selection (bool): Appliquer la sélection de caractéristiques
            k_best_features (int): Nombre de meilleures caractéristiques à conserver
            random_state (int): Seed pour la reproductibilité
        """
        self.data_path = data_path
        self.target_column = target_column
        self.max_rows = max_rows
        self.apply_pca = apply_pca
        self.pca_components = pca_components
        self.apply_feature_selection = apply_feature_selection
        self.k_best_features = k_best_features
        self.random_state = random_state

        # Pipelines et transformateurs
        self.preprocessor = None
        self.label_encoder = None
        self.pca_transformer = None
        self.feature_selector = None

        # Données
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.class_names = None

    def load_data(self, file_pattern='.csv'):
        """
        Charge les données à partir des fichiers CSV dans le dossier spécifié.

        Args:
            file_pattern (str): Motif pour filtrer les fichiers (default: '.csv')

        Returns:
            pandas.DataFrame: DataFrame contenant les données chargées
        """
        all_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)
                     if os.path.isfile(os.path.join(self.data_path, f)) and file_pattern in f]

        if not all_files:
            raise ValueError(f"Aucun fichier CSV trouvé dans {self.data_path}")

        dataframes = []
        total_rows = 0

        for file in all_files:
            if total_rows >= self.max_rows:
                break

            # Calcul du nombre de lignes à lire depuis ce fichier
            remaining_rows = self.max_rows - total_rows

            try:
                # Chargement sans en-têtes et avec gestion des types mixtes
                df = pd.read_csv(file, header=None, nrows=remaining_rows, low_memory=False)

                # Assignation de noms de colonnes génériques
                column_names = [f'col_{i}' for i in range(df.shape[1])]
                df.columns = column_names

                # La dernière colonne est généralement la cible dans KDD
                if self.target_column not in column_names:
                    df = df.rename(columns={column_names[-1]: self.target_column})
                    print(f"Colonne '{column_names[-1]}' renommée en '{self.target_column}'")

                dataframes.append(df)
                total_rows += len(df)
                print(f"Fichier {file} chargé: {len(df)} lignes, {len(df.columns)} colonnes")

            except Exception as e:
                warnings.warn(f"Erreur lors du chargement du fichier {file}: {str(e)}")

        if not dataframes:
            raise ValueError("Aucune donnée n'a pu être chargée")

        # Concatène tous les dataframes
        df = pd.concat(dataframes, ignore_index=True)

        print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df

    def clean_data_types(self, df):
        """
        Nettoie et corrige les types de données pour éviter les problèmes avec les types mixtes.

        Args:
            df (pandas.DataFrame): DataFrame à nettoyer

        Returns:
            pandas.DataFrame: DataFrame avec des types homogènes
        """
        # Pour chaque colonne, assurer l'homogénéité des types
        for col in df.columns:
            if col == self.target_column:
                # Pour la cible, convertir en chaîne de caractères pour assurer l'homogénéité
                df[col] = df[col].astype(str)
            else:
                # Pour les colonnes features, tenter une conversion numérique
                try:
                    # Vérifie si la colonne peut être numérique
                    pd.to_numeric(df[col], errors='raise')
                    # Convertir en float
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
                except (ValueError, TypeError):
                    # Si pas numérique, traiter comme catégorielle
                    df[col] = df[col].astype(str)

        return df

    def identify_features(self, df):
        """
        Identifie automatiquement les colonnes numériques et catégorielles.

        Args:
            df (pandas.DataFrame): DataFrame à analyser

        Returns:
            tuple: (numeric_features, categorical_features)
        """
        categorical_features = []
        numeric_features = []

        for col in df.columns:
            if col == self.target_column:
                continue

            # Si la colonne est numérique
            if pd.api.types.is_numeric_dtype(df[col]):
                # Vérifier si c'est une colonne avec peu de valeurs uniques (probablement catégorielle)
                if df[col].nunique() < 10:
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)

        return numeric_features, categorical_features

    def preprocess(self):
        """
        Effectue le prétraitement complet des données.

        Returns:
            tuple: (X_processed, y, feature_names)
        """
        try:
            # Chargement des données
            df = self.load_data()

            # Nettoyage des types de données
            df = self.clean_data_types(df)

            # Séparation des caractéristiques et de la cible
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            # Identification des types de caractéristiques
            numeric_features, categorical_features = self.identify_features(df)
            print(f"Caractéristiques numériques: {len(numeric_features)}")
            print(f"Caractéristiques catégorielles: {len(categorical_features)}")

            # Encodage des étiquettes cibles
            if y.dtype == 'object' or y.dtype == 'string':
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
                self.class_names = self.label_encoder.classes_
                print(f"Classes détectées: {len(self.class_names)}")

            # Configuration des transformateurs
            transformers = []

            # Transformateur pour les caractéristiques numériques
            if numeric_features:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', numeric_transformer, numeric_features))

            # Transformateur pour les caractéristiques catégorielles
            if categorical_features:
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                transformers.append(('cat', categorical_transformer, categorical_features))

            # Si aucun transformateur n'est défini, on retourne les données telles quelles
            if not transformers:
                print("Aucune caractéristique à transformer")
                self.X_train = X.values
                self.y_train = y
                self.feature_names = X.columns.tolist()
                return self.X_train, self.y_train, self.feature_names

            # Création du préprocesseur
            self.preprocessor = ColumnTransformer(transformers=transformers)

            # Application du prétraitement
            X_processed = self.preprocessor.fit_transform(X)
            print(f"Dimensions après prétraitement: {X_processed.shape}")

            # Récupération des noms de caractéristiques
            feature_names = self._get_feature_names(X.columns, numeric_features, categorical_features)

            # Réduction de dimensionnalité (PCA)
            if self.apply_pca and X_processed.shape[1] > 2:
                print("Application de PCA...")

                # Pour les très grandes dimensions, limiter le nombre de composantes
                if isinstance(self.pca_components, float) and X_processed.shape[1] > 1000:
                    print("Très grande dimensionnalité détectée, limitation à 500 composantes maximum...")
                    max_components = min(500, X_processed.shape[1] - 1)
                    self.pca_transformer = PCA(n_components=max_components, random_state=self.random_state)
                else:
                    self.pca_transformer = PCA(n_components=self.pca_components, random_state=self.random_state)

                # Application de PCA
                X_processed = self.pca_transformer.fit_transform(X_processed)

                # Mise à jour des noms de caractéristiques pour PCA
                feature_names = [f"PC{i+1}" for i in range(X_processed.shape[1])]

                print(f"Dimensions après PCA: {X_processed.shape[1]}")

            # Sélection de caractéristiques
            if self.apply_feature_selection and not self.apply_pca and X_processed.shape[1] > self.k_best_features:
                print("Application de la sélection de caractéristiques...")
                # Utilise au maximum le nombre de caractéristiques disponibles ou k_best_features
                k = min(self.k_best_features, X_processed.shape[1])

                self.feature_selector = SelectKBest(f_classif, k=k)
                X_processed = self.feature_selector.fit_transform(X_processed, y)

                # Mise à jour des noms de caractéristiques
                selected_indices = self.feature_selector.get_support(indices=True)
                feature_names = [feature_names[i] for i in selected_indices]

                print(f"Caractéristiques sélectionnées: {len(feature_names)}")

            # Stockage des données prétraitées
            self.X_train = X_processed
            self.y_train = y
            self.feature_names = feature_names

            print(f"Prétraitement terminé. Dimensions finales: {X_processed.shape}")

            return X_processed, y, feature_names

        except Exception as e:
            print(f"Erreur lors du prétraitement: {str(e)}")
            raise

    def _get_feature_names(self, original_columns, numeric_features, categorical_features):
        """
        Récupère les noms des caractéristiques après transformation.

        Args:
            original_columns (list): Noms des colonnes d'origine
            numeric_features (list): Liste des caractéristiques numériques
            categorical_features (list): Liste des caractéristiques catégorielles

        Returns:
            list: Noms des caractéristiques après transformation
        """
        feature_names = []

        # Ajout des noms pour les caractéristiques numériques (inchangés)
        feature_names.extend(numeric_features)

        # Pour les caractéristiques catégorielles, on ajoute un nom générique
        # puisqu'on ne connaît pas à l'avance les catégories obtenues par OneHotEncoder
        for cat_feature in categorical_features:
            feature_names.append(f"{cat_feature}_encoded")

        return feature_names

    def get_preprocessed_data(self):
        """
        Renvoie les données prétraitées.

        Returns:
            tuple: (X_train, y_train, feature_names)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Les données n'ont pas encore été prétraitées. Appelez d'abord preprocess().")

        return self.X_train, self.y_train, self.feature_names

    def transform_new_data(self, X_new):
        """
        Applique les transformations apprises à de nouvelles données.

        Args:
            X_new (pandas.DataFrame): Nouvelles données à transformer

        Returns:
            numpy.ndarray: Données transformées
        """
        if self.preprocessor is None:
            raise ValueError("Le préprocesseur n'a pas été initialisé. Appelez d'abord preprocess().")

        # Application du prétraitement
        X_processed = self.preprocessor.transform(X_new)

        # Application de PCA si utilisé
        if self.pca_transformer is not None:
            X_processed = self.pca_transformer.transform(X_processed)

        # Application de la sélection de caractéristiques si utilisée
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)

        return X_processed

    def encode_target(self, y_new):
        """
        Encode de nouvelles étiquettes cibles.

        Args:
            y_new (array-like): Nouvelles étiquettes à encoder

        Returns:
            numpy.ndarray: Étiquettes encodées
        """
        if self.label_encoder is None:
            raise ValueError("L'encodeur d'étiquettes n'a pas été initialisé.")

        return self.label_encoder.transform(y_new)

    def decode_target(self, y_encoded):
        """
        Décode les étiquettes encodées vers leurs valeurs originales.

        Args:
            y_encoded (array-like): Étiquettes encodées à décoder

        Returns:
            numpy.ndarray: Étiquettes décodées
        """
        if self.label_encoder is None:
            raise ValueError("L'encodeur d'étiquettes n'a pas été initialisé.")

        return self.label_encoder.inverse_transform(y_encoded)

    def save_preprocessor(self, filepath):
        """
        Sauvegarde les composants de prétraitement pour une utilisation ultérieure.

        Args:
            filepath (str): Chemin où sauvegarder les composants
        """
        import joblib

        # Crée un dictionnaire avec les composants à sauvegarder
        components = {
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'pca_transformer': self.pca_transformer,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }

        # Sauvegarde les composants
        joblib.dump(components, filepath)
        print(f"Préprocesseur sauvegardé dans {filepath}")

    @classmethod
    def load_preprocessor(cls, filepath):
        """
        Charge les composants de prétraitement préalablement sauvegardés.

        Args:
            filepath (str): Chemin vers les composants sauvegardés

        Returns:
            PreprocessTabularData: Instance avec les composants chargés
        """
        import joblib

        # Charge les composants
        components = joblib.load(filepath)

        # Crée une nouvelle instance
        instance = cls(data_path=None)

        # Restaure les composants
        instance.preprocessor = components['preprocessor']
        instance.label_encoder = components['label_encoder']
        instance.pca_transformer = components['pca_transformer']
        instance.feature_selector = components['feature_selector']
        instance.feature_names = components['feature_names']
        instance.class_names = components['class_names']

        print(f"Préprocesseur chargé depuis {filepath}")
        return instance