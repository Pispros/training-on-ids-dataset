import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class PreprocessTabularData:
    """
    Classe optimisée pour prétraiter les données tabulaires du dataset UNSW-NB15.
    Inclut One-Hot Encoding, LabelEncoder, StandardScaler, PCA et SelectKBest
    avec des paramètres optimisés par défaut.
    """

    def __init__(self, max_samples=None):
        """
        Initialise la classe de prétraitement.

        Args:
            max_samples (int, optional): Nombre maximal d'échantillons à utiliser.
                                        Si None, utilise tous les échantillons disponibles.
        """
        self.max_samples = max_samples

        # Définir les noms des colonnes spécifiques à UNSW-NB15
        self.feature_names = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
            'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
            'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
            'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
            'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
            'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'attack_cat', 'label'
        ]

        # Identifier les features catégorielles
        self.categorical_features = ['proto', 'state', 'service', 'attack_cat']

        # Ajouter srcip et dstip comme catégoriels car ce sont des adresses IP
        self.categorical_features.extend(['srcip', 'dstip'])

        # Ports comme catégoriels car ils peuvent contenir des chaînes ou des entiers
        self.categorical_features.extend(['sport', 'dsport'])

        # Tous les autres sont numériques sauf la cible
        self.numeric_features = [col for col in self.feature_names
                               if col not in self.categorical_features and col != 'label']

        # Encoders et transformers
        self.label_encoder = LabelEncoder()
        self.categorical_encoders = {}  # Encoder séparé pour chaque feature catégorielle
        self.one_hot_encoders = {}
        self.scaler = StandardScaler()

        # Initialiser PCA et SelectKBest avec des paramètres optimisés par défaut
        # Pour UNSW-NB15, les recherches montrent que ~20 composantes PCA et ~25 features selectionnées donnent de bons résultats
        self.pca = PCA(n_components=20)
        self.feature_selector = SelectKBest(f_classif, k=25)

        # Flag pour savoir si PCA et SelectKBest sont utilisés
        self.use_pca = True
        self.use_feature_selection = True

    def load_data(self, data_path):
        """
        Charge les données UNSW-NB15.

        Args:
            data_path (str): Chemin vers le fichier ou dossier de données.

        Returns:
            pd.DataFrame: DataFrame contenant les données chargées.
        """
        if os.path.isdir(data_path):
            # Si data_path est un dossier, fusionner tous les fichiers CSV
            return self._merge_csv_files(data_path)
        else:
            # Si data_path est un fichier, charger directement
            df = pd.read_csv(data_path, header=None, names=self.feature_names, dtype='object', low_memory=False)

            # Limiter le nombre d'échantillons si spécifié
            if self.max_samples and len(df) > self.max_samples:
                df = df.sample(n=self.max_samples, random_state=42)

            return df

    def _merge_csv_files(self, directory):
        """
        Fusionne tous les fichiers CSV dans un répertoire en un seul DataFrame

        Args:
            directory (str): Chemin du répertoire contenant les fichiers CSV

        Returns:
            pd.DataFrame: DataFrame contenant les données fusionnées
        """
        print(f"Fusion des fichiers CSV dans {directory}...")

        all_data = []
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

        if not csv_files:
            raise ValueError(f"Aucun fichier CSV trouvé dans {directory}")

        for file in csv_files:
            file_path = os.path.join(directory, file)
            print(f"  Lecture de {file}...")
            # Lecture sans en-tête avec les noms de colonnes définis
            df = pd.read_csv(file_path, header=None, names=self.feature_names, dtype='object', low_memory=False)
            all_data.append(df)
            print(f"  {file}: {len(df)} lignes")

        merged_df = pd.concat(all_data, ignore_index=True)
        print(f"Fusion terminée. Total: {len(merged_df)} lignes")

        # Limiter le nombre d'échantillons si spécifié
        if self.max_samples and len(merged_df) > self.max_samples:
            merged_df = merged_df.sample(n=self.max_samples, random_state=42)
            print(f"Échantillonnage à {self.max_samples} lignes")

        return merged_df

    def preprocess(self, df, fit=True, pca_components=None, k_best=None):
        """
        Prétraite les données avec One-Hot Encoding, StandardScaler, et optionnellement PCA et SelectKBest.

        Args:
            df (pd.DataFrame): Le DataFrame à prétraiter.
            fit (bool): Si True, ajuste les transformateurs. Sinon, les applique seulement.
            pca_components (int, optional): Nombre de composantes à conserver pour PCA.
                                           Si None, utilise la valeur par défaut (20).
            k_best (int, optional): Nombre de meilleures features à sélectionner.
                                   Si None, utilise la valeur par défaut (25).

        Returns:
            tuple: (X_preprocessed, y) où X_preprocessed sont les features prétraitées
                  et y sont les étiquettes encodées.
        """
        # Mettre à jour les paramètres si spécifiés
        if pca_components is not None:
            self.pca = PCA(n_components=pca_components)
            self.use_pca = True

        if k_best is not None:
            self.feature_selector = SelectKBest(f_classif, k=k_best)
            self.use_feature_selection = True

        # Gestion des valeurs manquantes
        df = df.fillna(0)

        # Séparation des features et de la cible
        X = df.drop('label', axis=1)
        y = df['label']

        # Encoder les labels (0 pour normal, 1 pour attaque)
        if fit:
            y = self.label_encoder.fit_transform(y)
        else:
            y = self.label_encoder.transform(y)

        # Prétraitement des features catégorielles avec encodage
        X_categorical = self._preprocess_categorical_features(X, fit)

        # Prétraitement des features numériques avec StandardScaler
        X_numeric = self._scale_numeric_features(X, fit)

        # Concaténation des features prétraitées
        if X_categorical is not None and not X_categorical.empty:
            X_preprocessed = pd.concat([X_numeric, X_categorical], axis=1)
        else:
            X_preprocessed = X_numeric

        # Application de PCA si activé
        if self.use_pca:
            if fit:
                X_preprocessed = self.pca.fit_transform(X_preprocessed)
                print(f"PCA: {X_preprocessed.shape[1]} composantes conservées, expliquant "
                      f"{sum(self.pca.explained_variance_ratio_)*100:.2f}% de la variance")
            else:
                X_preprocessed = self.pca.transform(X_preprocessed)

        # Sélection des meilleures features si activée
        if self.use_feature_selection:
            if fit:
                X_preprocessed = self.feature_selector.fit_transform(X_preprocessed, y)

                # Afficher les scores des features sélectionnées si PCA n'est pas utilisé
                if not self.use_pca:
                    selected_features = np.where(self.feature_selector.get_support())[0]
                    feature_scores = self.feature_selector.scores_[selected_features]
                    feature_names = X_preprocessed.columns[selected_features] if hasattr(X_preprocessed, 'columns') else [f"feature_{i}" for i in selected_features]
                    print("Top 5 features sélectionnées:")
                    for i in range(min(5, len(feature_names))):
                        print(f"  {feature_names[i]}: {feature_scores[i]:.4f}")
            else:
                X_preprocessed = self.feature_selector.transform(X_preprocessed)

        return X_preprocessed, y

    def _preprocess_categorical_features(self, X, fit=True):
        """
        Prétraite les features catégorielles avec LabelEncoder puis OneHotEncoder.

        Args:
            X (pd.DataFrame): DataFrame contenant les features.
            fit (bool): Si True, ajuste les encodeurs. Sinon, les applique seulement.

        Returns:
            pd.DataFrame: DataFrame avec les features catégorielles encodées.
        """
        if not self.categorical_features:
            return None

        encoded_dfs = []

        for feature in self.categorical_features:
            if feature in X.columns:
                # Extraire la colonne
                feature_values = X[feature].astype(str)  # Convertir en string pour uniformiser

                if fit:
                    # Enregistrer les valeurs uniques pour cette feature
                    self.categorical_encoders[feature] = feature_values.unique()

                # Créer un DataFrame avec les valeurs encodées par OneHotEncoder
                feature_df = pd.DataFrame(feature_values, columns=[feature], index=X.index)

                if fit:
                    # Créer et ajuster un nouvel encodeur pour cette feature
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(feature_df)
                    self.one_hot_encoders[feature] = encoder
                else:
                    # Utiliser l'encodeur existant
                    if feature in self.one_hot_encoders:
                        encoded = self.one_hot_encoders[feature].transform(feature_df)
                    else:
                        continue

                # Créer un DataFrame à partir des résultats encodés
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{feature}_{i}" for i in range(encoded.shape[1])],
                    index=X.index
                )

                encoded_dfs.append(encoded_df)

        if encoded_dfs:
            return pd.concat(encoded_dfs, axis=1)
        else:
            return pd.DataFrame()

    def _scale_numeric_features(self, X, fit=True):
        """
        Applique StandardScaler aux features numériques.

        Args:
            X (pd.DataFrame): DataFrame contenant les features.
            fit (bool): Si True, ajuste le scaler. Sinon, l'applique seulement.

        Returns:
            pd.DataFrame: DataFrame avec les features numériques normalisées.
        """
        if not self.numeric_features:
            return pd.DataFrame()

        # Extraire seulement les features numériques existantes
        numeric_cols = [col for col in self.numeric_features if col in X.columns]
        if not numeric_cols:
            return pd.DataFrame()

        # Convertir en numérique (avec conversion des erreurs en NaN puis remplacement par 0)
        numeric_df = X[numeric_cols].copy()
        for col in numeric_cols:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce').fillna(0)

        if fit:
            scaled_values = self.scaler.fit_transform(numeric_df)
        else:
            scaled_values = self.scaler.transform(numeric_df)

        scaled_df = pd.DataFrame(
            scaled_values,
            columns=numeric_cols,
            index=X.index
        )

        return scaled_df

    def get_feature_importance(self, model=None):
        """
        Récupère l'importance des features si disponible.

        Args:
            model: Un modèle entraîné qui a un attribut feature_importances_.

        Returns:
            pd.DataFrame: DataFrame avec les noms des features et leur importance.
        """
        if model is None or not hasattr(model, 'feature_importances_'):
            return None

        # Obtenir les noms des features après sélection
        if self.use_feature_selection and self.feature_selector:
            selected_features = np.where(self.feature_selector.get_support())[0]
            feature_names = [f"feature_{i}" for i in selected_features]
        else:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

        # Créer un DataFrame avec l'importance des features
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        })

        return importance_df.sort_values(by='Importance', ascending=False)

    def set_pca(self, use_pca=True, n_components=20):
        """
        Active ou désactive l'utilisation de PCA et définit le nombre de composantes.

        Args:
            use_pca (bool): Si True, utilise PCA lors du prétraitement.
            n_components (int): Nombre de composantes à conserver.
        """
        self.use_pca = use_pca
        if use_pca:
            self.pca = PCA(n_components=n_components)

    def set_feature_selection(self, use_feature_selection=True, k_best=25):
        """
        Active ou désactive la sélection de features et définit le nombre de features à conserver.

        Args:
            use_feature_selection (bool): Si True, utilise SelectKBest lors du prétraitement.
            k_best (int): Nombre de meilleures features à sélectionner.
        """
        self.use_feature_selection = use_feature_selection
        if use_feature_selection:
            self.feature_selector = SelectKBest(f_classif, k=k_best)