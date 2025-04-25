# Détection d'Intrusions Réseau par Apprentissage Automatique

Ce dépôt contient l'implémentation et l'évaluation de différentes méthodes d'apprentissage automatique pour la détection d'intrusions réseau sur trois datasets de référence : KDD CUP, UNSW-NB15 et CICIDS2018.

## 📋 Vue d'ensemble

L'objectif de ce projet est d'évaluer et de comparer l'efficacité de différentes approches de machine learning pour la détection d'intrusions réseau:
- Entraînement classique avec des algorithmes standards
- Optimisation des hyperparamètres avec Hyperopt
- Méthodes d'apprentissage en contexte (In-Context Learning) avec TabPFN et TabICL

Chaque méthode est testée sur trois datasets différents avec plusieurs seeds pour garantir la robustesse des résultats.

## 🔍 Datasets

### KDD CUP
- Benchmark classique pour la détection d'intrusion
- Évaluations avec 10 seeds différentes

### UNSW-NB15
- Dataset moderne contenant des trafics réseau normaux et d'attaque
- Évaluations sur 4 fichiers différents avec 5 seeds pour chacun

### CICIDS2018
- Dataset récent couvrant diverses catégories d'attaques
- Évaluations sur 10 fichiers différents avec 5 seeds pour chacun
- Tests supplémentaires de prévision de séries temporelles (régression)

## 🧪 Méthodologie

Pour chaque dataset, nous avons appliqué trois phases d'expérimentation:

1. **Entraînement classique**
   - Utilisation d'algorithmes standards de machine learning
   - Évaluation des performances de base

2. **Optimisation des hyperparamètres**
   - Utilisation de Hyperopt pour optimiser les modèles
   - Recherche automatisée des meilleurs hyperparamètres

3. **Méthodes ICL (In-Context Learning)**
   - Implémentation de TabPFN (Prior-Data Fitted Networks)
   - Implémentation de TabICL (Tabular In-Context Learning)
   - Évaluation des performances sans entraînement explicite

4. **Tâche de régression**
   - Prévision de séries temporelles sur CICIDS2018
   - Métriques de performance spécifiques à la régression

## 📊 Structure des expériences

- **KDD CUP**: 10 seeds × 3 phases (entraînement, hyperopt, TabPFN/TabICL)
- **UNSW-NB15**: 5 seeds × 4 fichiers × 3 phases
- **CICIDS2018**: 5 seeds × 10 fichiers × 3 phases + tâche de régression

## 📁 Structure du dépôt

```
├── core/
│   ├── kdd_cup/
│   │   ├── classic_training/
│   │   ├── hyperopt_optimization/
│   │   ├── icl_methods/
│   │   │   ├── tabpfn/
│   │   │   └── tabicl/
│   │   └── results/
│   ├── unsw_nb15/
│   │   ├── classic_training/
│   │   ├── hyperopt_optimization/
│   │   ├── icl_methods/
│   │   │   ├── tabpfn/
│   │   │   └── tabicl/
│   │   └── results/
│   ├── cicids2018/
│   │   ├── classic_training/
│   │   ├── hyperopt_optimization/
│   │   ├── icl_methods/
│   │   │   ├── tabpfn/
│   │   │   └── tabicl/
│   │   ├── time_series/
│   │   └── results/
│   └── notebooks/
│       ├── kdd_cup.ipynb
│       ├── unsw_nb15.ipynb
│       └── cicids2018.ipynb
└── preprocessor/
    ├── kdd_preprocessor.py
    ├── unsw_preprocessor.py
    ├── cicids_preprocessor.py
    ├── utils.py
    └── visualization.py
```

## 🚀 Installation et utilisation

```bash
# Cloner le dépôt
git clone https://github.com/Pispros/training-on-ids-dataset
cd detection-intrusion-ml

# Installer les dépendances
pip install -r requirements.txt

# Exécuter les notebooks
jupyter notebook notebooks/
```

## 📔 Notebooks

Les notebooks Colab contiennent l'implémentation complète et les résultats des expériences:

- [KDD CUP Notebook](https://colab.research.google.com/drive/1JvMTkVgEu2awt2IBXOWV9z33LMN15Z43?usp=sharing)
- [UNSW-NB15 Notebook](https://colab.research.google.com/drive/1-5vDR73L9d_qvdWXx4qCxcQudkIcWSWY?usp=sharing)
- [CICIDS2018 Notebook](https://colab.research.google.com/drive/1mz_lpUrGve69-umgS82PYYAJ0nY7gpWo?usp=sharing)

## 📈 Résultats principaux

### KDD CUP
- Résumé des performances sur 10 seeds
- Comparaison entre les approches classiques et ICL

### UNSW-NB15
- Résultats sur les 4 fichiers différents
- Impact de l'optimisation des hyperparamètres

### CICIDS2018
- Performances à travers les 10 fichiers
- Analyse des résultats de prévision de séries temporelles

### Lien Drive vers les résultats
<img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" width="16" height="16" /> [Lien Google Drive](https://drive.google.com/drive/folders/17WC3Tv17FralhxkWiMLdcqUzndMXmB4k?usp=sharing)

## 🔧 Technologies utilisées

- Python
- Scikit-learn
- Hyperopt
- TabPFN
- TabICL
- Pandas, NumPy
- Matplotlib, Seaborn
