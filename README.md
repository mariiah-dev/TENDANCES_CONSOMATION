# Analyse des Tendances de Consommation Clients

## Description
Ce projet permet d'analyser les tendances de consommation à partir de données clients en utilisant le clustering K-Means. Il génère automatiquement des insights et des visualisations pour comprendre les différents segments de clients.

## Fonctionnalités
- Détection automatique du nombre optimal de clusters (méthode du coude + silhouette)
- Segmentation des clients basée sur de multiples critères
- Génération d'insights détaillés pour chaque segment
- Visualisations avancées :
  - Distribution des clusters
  - Profils démographiques
  - Comportements d'achat
  - Tendances saisonnières
  - Préférences de paiement
  - Utilisation des promotions
  - Matrices de corrélation

## Structure du Projet
```
📦 TENDANCES_CONSOMATION
 ┣ 📜 customer_clustering_v2.py   # Script principal Python
 ┣ 📜 ExcelVBACode.txt           # Code VBA pour l'intégration Excel
 ┣ 📜 requirements.txt           # Dépendances Python
 ┣ 📜 README.md                  # Documentation
 ┗ 📜 shopping_trends.xlsx       # Données d'exemple (Excel)
```

## Prérequis
- Python 3.8+
- Modules Python (voir requirements.txt) :
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - kneed
  - openpyxl
  - Pillow

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
```

2. Activer l'environnement virtuel :
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Via Python
```bash
python customer_clustering_v2.py "chemin/vers/fichier.xlsx"
```

### Via Excel
1. Ouvrir le classeur Excel cible
2. Importer le code VBA depuis `ExcelVBACode.txt`
3. Utiliser le formulaire pour sélectionner le fichier et lancer l'analyse

## Format des Données
Le fichier d'entrée doit contenir les colonnes suivantes :
- Customer ID
- Age
- Gender
- Item Purchased
- Category
- Purchase Amount (USD)
- Location
- Size
- Color
- Season
- Review Rating
- Subscription Status
- Payment Method
- Shipping Type
- Discount Applied
- Promo Code Used
- Previous Purchases
- Frequency of Purchases

## Résultats
Le script génère un nouveau fichier Excel avec :

1. **Onglet "Données analysées"**
   - Données originales avec attribution des clusters

2. **Onglet "Insights"**
   - Tableau détaillé des caractéristiques de chaque cluster
   - Informations démographiques
   - Comportements d'achat
   - Préférences

3. **Onglet "Visualisations"**
   - Méthode du coude et silhouette
   - Distribution des clusters
   - Analyses par catégorie
   - Tendances saisonnières
   - Comportements clients
   - Paiements et promotions
   - Corrélations

## Caractéristiques Techniques

### Prétraitement
- Encodage des variables catégorielles (LabelEncoder)
- Standardisation des variables numériques (StandardScaler)
- Gestion automatique des valeurs booléennes

### Clustering
- Détermination automatique du nombre optimal de clusters
- Utilisation combinée de la méthode du coude et du score de silhouette
- Clustering K-Means avec initialisation contrôlée (random_state)

### Visualisations
- Style matplotlib personnalisé
- Palette de couleurs Seaborn "husl"
- Mise en page automatique (tight_layout)
- Redimensionnement adaptatif des images dans Excel

### Export Excel
- Formatage automatique des tableaux
- Ajustement automatique des largeurs de colonnes
- Mise en forme conditionnelle
- Intégration des images redimensionnées

## Notes
- Les graphiques sont sauvegardés temporairement puis nettoyés après l'export
- Les valeurs réelles (non standardisées) sont utilisées pour les insights
- L'interface Excel permet une utilisation sans connaissances en Python

