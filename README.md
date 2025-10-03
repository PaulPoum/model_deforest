README.md# README.md

## 🌳 Modélisation et Analyse de la Déforestation - Thèse Doctorale

### Description
Cette application Streamlit est un outil interactif conçu pour l'analyse et la modélisation de la déforestation dans le cadre d'une thèse doctorale. Elle intègre des données historiques (2000-2024) sur les forêts denses, les plantations, les cultures agricoles, la population et la séquestration de CO2. Les fonctionnalités principales incluent :

- **Visualisation des données** : Graphiques interactifs (Plotly) et tableaux descriptifs.
- **Analyse des tendances** : Corrélations, taux de changement et tendances linéaires.
- **Modélisation prédictive** : Régressions linéaires pour prédire l'évolution des forêts denses et la séquestration CO2.
- **Scénarios futurs** : Simulations basées sur les scénarios GIEC SSP (Shared Socioeconomic Pathways) pour projeter les impacts jusqu'en 2100.
- **Rapport scientifique** : Synthèse exécutive, recommandations et indicateurs de suivi.

L'application utilise des données fictives ou simulées pour démontrer les dynamiques socio-environnementales liées à la croissance démographique et à l'expansion agricole.

### Prérequis
- Python 3.8+ installé.
- Accès à un environnement virtuel recommandé (ex. : `venv`).

### Installation
1. Clonez ou téléchargez le projet.
2. Créez un environnement virtuel :
   ```
   python -m venv env
   source env/bin/activate  # Sur Linux/Mac
   # ou
   env\Scripts\activate  # Sur Windows
   ```
3. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```
4. Lancez l'application :
   ```
   streamlit run app.py
   ```
   (Remplacez `app.py` par le nom de votre fichier principal contenant la fonction `main()`.)

L'application s'ouvrira dans votre navigateur à l'adresse `http://localhost:8501`.

### Utilisation
- **Navigation** : Utilisez la barre latérale pour sélectionner une section (Données, Analyse, Modélisation, Scénarios, Rapport).
- **Interactions** : Sliders, selectbox et radio buttons pour configurer les scénarios et visualisations.
- **Exports** : Boutons de téléchargement pour CSV (données, projections) et un placeholder pour PDF (rapport).

### Structure du Code
- **load_data()** : Charge les données historiques sous forme de DataFrame Pandas.
- **train_models()** : Entraîne des modèles de régression linéaire (sklearn) pour les forêts denses et la séquestration CO2.
- **main()** : Fonction principale gérant l'interface Streamlit avec des sections conditionnelles.

### Limites et Améliorations
- Données simulées : Remplacez par des données réelles (ex. : via API ou CSV externe).
- Modèles : Les régressions linéaires sont basiques ; envisagez des modèles plus avancés (ex. : Random Forest via sklearn).
- PDF Export : Implémentez une bibliothèque comme `reportlab` ou `weasyprint` pour générer le rapport PDF.
- Déploiement : Hébergez sur Streamlit Cloud ou Heroku pour un accès public.

### Auteur
- Projet développé pour une thèse doctorale en sciences environnementales.
- Contact : [Votre nom ou email].

### Licence
MIT License - libre pour usage académique et recherche.

---

# requirements.txt
```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
```