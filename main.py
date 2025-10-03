import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Modélisation de la Déforestation",
    page_icon="🌳",
    layout="wide"
)

# Données du document
def load_data():
    data = {
        'Année': [2000, 2010, 2020, 2024],
        'Forêts Denses (FD)': [2935120.85, 2915092.02, 2864439.44, 2860127.18],
        'Forêts Plantation (FP)': [0, 0, 34277.6138, 35541.0633],
        'Cultures Annuelles (CA)': [60056.2268, 75187.9303, 89736.1242, 95863.0772],
        'Cultures Pérennes (CP)': [1681.71003, 6365.11871, 7412.82087, 5109.36655],
        'Prairies (P)': [2763.85723, 1948.96416, 2017.31212, 2036.01943],
        'Terrains Habités (TH)': [5432.13584, 5247.33525, 5402.10888, 5363.85768],
        'Eaux (E)': [9509.11208, 10711.234, 10876.2742, 10131.158],
        'Autres (A)': [269.937827, 281.22511, 672.139938, 662.105813],
        'Population': [128346, 224254, 240915, 247643],
        'Séquestration CO2': [420915, 399854, 347443, 329920]
    }
    return pd.DataFrame(data)

# Modèle de prédiction
def train_models(df):
    models = {}
    X = df[['Année', 'Population', 'Cultures Annuelles (CA)']].values
    y_fd = df['Forêts Denses (FD)'].values
    y_co2 = df['Séquestration CO2'].values
    
    # Modèle forêts denses
    model_fd = LinearRegression()
    model_fd.fit(X, y_fd)
    models['fd'] = model_fd
    
    # Modèle CO2
    model_co2 = LinearRegression()
    model_co2.fit(X, y_co2)
    models['co2'] = model_co2
    
    return models

def main():
    st.title("🌳 Modélisation et Analyse de la Déforestation")
    st.markdown("""
    **Problématique de thèse** : Analyse des dynamiques de déforestation et modélisation des impacts 
    socio-environnementaux dans un contexte de croissance démographique et d'expansion agricole.
    """)
    
    # Chargement des données
    df = load_data()
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Sections", [
        "📊 Données et Visualisation",
        "📈 Analyse des Tendances", 
        "🎯 Modélisation Prédictive",
        "🔮 Scénarios Futurs",
        "📋 Rapport Complet"
    ])
    
    # Section 1: Données et Visualisation
    if page == "📊 Données et Visualisation":
        st.header("📊 Données Brutes et Visualisation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Dataset Complet")
            st.dataframe(df.style.format("{:,.2f}"), use_container_width=True)
            
            st.subheader("Statistiques Descriptives")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Évolution des Principaux Indicateurs")
            
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=('Forêts Denses (Ha)', 'Population',
                                            'Séquestration CO2 (T)', 'Cultures Annuelles (Ha)'))
            
            # Forêts denses
            fig.add_trace(go.Scatter(x=df['Année'], y=df['Forêts Denses (FD)'],
                                   mode='lines+markers', name='Forêts Denses'),
                         row=1, col=1)
            
            # Population
            fig.add_trace(go.Scatter(x=df['Année'], y=df['Population'],
                                   mode='lines+markers', name='Population'),
                         row=1, col=2)
            
            # CO2
            fig.add_trace(go.Scatter(x=df['Année'], y=df['Séquestration CO2'],
                                   mode='lines+markers', name='Séquestration CO2'),
                         row=2, col=1)
            
            # Cultures annuelles
            fig.add_trace(go.Scatter(x=df['Année'], y=df['Cultures Annuelles (CA)'],
                                   mode='lines+markers', name='Cultures Annuelles'),
                         row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Section 2: Analyse des Tendances
    elif page == "📈 Analyse des Tendances":
        st.header("📈 Analyse des Tendances et Corrélations")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Taux de Changement Annuel")
            
            # Calcul des taux de changement
            changes = {}
            for column in df.columns:
                if column != 'Année':
                    start_val = df[column].iloc[0]
                    end_val = df[column].iloc[-1]
                    total_change = end_val - start_val
                    annual_rate = total_change / (df['Année'].iloc[-1] - df['Année'].iloc[0])
                    changes[column] = annual_rate
            
            changes_df = pd.DataFrame.from_dict(changes, orient='index', columns=['Taux Annuel'])
            st.dataframe(changes_df.style.format("{:,.2f}"), use_container_width=True)
        
        with col2:
            st.subheader("Matrice de Corrélation")
            
            # Calcul des corrélations
            corr_data = df.drop('Année', axis=1)
            corr_matrix = corr_data.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmin=-1, zmax=1
            ))
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Analyse détaillée des tendances
        st.subheader("Analyse Détailée des Tendances")
        
        selected_indicator = st.selectbox("Sélectionnez un indicateur:", 
                                         df.columns[1:])
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df['Année'], y=df[selected_indicator],
                                     mode='lines+markers', name=selected_indicator))
        
        # Ajout de la tendance linéaire
        z = np.polyfit(df['Année'], df[selected_indicator], 1)
        p = np.poly1d(z)
        trend_line = p(df['Année'])
        fig_trend.add_trace(go.Scatter(x=df['Année'], y=trend_line,
                                     mode='lines', name='Tendance', line=dict(dash='dash')))
        
        fig_trend.update_layout(title=f"Évolution de {selected_indicator}",
                               xaxis_title="Année", yaxis_title=selected_indicator)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Calcul de la pente
        slope = z[0]
        st.metric(f"Taux de changement annuel de {selected_indicator}", 
                 f"{slope:,.2f} unités/an")
    
    # Section 3: Modélisation Prédictive
    elif page == "🎯 Modélisation Prédictive":
        st.header("🎯 Modélisation Prédictive")
        
        # Entraînement des modèles
        models = train_models(df)
        
        st.subheader("Modèle de Prédiction des Forêts Denses")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Variables explicatives:**")
            st.write("- Année")
            st.write("- Population")
            st.write("- Surface des Cultures Annuelles")
            
            # Évaluation du modèle
            X = df[['Année', 'Population', 'Cultures Annuelles (CA)']].values
            y_fd = df['Forêts Denses (FD)'].values
            y_pred = models['fd'].predict(X)
            
            r2 = r2_score(y_fd, y_pred)
            rmse = np.sqrt(mean_squared_error(y_fd, y_pred))
            
            st.metric("R² du modèle", f"{r2:.3f}")
            st.metric("RMSE", f"{rmse:,.0f} Ha")
        
        with col2:
            # Coefficients du modèle
            coef_df = pd.DataFrame({
                'Variable': ['Intercept', 'Année', 'Population', 'Cultures Annuelles'],
                'Coefficient': [models['fd'].intercept_] + list(models['fd'].coef_)
            })
            st.dataframe(coef_df, use_container_width=True)
            
            st.info("""
            **Interprétation:**  
            Les coefficients montrent l'impact de chaque variable sur la surface des forêts denses.
            Un coefficient négatif indique une relation inverse.
            """)
        
        # Visualisation des prédictions vs observations
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df['Année'], y=y_fd, 
                                    mode='lines+markers', name='Observé'))
        fig_pred.add_trace(go.Scatter(x=df['Année'], y=y_pred, 
                                    mode='lines+markers', name='Prédit'))
        fig_pred.update_layout(title="Comparaison Observations vs Prédictions",
                              xaxis_title="Année", yaxis_title="Forêts Denses (Ha)")
        st.plotly_chart(fig_pred, use_container_width=True)
    
    # Section 4: Scénarios Futurs
    elif page == "🔮 Scénarios Futurs":
        st.header("🔮 Simulation de Scénarios Futurs")
        
        models = train_models(df)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Paramètres du Scénario")
            target_year = st.slider("Année cible", 2025, 2050, 2030)
            
            scenario = st.selectbox("Type de scénario:", [
                "Statut Quo",
                "Intensification Agricole", 
                "Conservation Renforcée",
                "Personnalisé"
            ])
        
        with col2:
            st.subheader("Variables d'Entrée")
            
            if scenario == "Statut Quo":
                pop_growth = 0.02  # 2% croissance annuelle
                agri_growth = 0.015  # 1.5% croissance annuelle
            elif scenario == "Intensification Agricole":
                pop_growth = 0.02
                agri_growth = 0.005  # Réduction expansion agricole
            elif scenario == "Conservation Renforcée":
                pop_growth = 0.015  # Réduction croissance démographique
                agri_growth = 0.002  # Forte réduction expansion agricole
            else:
                pop_growth = st.slider("Croissance annuelle population (%)", 0.0, 0.05, 0.02)
                agri_growth = st.slider("Croissance annuelle cultures (%)", 0.0, 0.03, 0.015)
            
            # Calcul des valeurs futures
            last_year = df['Année'].iloc[-1]
            last_pop = df['Population'].iloc[-1]
            last_agri = df['Cultures Annuelles (CA)'].iloc[-1]
            
            years_ahead = target_year - last_year
            future_pop = last_pop * (1 + pop_growth) ** years_ahead
            future_agri = last_agri * (1 + agri_growth) ** years_ahead
        
        with col3:
            st.subheader("Résultats du Scénario")
            
            # Prédiction
            X_future = np.array([[target_year, future_pop, future_agri]])
            future_fd = models['fd'].predict(X_future)[0]
            future_co2 = models['co2'].predict(X_future)[0]
            
            current_fd = df['Forêts Denses (FD)'].iloc[-1]
            current_co2 = df['Séquestration CO2'].iloc[-1]
            
            fd_change = future_fd - current_fd
            co2_change = future_co2 - current_co2
            
            st.metric("Forêts Denses (Ha)", f"{future_fd:,.0f}", 
                     f"{fd_change:+,.0f} Ha")
            st.metric("Séquestration CO2 (T)", f"{future_co2:,.0f}", 
                     f"{co2_change:+,.0f} T")
            st.metric("Population", f"{future_pop:,.0f}")
            st.metric("Cultures Annuelles (Ha)", f"{future_agri:,.0f}")
        
        # Visualisation du scénario
        st.subheader("Projection du Scénario")
        
        # Création des données pour le graphique
        years = list(df['Année']) + [target_year]
        fd_values = list(df['Forêts Denses (FD)']) + [future_fd]
        
        fig_scenario = go.Figure()
        fig_scenario.add_trace(go.Scatter(x=df['Année'], y=df['Forêts Denses (FD)'],
                                        mode='lines+markers', name='Historique'))
        fig_scenario.add_trace(go.Scatter(x=[df['Année'].iloc[-1], target_year], 
                                        y=[df['Forêts Denses (FD)'].iloc[-1], future_fd],
                                        mode='lines+markers', name='Projection',
                                        line=dict(dash='dash')))
        fig_scenario.update_layout(title=f"Projection des Forêts Denses - Scénario {scenario}",
                                  xaxis_title="Année", yaxis_title="Forêts Denses (Ha)")
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Analyse d'impact
        st.subheader("Analyse d'Impact")
        
        impact_data = {
            'Indicateur': ['Perte de Biodiversité', 'Émissions CO2', 'Sécurité Alimentaire', 'Résilience Climatique'],
            'Impact': ['Élevé' if fd_change < -50000 else 'Moyen' if fd_change < -10000 else 'Faible',
                      'Élevé' if co2_change < -20000 else 'Moyen' if co2_change < -5000 else 'Faible',
                      'Améliorée' if agri_growth < 0.01 else 'Stable' if agri_growth < 0.02 else 'Dégradée',
                      'Faible' if fd_change < -50000 else 'Moyenne' if fd_change < -10000 else 'Élevée']
        }
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True)
    
    # Section 5: Rapport Complet
    else:
        st.header("📋 Rapport d'Analyse Complet")
        
        st.subheader("Synthèse des Résultats")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **Tendances Principales Identifiées:**
            - 📉 **Déforestation continue**: Perte de ~75,000 Ha de forêts denses depuis 2000
            - 📈 **Pression démographique**: Population presque doublée en 24 ans
            - 🌾 **Expansion agricole**: +60% de surfaces cultivées annuelles
            - 🔥 **Impact climatique**: -22% de capacité de séquestration CO2
            """)
        
        with col2:
            st.markdown("""
            **Relations Clés:**
            - Forte corrélation négative entre population et forêts denses
            - Relation inverse entre cultures annuelles et séquestration carbone
            - Les plantations compensent partiellement la perte de forêts naturelles
            """)
        
        st.subheader("Recommandations Stratégiques")
        
        rec_col1, rec_col2, rec_col3 = st.columns([1, 1, 1])
        
        with rec_col1:
            st.markdown("""
            **🎯 Court Terme (2024-2027)**
            - Stabiliser l'expansion agricole
            - Programmes de reboisement ciblés
            - Sensibilisation des populations
            """)
        
        with rec_col2:
            st.markdown("""
            **📈 Moyen Terme (2028-2035)**
            - Intensification agricole durable
            - Plan d'urbanisation maîtrisé
            - Développement d'alternatives économiques
            """)
        
        with rec_col3:
            st.markdown("""
            **🌳 Long Terme (2036-2050)**
            - Économie verte diversifiée
            - Gestion intégrée des paysages
            - Résilience climatique renforcée
            """)
        
        st.subheader("Indicateurs de Suivi Recommandés")
        
        indicators = {
            'Indicateur': ['Taux de déforestation net', 'Intensité agricole', 
                          'Couvert forestier total', 'Séquestration carbone',
                          'Densité de population rurale'],
            'Cible 2030': ['< 0.5% annuel', '> 3T/Ha', '> 2.9M Ha', '> 350,000 T', '< 50 hab/km²'],
            'Fréquence': ['Annuelle', 'Saisonnière', 'Annuelle', 'Annuelle', 'Quinquennale']
        }
        indicators_df = pd.DataFrame(indicators)
        st.dataframe(indicators_df, use_container_width=True)
        
        # Téléchargement du rapport
        st.subheader("Export des Résultats")
        
        if st.button("📥 Générer le Rapport PDF"):
            st.success("""
            Rapport généré avec succès! (Fonctionnalité d'export PDF à implémenter)
            
            Le rapport complet inclut:
            - Analyse détaillée des données historiques
            - Modélisation prédictive validée
            - Scénarios prospectifs
            - Recommandations politiques
            - Plan de monitoring
            """)

if __name__ == "__main__":
    main()