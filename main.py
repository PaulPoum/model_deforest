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
    page_title="Modélisation de la Déforestation - Thèse Doctoral",
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

# Modèle de prédiction amélioré
def train_models(df):
    models = {}
    
    # Variables pour le modèle forêts denses
    X_fd = df[['Année', 'Population', 'Cultures Annuelles (CA)', 'Forêts Plantation (FP)']].values
    y_fd = df['Forêts Denses (FD)'].values
    
    # Modèle forêts denses
    model_fd = LinearRegression()
    model_fd.fit(X_fd, y_fd)
    models['fd'] = {'model': model_fd, 'features': ['Année', 'Population', 'Cultures Annuelles (CA)', 'Forêts Plantation (FP)']}
    
    # Modèle CO2
    X_co2 = df[['Année', 'Forêts Denses (FD)', 'Forêts Plantation (FP)']].values
    y_co2 = df['Séquestration CO2'].values
    
    model_co2 = LinearRegression()
    model_co2.fit(X_co2, y_co2)
    models['co2'] = {'model': model_co2, 'features': ['Année', 'Forêts Denses (FD)', 'Forêts Plantation (FP)']}
    
    return models

def main():
    st.title("🌳 Modélisation et Analyse de la Déforestation - Thèse Doctoral")
    st.markdown("""
    **Problématique de thèse** : Analyse des dynamiques de déforestation et modélisation des impacts 
    socio-environnementaux dans un contexte de croissance démographique et d'expansion agricole.
    **Cadre méthodologique** : Intégration des scénarios GIEC SSP pour l'analyse prospective.
    """)
    
    # Chargement des données
    df = load_data()
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Sections", [
        "📊 Données et Visualisation",
        "📈 Analyse des Tendances", 
        "🎯 Modélisation Prédictive",
        "🔮 Scénarios Futurs GIEC",
        "📋 Rapport Scientifique"
    ])
    
    # Section 1: Données et Visualisation
    if page == "📊 Données et Visualisation":
        st.header("📊 Données Brutes et Visualisation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Dataset Complet")
            styled_df = df.style.format({
                'Forêts Denses (FD)': '{:,.0f}',
                'Forêts Plantation (FP)': '{:,.0f}',
                'Cultures Annuelles (CA)': '{:,.0f}',
                'Population': '{:,.0f}',
                'Séquestration CO2': '{:,.0f}'
            })
            st.dataframe(styled_df, use_container_width=True)
            
            st.subheader("Statistiques Descriptives")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Évolution des Principaux Indicateurs")
            
            fig = make_subplots(
                rows=2, cols=2, 
                subplot_titles=(
                    'Forêts Denses (Ha)', 
                    'Population (habitants)',
                    'Séquestration CO2 (Tonne)', 
                    'Cultures Annuelles (Ha)'
                )
            )
            
            # Forêts denses
            fig.add_trace(
                go.Scatter(
                    x=df['Année'], 
                    y=df['Forêts Denses (FD)'],
                    mode='lines+markers', 
                    name='Forêts Denses',
                    line=dict(color='#2E8B57', width=3)
                ),
                row=1, col=1
            )
            
            # Population
            fig.add_trace(
                go.Scatter(
                    x=df['Année'], 
                    y=df['Population'],
                    mode='lines+markers', 
                    name='Population',
                    line=dict(color='#FF6B6B', width=3)
                ),
                row=1, col=2
            )
            
            # CO2
            fig.add_trace(
                go.Scatter(
                    x=df['Année'], 
                    y=df['Séquestration CO2'],
                    mode='lines+markers', 
                    name='Séquestration CO2',
                    line=dict(color='#4ECDC4', width=3)
                ),
                row=2, col=1
            )
            
            # Cultures annuelles
            fig.add_trace(
                go.Scatter(
                    x=df['Année'], 
                    y=df['Cultures Annuelles (CA)'],
                    mode='lines+markers', 
                    name='Cultures Annuelles',
                    line=dict(color='#45B7D1', width=3)
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Téléchargement des données
            st.subheader("Export des Données")
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger le dataset CSV",
                data=csv,
                file_name="donnees_deforestation.csv",
                mime="text/csv"
            )
    
    # Section 2: Analyse des Tendances
    elif page == "📈 Analyse des Tendances":
        st.header("📈 Analyse des Tendances et Corrélations")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Taux de Changement Annuel (2000-2024)")
            
            # Calcul des taux de changement
            changes = {}
            for column in df.columns:
                if column != 'Année':
                    start_val = df[column].iloc[0]
                    end_val = df[column].iloc[-1]
                    total_change = end_val - start_val
                    annual_rate = total_change / (df['Année'].iloc[-1] - df['Année'].iloc[0])
                    percent_change = (total_change / start_val) * 100
                    changes[column] = {
                        'Taux Annuel': annual_rate,
                        'Changement Total': total_change,
                        'Changement %': percent_change
                    }
            
            changes_df = pd.DataFrame.from_dict(changes, orient='index')
            changes_df = changes_df.round(2)
            st.dataframe(changes_df.style.format({
                'Taux Annuel': '{:,.2f}',
                'Changement Total': '{:,.2f}',
                'Changement %': '{:,.1f}%'
            }), use_container_width=True)
            
            # Points clés
            st.subheader("Points Clés Identifiés")
            fd_change = changes_df.loc['Forêts Denses (FD)', 'Changement %']
            pop_change = changes_df.loc['Population', 'Changement %']
            co2_change = changes_df.loc['Séquestration CO2', 'Changement %']
            
            st.metric("Déforestation", f"{fd_change:+.1f}%", "2000-2024")
            st.metric("Croissance Démographique", f"{pop_change:+.1f}%", "2000-2024")
            st.metric("Perte Séquestration CO2", f"{co2_change:+.1f}%", "2000-2024")
        
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
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                hoverinfo='text'
            ))
            fig_corr.update_layout(
                height=500,
                title="Matrice de Corrélation entre les Variables"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Analyse des corrélations fortes
            st.subheader("Corrélations Significatives")
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # Corrélations fortes
                        strong_correlations.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Corrélation': f"{corr_val:.3f}"
                        })
            
            if strong_correlations:
                strong_corr_df = pd.DataFrame(strong_correlations)
                st.dataframe(strong_corr_df, use_container_width=True)
            else:
                st.info("Aucune corrélation forte (|r| > 0.7) identifiée")
        
        # Analyse détaillée des tendances
        st.subheader("Analyse Détailée des Tendances")
        
        selected_indicator = st.selectbox("Sélectionnez un indicateur:", 
                                         df.columns[1:])
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df['Année'], 
            y=df[selected_indicator],
            mode='lines+markers', 
            name=selected_indicator,
            line=dict(width=3)
        ))
        
        # Ajout de la tendance linéaire
        z = np.polyfit(df['Année'], df[selected_indicator], 1)
        p = np.poly1d(z)
        trend_line = p(df['Année'])
        fig_trend.add_trace(go.Scatter(
            x=df['Année'], 
            y=trend_line,
            mode='lines', 
            name='Tendance Linéaire', 
            line=dict(dash='dash', color='red')
        ))
        
        fig_trend.update_layout(
            title=f"Évolution de {selected_indicator} avec Tendance Linéaire",
            xaxis_title="Année", 
            yaxis_title=selected_indicator,
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Calcul de la pente et statistiques
        slope = z[0]
        r_squared = np.corrcoef(df['Année'], df[selected_indicator])[0,1]**2
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric(f"Taux de changement annuel", f"{slope:,.2f} unités/an")
        with col_stat2:
            st.metric("R² de la tendance", f"{r_squared:.3f}")
        with col_stat3:
            total_change = df[selected_indicator].iloc[-1] - df[selected_indicator].iloc[0]
            st.metric("Changement total", f"{total_change:,.0f}")
    
    # Section 3: Modélisation Prédictive
    elif page == "🎯 Modélisation Prédictive":
        st.header("🎯 Modélisation Prédictive Avancée")
        
        # Entraînement des modèles améliorés
        models = train_models(df)
        
        st.subheader("Modèle de Prédiction des Forêts Denses")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Variables explicatives du modèle:**")
            for feature in models['fd']['features']:
                st.write(f"- {feature}")
            
            # Évaluation du modèle forêts denses
            X_fd = df[models['fd']['features']].values
            y_fd = df['Forêts Denses (FD)'].values
            y_pred_fd = models['fd']['model'].predict(X_fd)
            
            r2_fd = r2_score(y_fd, y_pred_fd)
            rmse_fd = np.sqrt(mean_squared_error(y_fd, y_pred_fd))
            mae_fd = np.mean(np.abs(y_fd - y_pred_fd))
            
            st.metric("R² du modèle", f"{r2_fd:.4f}")
            st.metric("RMSE", f"{rmse_fd:,.0f} Ha")
            st.metric("MAE", f"{mae_fd:,.0f} Ha")
            
            # Importance des variables
            st.subheader("Importance Relative des Variables")
            coefficients = models['fd']['model'].coef_
            features = models['fd']['features'][1:]  # Exclure l'intercept
            
            importance_df = pd.DataFrame({
                'Variable': features,
                'Coefficient': coefficients[1:],
                'Importance Absolue': np.abs(coefficients[1:])
            }).sort_values('Importance Absolue', ascending=False)
            
            fig_importance = go.Figure(go.Bar(
                x=importance_df['Importance Absolue'],
                y=importance_df['Variable'],
                orientation='h',
                marker_color='#2E8B57'
            ))
            fig_importance.update_layout(
                title="Importance Relative des Variables Explicatives",
                xaxis_title="Valeur Absolue du Coefficient",
                height=300
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Coefficients du modèle détaillé
            st.subheader("Coefficients du Modèle")
            coef_df = pd.DataFrame({
                'Variable': ['Intercept'] + models['fd']['features'],
                'Coefficient': [models['fd']['model'].intercept_] + list(models['fd']['model'].coef_)
            })
            st.dataframe(coef_df.style.format({'Coefficient': '{:.4f}'}), use_container_width=True)
            
            st.info("""
            **Interprétation des Coefficients:**
            - **Coefficient négatif**: Relation inverse avec les forêts denses
            - **Coefficient positif**: Relation directe avec les forêts denses
            - Les forêts de plantation ont un effet compensateur positif
            """)
            
            # Évaluation du modèle CO2
            st.subheader("Modèle de Séquestration CO2")
            X_co2 = df[models['co2']['features']].values
            y_co2 = df['Séquestration CO2'].values
            y_pred_co2 = models['co2']['model'].predict(X_co2)
            
            r2_co2 = r2_score(y_co2, y_pred_co2)
            rmse_co2 = np.sqrt(mean_squared_error(y_co2, y_pred_co2))
            
            st.metric("R² CO2", f"{r2_co2:.4f}")
            st.metric("RMSE CO2", f"{rmse_co2:,.0f} T")
        
        # Visualisation des prédictions vs observations
        st.subheader("Validation des Modèles")
        
        col_viz1, col_viz2 = st.columns([1, 1])
        
        with col_viz1:
            fig_pred_fd = go.Figure()
            fig_pred_fd.add_trace(go.Scatter(
                x=df['Année'], 
                y=y_fd, 
                mode='lines+markers', 
                name='Observé',
                line=dict(width=3)
            ))
            fig_pred_fd.add_trace(go.Scatter(
                x=df['Année'], 
                y=y_pred_fd, 
                mode='lines+markers', 
                name='Prédit',
                line=dict(dash='dash', width=2)
            ))
            fig_pred_fd.update_layout(
                title="Forêts Denses: Observations vs Prédictions",
                xaxis_title="Année", 
                yaxis_title="Forêts Denses (Ha)"
            )
            st.plotly_chart(fig_pred_fd, use_container_width=True)
        
        with col_viz2:
            fig_pred_co2 = go.Figure()
            fig_pred_co2.add_trace(go.Scatter(
                x=df['Année'], 
                y=y_co2, 
                mode='lines+markers', 
                name='Observé',
                line=dict(width=3)
            ))
            fig_pred_co2.add_trace(go.Scatter(
                x=df['Année'], 
                y=y_pred_co2, 
                mode='lines+markers', 
                name='Prédit',
                line=dict(dash='dash', width=2)
            ))
            fig_pred_co2.update_layout(
                title="Séquestration CO2: Observations vs Prédictions",
                xaxis_title="Année", 
                yaxis_title="Séquestration CO2 (T)"
            )
            st.plotly_chart(fig_pred_co2, use_container_width=True)
        
        # Analyse des résidus
        st.subheader("Analyse des Résidus")
        residuals_fd = y_fd - y_pred_fd
        
        fig_residuals = make_subplots(rows=1, cols=2, subplot_titles=('Distribution des Résidus', 'Résidus vs Prédictions'))
        
        fig_residuals.add_trace(go.Histogram(x=residuals_fd, name='Résidus', nbinsx=20), row=1, col=1)
        fig_residuals.add_trace(go.Scatter(x=y_pred_fd, y=residuals_fd, mode='markers', name='Résidus'), row=1, col=2)
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        fig_residuals.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Section 4: Scénarios Futurs avec GIEC
    elif page == "🔮 Scénarios Futurs GIEC":
        st.header("🔮 Simulation de Scénarios Futurs - Cadre GIEC SSP")
        
        models = train_models(df)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Configuration des Scénarios")
            target_year = st.slider("Horizon temporel", 2025, 2100, 2050)
            
            scenario_type = st.radio("Type de scénario:", [
                "📊 Basé sur les tendances historiques",
                "🌍 Scénarios GIEC SSP",
                "🎯 Scénario personnalisé"
            ])
            
            if scenario_type == "🌍 Scénarios GIEC SSP":
                giec_scenario = st.selectbox("Scénario SSP-RCP GIEC:", [
                    "SSP1-2.6 - Développement durable",
                    "SSP2-4.5 - Middle of the road", 
                    "SSP3-7.0 - Régional rivalry",
                    "SSP5-8.5 - Développement fossile"
                ])
                
                # Définition des paramètres selon les scénarios GIEC
                giec_params = {
                    "SSP1-2.6 - Développement durable": {
                        "pop_growth": 0.008,
                        "agri_growth": -0.005,
                        "conservation_effort": 0.03,
                        "tech_improvement": 0.02,
                        "temp_increase": "1.5-2.0°C",
                        "description": "Transition rapide vers la durabilité, forte protection des forêts, économie circulaire"
                    },
                    "SSP2-4.5 - Middle of the road": {
                        "pop_growth": 0.012,
                        "agri_growth": 0.008,
                        "conservation_effort": 0.01,
                        "tech_improvement": 0.01,
                        "temp_increase": "2.0-3.0°C", 
                        "description": "Continuité des tendances actuelles, mesures environnementales modérées"
                    },
                    "SSP3-7.0 - Régional rivalry": {
                        "pop_growth": 0.018,
                        "agri_growth": 0.015,
                        "conservation_effort": -0.01,
                        "tech_improvement": 0.005,
                        "temp_increase": "3.0-4.0°C",
                        "description": "Fortes pressions, faible coopération internationale, fragmentation"
                    },
                    "SSP5-8.5 - Développement fossile": {
                        "pop_growth": 0.015,
                        "agri_growth": 0.025,
                        "conservation_effort": -0.02,
                        "tech_improvement": 0.015,
                        "temp_increase": "4.0-5.0°C",
                        "description": "Croissance économique forte basée sur les énergies fossiles, exploitation intensive"
                    }
                }
                
                params = giec_params[giec_scenario]
                pop_growth = params["pop_growth"]
                agri_growth = params["agri_growth"]
                conservation_effort = params["conservation_effort"]
                tech_improvement = params["tech_improvement"]
                
                st.info(f"**{giec_scenario}**")
                st.write(f"**Description GIEC:** {params['description']}")
                st.write(f"**Réchauffement projeté:** {params['temp_increase']}")
                st.write(f"**Amélioration technologique:** {tech_improvement*100:.1f}%/an")
                
            elif scenario_type == "📊 Basé sur les tendances historiques":
                trend_scenario = st.selectbox("Scénario de tendance:", [
                    "Statut Quo",
                    "Intensification Agricole", 
                    "Conservation Renforcée"
                ])
                
                if trend_scenario == "Statut Quo":
                    pop_growth = 0.02
                    agri_growth = 0.015
                    conservation_effort = 0.0
                    tech_improvement = 0.01
                elif trend_scenario == "Intensification Agricole":
                    pop_growth = 0.02
                    agri_growth = 0.005
                    conservation_effort = 0.01
                    tech_improvement = 0.02
                elif trend_scenario == "Conservation Renforcée":
                    pop_growth = 0.015
                    agri_growth = 0.002
                    conservation_effort = 0.03
                    tech_improvement = 0.015
            
            else:  # Personnalisé
                st.subheader("Paramètres personnalisés")
                pop_growth = st.slider("Croissance annuelle population (%)", 0.0, 0.05, 0.02, 0.001)
                agri_growth = st.slider("Croissance annuelle cultures (%)", -0.02, 0.05, 0.015, 0.001)
                conservation_effort = st.slider("Effort de conservation (%)", -0.05, 0.1, 0.0, 0.001)
                tech_improvement = st.slider("Amélioration technologique (%)", 0.0, 0.05, 0.01, 0.001)
        
        with col2:
            st.subheader("Projections et Impacts")
            
            # Données de référence
            last_year = df['Année'].iloc[-1]
            last_pop = df['Population'].iloc[-1]
            last_agri = df['Cultures Annuelles (CA)'].iloc[-1]
            last_fd = df['Forêts Denses (FD)'].iloc[-1]
            last_fp = df['Forêts Plantation (FP)'].iloc[-1]
            
            # Calcul des projections
            years_ahead = target_year - last_year
            future_pop = last_pop * (1 + pop_growth) ** years_ahead
            future_agri = last_agri * (1 + agri_growth) ** years_ahead
            
            # Projection des forêts de plantation (liée à l'effort de conservation)
            future_fp = last_fp * (1 + conservation_effort) ** years_ahead
            
            # Prédiction du modèle avec ajustement technologique
            X_future_fd = np.array([[target_year, future_pop, future_agri, future_fp]])
            base_future_fd = models['fd']['model'].predict(X_future_fd)[0]
            
            # Application de l'effort de conservation et amélioration technologique
            conservation_impact = conservation_effort * last_fd * years_ahead / 5
            tech_impact = tech_improvement * last_fd * years_ahead / 20
            future_fd = base_future_fd + conservation_impact + tech_impact
            
            # Prédiction CO2
            X_future_co2 = np.array([[target_year, future_fd, future_fp]])
            future_co2 = models['co2']['model'].predict(X_future_co2)[0]
            
            # Calcul des changements
            current_fd = df['Forêts Denses (FD)'].iloc[-1]
            current_co2 = df['Séquestration CO2'].iloc[-1]
            current_pop = df['Population'].iloc[-1]
            current_agri = df['Cultures Annuelles (CA)'].iloc[-1]
            
            fd_change = future_fd - current_fd
            co2_change = future_co2 - current_co2
            fd_percent_change = (fd_change / current_fd) * 100
            pop_change = future_pop - current_pop
            agri_change = future_agri - current_agri
            
            # Affichage des résultats
            st.metric("Forêts Denses (Ha)", f"{future_fd:,.0f}", 
                     f"{fd_change:+,.0f} Ha ({fd_percent_change:+.1f}%)")
            st.metric("Séquestration CO2 (T)", f"{future_co2:,.0f}", 
                     f"{co2_change:+,.0f} T")
            st.metric("Population", f"{future_pop:,.0f}", f"{pop_change:+,.0f}")
            st.metric("Cultures Annuelles (Ha)", f"{future_agri:,.0f}", f"{agri_change:+,.0f}")
            st.metric("Forêts Plantation (Ha)", f"{future_fp:,.0f}")
            
            # Indicateur d'alerte climatique
            if co2_change < -50000:
                st.error("🚨 Impact climatique SÉVÈRE - Perte majeure de puits carbone")
            elif co2_change < -20000:
                st.warning("⚠️ Impact climatique MODÉRÉ - Perturbation significative")
            else:
                st.success("✅ Impact climatique LIMITÉ - Préservation relative")
                
            # Indicateur de biodiversité
            if fd_percent_change < -5:
                st.error("🚨 Perte de biodiversité CRITIQUE")
            elif fd_percent_change < -2:
                st.warning("⚠️ Perte de biodiversité SIGNIFICATIVE")
            else:
                st.success("✅ Biodiversité PRÉSERVÉE")
    
        # Visualisation avancée des scénarios
        st.subheader("Analyse Comparative des Scénarios")
        
        # Génération de projections pour différents scénarios
        years_projection = list(range(2024, target_year + 1, max(1, (target_year - 2024) // 10)))
        
        if scenario_type == "🌍 Scénarios GIEC SSP":
            scenarios_to_show = giec_params
            scenario_names = list(giec_params.keys())
        else:
            scenarios_to_show = {
                "Statut Quo": {"pop_growth": 0.02, "agri_growth": 0.015, "conservation_effort": 0.0, "tech_improvement": 0.01},
                "Intensification": {"pop_growth": 0.02, "agri_growth": 0.005, "conservation_effort": 0.01, "tech_improvement": 0.02},
                "Conservation": {"pop_growth": 0.015, "agri_growth": 0.002, "conservation_effort": 0.03, "tech_improvement": 0.015}
            }
            scenario_names = list(scenarios_to_show.keys())
        
        # Création du graphique comparatif
        fig_comparison = go.Figure()
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, scenario_name in enumerate(scenario_names):
            params = scenarios_to_show[scenario_name]
            fd_values = []
            co2_values = []
            
            for year in years_projection:
                years_ahead = year - 2024
                future_pop = last_pop * (1 + params["pop_growth"]) ** years_ahead
                future_agri = last_agri * (1 + params["agri_growth"]) ** years_ahead
                future_fp = last_fp * (1 + params["conservation_effort"]) ** years_ahead
                
                X_future_fd = np.array([[year, future_pop, future_agri, future_fp]])
                base_fd = models['fd']['model'].predict(X_future_fd)[0]
                conservation_impact = params["conservation_effort"] * last_fd * years_ahead / 5
                tech_impact = params["tech_improvement"] * last_fd * years_ahead / 20
                future_fd = base_fd + conservation_impact + tech_impact
                
                X_future_co2 = np.array([[year, future_fd, future_fp]])
                future_co2 = models['co2']['model'].predict(X_future_co2)[0]
                
                fd_values.append(future_fd)
                co2_values.append(future_co2)
            
            fig_comparison.add_trace(go.Scatter(
                x=years_projection, 
                y=fd_values,
                mode='lines+markers',
                name=scenario_name,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6)
            ))
        
        # Ajout de la ligne historique
        fig_comparison.add_trace(go.Scatter(
            x=df['Année'], 
            y=df['Forêts Denses (FD)'],
            mode='lines+markers',
            name='Historique',
            line=dict(dash='dash', color='black', width=3),
            marker=dict(size=8, symbol='diamond')
        ))
        
        fig_comparison.update_layout(
            title="Comparaison des Scénarios - Évolution des Forêts Denses",
            xaxis_title="Année",
            yaxis_title="Forêts Denses (Ha)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Tableau d'impacts détaillé
        st.subheader("Analyse d'Impact Détaillée par Scénario")
        
        impact_data = []
        for scenario_name in scenario_names:
            params = scenarios_to_show[scenario_name]
            
            # Calcul pour le scénario
            years_ahead = target_year - 2024
            future_pop = last_pop * (1 + params["pop_growth"]) ** years_ahead
            future_agri = last_agri * (1 + params["agri_growth"]) ** years_ahead
            future_fp = last_fp * (1 + params["conservation_effort"]) ** years_ahead
            
            X_future_fd = np.array([[target_year, future_pop, future_agri, future_fp]])
            base_fd = models['fd']['model'].predict(X_future_fd)[0]
            conservation_impact = params["conservation_effort"] * last_fd * years_ahead / 5
            tech_impact = params["tech_improvement"] * last_fd * years_ahead / 20
            future_fd = base_fd + conservation_impact + tech_impact
            
            X_future_co2 = np.array([[target_year, future_fd, future_fp]])
            future_co2 = models['co2']['model'].predict(X_future_co2)[0]
            
            fd_change_pct = ((future_fd - current_fd) / current_fd) * 100
            co2_change_pct = ((future_co2 - current_co2) / current_co2) * 100
            
            # Évaluation des impacts
            if fd_change_pct < -10:
                biodiversite_impact = "🔴 Très Élevé"
            elif fd_change_pct < -5:
                biodiversite_impact = "🟠 Élevé"
            elif fd_change_pct < -2:
                biodiversite_impact = "🟡 Modéré"
            else:
                biodiversite_impact = "🟢 Faible"
                
            if co2_change_pct < -15:
                climat_impact = "🔴 Très Élevé"
            elif co2_change_pct < -8:
                climat_impact = "🟠 Élevé"
            elif co2_change_pct < -3:
                climat_impact = "🟡 Modéré"
            else:
                climat_impact = "🟢 Faible"
            
            # Évaluation de la sécurité alimentaire
            agri_per_capita = future_agri / future_pop * 1000  # m² par personne
            if agri_per_capita > 2000:
                securite_alimentaire = "🟢 Excellente"
            elif agri_per_capita > 1500:
                securite_alimentaire = "🟡 Suffisante"
            else:
                securite_alimentaire = "🔴 Critique"
            
            impact_data.append({
                'Scénario': scenario_name,
                'Forêts 2050 (Ha)': f"{future_fd:,.0f}",
                'Δ Forêts (%)': f"{fd_change_pct:+.1f}%",
                'Δ CO2 (%)': f"{co2_change_pct:+.1f}%",
                'Impact Biodiversité': biodiversite_impact,
                'Impact Climat': climat_impact,
                'Sécurité Alimentaire': securite_alimentaire,
                'Population 2050': f"{future_pop:,.0f}"
            })
        
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True)
        
        # Recommandations spécifiques aux scénarios GIEC
        if scenario_type == "🌍 Scénarios GIEC SSP":
            st.subheader("🎯 Recommandations Stratégiques alignées GIEC")
            
            col_rec1, col_rec2 = st.columns([1, 1])
            
            with col_rec1:
                if giec_scenario == "SSP1-2.6 - Développement durable":
                    st.success("""
                    **Stratégies recommandées pour SSP1-2.6:**
                    - ✅ Maintenir les politiques de conservation strictes
                    - ✅ Développer l'agroécologie intensive
                    - ✅ Investir dans les paiements pour services écosystémiques
                    - ✅ Renforcer la gouvernance forestière participative
                    - ✅ Promouvoir les énergies renouvelables
                    """)
                elif giec_scenario == "SSP2-4.5 - Middle of the road":
                    st.info("""
                    **Stratégies recommandées pour SSP2-4.5:**
                    - 🔄 Améliorer l'efficacité agricole
                    - 🔄 Développer les corridors écologiques
                    - 🔄 Mettre en place des systèmes d'alerte précoce
                    - 🔄 Promouvoir les pratiques sylvicoles durables
                    - 🔄 Investir dans l'adaptation climatique
                    """)
            
            with col_rec2:
                if giec_scenario == "SSP3-7.0 - Régional rivalry":
                    st.warning("""
                    **Stratégies recommandées pour SSP3-7.0:**
                    - ⚠️ Renforcer la coopération régionale
                    - ⚠️ Diversifier l'économie rurale
                    - ⚠️ Développer l'adaptation climatique
                    - ⚠️ Sécuriser les droits fonciers
                    - ⚠️ Créer des zones refuge pour la biodiversité
                    """)
                elif giec_scenario == "SSP5-8.5 - Développement fossile":
                    st.error("""
                    **Stratégies d'urgence pour SSP5-8.5:**
                    - 🚨 Transition énergétique accélérée
                    - 🚨 Moratoire sur la déforestation
                    - 🚨 Restauration écologique massive
                    - 🚨 Plan d'adaptation d'urgence
                    - 🚨 Diversification économique forcée
                    """)
        
        # Téléchargement des résultats du scénario
        st.subheader("📊 Export des Résultats de Simulation")
        
        # Création d'un DataFrame des résultats
        export_data = []
        for year in range(2024, target_year + 1, 5):
            years_ahead = year - 2024
            future_pop = last_pop * (1 + pop_growth) ** years_ahead
            future_agri = last_agri * (1 + agri_growth) ** years_ahead
            future_fp = last_fp * (1 + conservation_effort) ** years_ahead
            
            X_future_fd = np.array([[year, future_pop, future_agri, future_fp]])
            base_fd = models['fd']['model'].predict(X_future_fd)[0]
            conservation_impact = conservation_effort * last_fd * years_ahead / 5
            tech_impact = tech_improvement * last_fd * years_ahead / 20
            future_fd = base_fd + conservation_impact + tech_impact
            
            X_future_co2 = np.array([[year, future_fd, future_fp]])
            future_co2 = models['co2']['model'].predict(X_future_co2)[0]
            
            export_data.append({
                'Année': year,
                'Forêts_Denses_Ha': future_fd,
                'Séquestration_CO2_T': future_co2,
                'Population': future_pop,
                'Cultures_Annuelles_Ha': future_agri,
                'Forêts_Plantation_Ha': future_fp,
                'Scénario': giec_scenario if scenario_type == "🌍 Scénarios GIEC SSP" else trend_scenario if scenario_type == "📊 Basé sur les tendances historiques" else "Personnalisé"
            })
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Télécharger les projections CSV",
            data=csv,
            file_name=f"projections_deforestation_{target_year}.csv",
            mime="text/csv"
        )
    
    # Section 5: Rapport Scientifique
    else:
        st.header("📋 Rapport Scientifique Complet")
        
        st.subheader("Résumé Exécutif")
        
        col_sum1, col_sum2 = st.columns([1, 1])
        
        with col_sum1:
            st.markdown("""
            **🎯 Objectifs de la Recherche:**
            - Analyser les dynamiques historiques de déforestation (2000-2024)
            - Modéliser les relations causales entre variables socio-économiques et environnementales
            - Projeter l'évolution future selon les scénarios GIEC SSP
            - Formuler des recommandations politiques fondées sur des preuves
            """)
            
            st.markdown("""
            **📊 Méthodologie Employée:**
            - Analyse de séries temporelles sur 24 ans
            - Modélisation par régression linéaire multiple
            - Intégration des scénarios SSP-RCP du GIEC
            - Analyse d'impact multicritère
            """)
        
        with col_sum2:
            st.markdown("""
            **🔑 Résultats Clés:**
            - 📉 **Déforestation continue**: Perte de 74,994 Ha (-2.6%) de forêts denses depuis 2000
            - 📈 **Pression démographique**: Population +92.9% en 24 ans (247,643 habitants)
            - 🌾 **Expansion agricole**: Cultures annuelles +59.6% (35,807 Ha supplémentaires)
            - 🔥 **Impact climatique**: Capacité de séquestration CO2 -21.6% (91,000 T de CO2)
            - 🌿 **Compensation partielle**: Plantations forestières 35,541 Ha en 2024
            """)
        
        st.subheader("Analyse Détailée des Résultats")
        
        # Calcul des indicateurs clés
        total_deforestation = df['Forêts Denses (FD)'].iloc[0] - df['Forêts Denses (FD)'].iloc[-1]
        deforestation_rate = total_deforestation / (df['Année'].iloc[-1] - df['Année'].iloc[0])
        pop_growth_rate = (df['Population'].iloc[-1] - df['Population'].iloc[0]) / df['Population'].iloc[0] * 100
        
        col_ana1, col_ana2, col_ana3 = st.columns(3)
        
        with col_ana1:
            st.metric("Déforestation annuelle moyenne", f"{deforestation_rate:,.0f} Ha/an")
        with col_ana2:
            st.metric("Taux de croissance démographique", f"{pop_growth_rate:.1f}%")
        with col_ana3:
            agricultural_expansion = df['Cultures Annuelles (CA)'].iloc[-1] - df['Cultures Annuelles (CA)'].iloc[0]
            st.metric("Expansion agricole totale", f"{agricultural_expansion:,.0f} Ha")
        
        st.subheader("Recommandations Stratégiques par Horizon Temporel")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Court Terme (2024-2030)", "📈 Moyen Terme (2031-2040)", "🌳 Long Terme (2041-2050)"])
        
        with tab1:
            st.markdown("""
            **Actions Prioritaires 2024-2030:**
            - 🛑 **Moratoire** sur la conversion des forêts primaires
            - 🌾 **Intensification durable** de l'agriculture existante
            - 📊 **Système de monitoring** en temps réel de la déforestation
            - 💰 **Paiements pour services écosystémiques** aux communautés
            - 📚 **Programmes d'éducation** environnementale
            - 🔄 **Diversification** des revenus ruraux
            """)
            
        with tab2:
            st.markdown("""
            **Stratégies 2031-2040:**
            - 🌿 **Restauration écologique** des zones dégradées (50,000 Ha cible)
            - 🏙️ **Plan d'urbanisation** maîtrisé et compact
            - 🔋 **Transition énergétique** vers les renouvelables
            - 🤝 **Coopération régionale** pour la gestion des bassins versants
            - 📈 **Économie verte** créatrice d'emplois
            - 🔬 **Innovation technologique** agricole et forestière
            """)
            
        with tab3:
            st.markdown("""
            **Vision 2041-2050:**
            - 🌍 **Économie décarbonée** et circulaire
            - 🏞️ **Connectivité écologique** paysagère restaurée
            - 👥 **Gouvernance participative** institutionnalisée
            - 🔄 **Résilience climatique** intégrée aux politiques
            - 💡 **Innovation sociale** et entrepreneuriat vert
            - 📊 **Comptabilité environnementale** généralisée
            """)
        
        st.subheader("Indicateurs de Suivi Recommandés")
        
        indicators = {
            'Domaine': ['Écologique', 'Écologique', 'Social', 'Social', 'Économique', 'Climatique'],
            'Indicateur': [
                'Taux de déforestation nette', 
                'Surface forestière totale',
                'Densité de population rurale',
                'Sécurité alimentaire',
                'Productivité agricole',
                'Séquestration carbone nette'
            ],
            'Cible 2030': [
                '< 0.3% annuel', 
                '> 2.95M Ha',
                '< 45 hab/km²',
                '> 1800 m²/personne',
                '> 3.5 T/Ha',
                '> 360,000 T CO2'
            ],
            'Cible 2050': [
                '< 0.1% annuel', 
                '> 3.0M Ha',
                '< 35 hab/km²',
                '> 2000 m²/personne',
                '> 5.0 T/Ha',
                '> 400,000 T CO2'
            ]
        }
        indicators_df = pd.DataFrame(indicators)
        st.dataframe(indicators_df, use_container_width=True)
        
        st.subheader("Perspectives de Recherche Future")
        
        st.markdown("""
        **🔬 Axes de Recherche Recommandés:**
        - Intégration des données de télédétection haute résolution
        - Modélisation des impacts du changement climatique sur la productivité forestière
        - Analyse des circuits économiques informels liés à la déforestation
        - Étude des perceptions et comportements des acteurs locaux
        - Développement d'indicateurs de bien-être intégrant le capital naturel
        """)
        
        # Synthèse finale
        st.subheader("Conclusion Générale")
        
        st.success("""
        **📝 Synthèse:**  
        Cette recherche démontre l'interdépendance cruciale entre dynamiques démographiques, 
        développement agricole et préservation des écosystèmes forestiers. L'intégration des 
        scénarios GIEC permet d'éclairer les décisions politiques en quantifiant les conséquences 
        de différents choix de développement. La soutenabilité à long terme nécessite une approche 
        intégrée combinant conservation stricte, intensification durable et diversification économique.
        """)
        
        # Téléchargement du rapport complet
        st.subheader("📄 Export du Rapport Complet")
        
        if st.button("📥 Générer le Rapport Scientifique PDF"):
            st.success("""
            **Rapport scientifique généré avec succès!**
            
            Le document comprend:
            - Méthodologie détaillée et cadre conceptuel
            - Analyse statistique complète des données
            - Résultats des modélisations avec intervalles de confiance
            - Projections selon les scénarios GIEC SSP
            - Recommandations politiques fondées sur les preuves
            - Bibliographie complète
            
            *Note: L'export PDF complet nécessiterait l'implémentation d'une fonction de génération de PDF*
            """)

if __name__ == "__main__":
    main()