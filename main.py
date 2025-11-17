import warnings
warnings.filterwarnings("ignore")

import logging
from datetime import datetime
import json

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone

import xgboost as xgb

import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# CONFIG STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🌍 Plateforme Avancée de Modélisation de la Déforestation",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. DONNÉES FORESTIÈRES PAR DÉFAUT
# =============================================================================

@st.cache_data(ttl=3600)
def load_default_forest_data():
    """
    Charge un dataset forestier par défaut (interpolé annuellement 2000–2024).
    Colonnes : Année, Forêts Denses (FD), Forêts Plantation (FP),
    Cultures Annuelles (CA), Cultures Pérennes (CP), Prairies (P),
    Terrains Habités (TH), Eaux (E), Autres (A), Population, Séquestration CO2.
    """
    base_data = {
        "Année": [2000, 2005, 2010, 2015, 2020, 2024],
        "Forêts Denses (FD)": [2935120.85, 2928000.00, 2915092.02, 2885000.00, 2864439.44, 2860127.18],
        "Forêts Plantation (FP)": [0.0, 5000.00, 8000.00, 20000.00, 34277.61, 35541.06],
        "Cultures Annuelles (CA)": [60056.23, 65000.00, 75187.93, 82000.00, 89736.12, 95863.08],
        "Cultures Pérennes (CP)": [1681.71, 3000.00, 6365.12, 7000.00, 7412.82, 5109.37],
        "Prairies (P)": [2763.86, 2500.00, 1948.96, 2000.00, 2017.31, 2036.02],
        "Terrains Habités (TH)": [5432.14, 5300.00, 5247.34, 5300.00, 5402.11, 5363.86],
        "Eaux (E)": [9509.11, 10000.00, 10711.23, 10500.00, 10876.27, 10131.16],
        "Autres (A)": [269.94, 275.00, 281.23, 400.00, 672.14, 662.11],
        "Population": [128346, 160000, 224254, 235000, 240915, 247643],
        "Séquestration CO2": [420915, 410000, 399854, 370000, 347443, 329920],
    }

    df_base = pd.DataFrame(base_data)

    years_full = list(range(2000, 2025))
    df_full = pd.DataFrame({"Année": years_full})

    for col in df_base.columns:
        if col != "Année":
            df_full[col] = np.interp(
                years_full,
                df_base["Année"],
                df_base[col]
            )

    return df_full


# =============================================================================
# 2. DONNÉES CLIMATIQUES ONACC (STANDARD)
# =============================================================================

@st.cache_data
def load_onacc_climate_csv(uploaded_file):
    """
    Charge un fichier climat ONACC standard :
    ID_Localité, Localité, Type_Localité, Latitude, Longitude,
    Année, Precipitation, Temperature, Source_Climat, Scenario_Climat
    """
    df = pd.read_csv(uploaded_file)

    required_cols = [
        "ID_Localité", "Localité", "Type_Localité",
        "Latitude", "Longitude", "Année",
        "Precipitation", "Temperature",
        "Source_Climat", "Scenario_Climat",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes dans le fichier climat ONACC : {missing}")
        st.stop()

    df["Année"] = df["Année"].astype(int)
    df["Precipitation"] = df["Precipitation"].astype(float)
    df["Temperature"] = df["Temperature"].astype(float)

    return df


def merge_climate_data(df_main, df_climate, locality):
    """
    Intègre des données climatiques externes (ONACC) dans le dataset principal.

    df_main : DataFrame principal (forestier) avec une colonne 'Année'
    df_climate : DataFrame climat ONACC
    locality : nom de la localité à utiliser
    """
    df_clim = df_climate.copy()
    df_clim = df_clim[df_clim["Localité"] == locality]

    # Agrégation annuelle au cas où
    df_clim = (
        df_clim
        .groupby("Année", as_index=False)[["Precipitation", "Temperature"]]
        .mean()
    )

    df_merged = df_main.merge(df_clim, on="Année", how="left")
    df_merged["Precipitation"] = df_merged["Precipitation"].interpolate()
    df_merged["Temperature"] = df_merged["Temperature"].interpolate()

    return df_merged


# =============================================================================
# 3. ENRICHISSEMENT DU DATASET
# =============================================================================

def enrich_dataset(df, df_climate=None, locality=None):
    """
    Enrichit le dataset forestier avec :
      - Climat (ERA5/CMIP/ONACC) si fourni, sinon climat simulé
      - Variables économiques simulées
      - Indices composites (pression anthropique, résilience)
      - Taux de croissance annuels
    """
    df_enriched = df.copy()

    # 1. Climat : réel (ONACC) ou simulé
    if df_climate is not None and locality is not None:
        df_enriched = merge_climate_data(df_enriched, df_climate, locality)
    else:
        np.random.seed(42)
        df_enriched["Precipitation"] = (
            np.random.normal(1500, 200, len(df_enriched))
            + (df_enriched["Année"] - df_enriched["Année"].min()) * 5
        )
        df_enriched["Temperature"] = (
            25 + (df_enriched["Année"] - df_enriched["Année"].min()) * 0.02
        )

    # 2. Variables économiques simulées
    np.random.seed(42)
    df_enriched["PIB_Agricole"] = (
        df_enriched["Cultures Annuelles (CA)"]
        * np.random.normal(1000, 100, len(df_enriched))
    )
    df_enriched["Investissement_Conservation"] = df_enriched["Forêts Plantation (FP)"] * 500

    # 3. Indices composites
    df_enriched["Pression_Anthropique"] = (
        df_enriched["Population"] / df_enriched["Forêts Denses (FD)"] * 1_000_000
    )
    df_enriched["Résilience_Ecologique"] = (
        df_enriched["Forêts Plantation (FP)"] / (df_enriched["Cultures Annuelles (CA)"] + 1.0)
    )

    # 4. Taux de changement annuel
    for column in ["Forêts Denses (FD)", "Population", "Cultures Annuelles (CA)"]:
        df_enriched[f"{column}_Croissance"] = df_enriched[column].pct_change() * 100.0

    return df_enriched


# =============================================================================
# 4. MODÉLISATION AVANCÉE (CORRIGÉE)
# =============================================================================

def train_advanced_models(df, model_config):
    """
    Entraîne plusieurs modèles et compare leurs performances.

    df : DataFrame enrichi (incluant au minimum :
         Année, Forêts Denses (FD), Forêts Plantation (FP),
         Cultures Annuelles (CA), Population, Séquestration CO2,
         Temperature, Pression_Anthropique, Precipitation)
    model_config : dict retourné par setup_sidebar()
    """
    models = {}

    # Features pour forêts denses
    features_fd = [
        "Année",
        "Population",
        "Cultures Annuelles (CA)",
        "Forêts Plantation (FP)",
        "Pression_Anthropique",
        "Temperature",
        "Precipitation",
    ]

    # Vérifier que les features existent
    missing_feats = [f for f in features_fd if f not in df.columns]
    if missing_feats:
        st.error(f"Variables manquantes pour l'entraînement : {missing_feats}")
        st.stop()

    X = df[features_fd]
    y_fd = df["Forêts Denses (FD)"]
    y_co2 = df["Séquestration CO2"]

    # Paramètres
    params = model_config.get("model_params", {}) or {}
    use_cv = model_config.get("cross_validation", True)

    # Validation croisée temporelle
    tscv = None
    if use_cv and len(df) >= 10:
        n_splits = min(3, max(2, len(df) // 5))
        tscv = TimeSeriesSplit(n_splits=n_splits)

    model_type = model_config.get("model_type", "AutoML")

    if model_type == "Régression Linéaire":
        model_configs = {"linear": LinearRegression()}

    elif model_type == "Random Forest":
        model_configs = {
            "random_forest": RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=42,
            )
        }

    elif model_type == "Gradient Boosting":
        model_configs = {
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 3),
                random_state=42,
            )
        }

    elif model_type == "XGBoost":
        model_configs = {
            "xgboost": xgb.XGBRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 4),
                random_state=42,
            )
        }

    else:  # AutoML
        model_configs = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=42
            ),
            "xgboost": xgb.XGBRegressor(
                n_estimators=100, max_depth=4, random_state=42
            ),
        }

    best_score = -np.inf
    best_model_fd = None
    best_model_name = None

    # Entraînement / évaluation pour FD
    for name, model in model_configs.items():
        cv_scores = None
        mean_score = None

        try:
            if tscv is not None:
                cv_scores = cross_val_score(model, X, y_fd, cv=tscv, scoring="r2")
                mean_score = float(cv_scores.mean())

            model.fit(X, y_fd)

            y_pred = model.predict(X)
            r2 = r2_score(y_fd, y_pred)
            rmse = np.sqrt(mean_squared_error(y_fd, y_pred))
            mae = mean_absolute_error(y_fd, y_pred)

            selection_score = mean_score if mean_score is not None else r2
            if selection_score > best_score:
                best_score = selection_score
                best_model_fd = model
                best_model_name = name

            models[f"fd_{name}"] = {
                "model": model,
                "cv_score": mean_score,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "features": features_fd,
                "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
            }

        except Exception as e:
            st.warning(f"Erreur avec le modèle {name} : {e}")
            continue

    # Meilleur modèle FD
    if best_model_fd is not None:
        best_model_fd.fit(X, y_fd)
        y_pred_best = best_model_fd.predict(X)

        models["fd_best"] = {
            "model": best_model_fd,
            "name": best_model_name,
            "cv_score": best_score if tscv is not None else None,
            "r2": r2_score(y_fd, y_pred_best),
            "rmse": np.sqrt(mean_squared_error(y_fd, y_pred_best)),
            "mae": mean_absolute_error(y_fd, y_pred_best),
            "features": features_fd,
        }

    # Modèle CO2 linéaire
    features_co2 = ["Année", "Forêts Denses (FD)", "Forêts Plantation (FP)", "Temperature"]
    missing_co2 = [f for f in features_co2 if f not in df.columns]
    if missing_co2:
        st.error(f"Variables manquantes pour le modèle CO2 : {missing_co2}")
        st.stop()

    X_co2 = df[features_co2]
    model_co2 = LinearRegression()
    model_co2.fit(X_co2, y_co2)
    y_pred_co2 = model_co2.predict(X_co2)

    models["co2"] = {
        "model": model_co2,
        "r2": r2_score(y_co2, y_pred_co2),
        "rmse": np.sqrt(mean_squared_error(y_co2, y_pred_co2)),
        "features": features_co2,
    }

    return models


def calculate_confidence_intervals(model, X, y, n_bootstrap=100):
    """
    Calcule les intervalles de confiance par bootstrap.
    Retourne (mean_pred, std_pred).
    """
    predictions = []

    for _ in range(n_bootstrap):
        try:
            indices = np.random.choice(len(X), len(X), replace=True)
            if hasattr(X, "iloc"):
                X_boot = X.iloc[indices]
                y_boot = y.iloc[indices]
            else:
                X_boot = X[indices]
                y_boot = y[indices]

            model_boot = clone(model)
            model_boot.fit(X_boot, y_boot)
            pred = model_boot.predict(X)
            predictions.append(pred)
        except Exception:
            continue

    if len(predictions) == 0:
        base_pred = model.predict(X)
        return base_pred, np.zeros_like(base_pred)

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    return mean_pred, std_pred


# =============================================================================
# 5. INTERFACE UTILISATEUR : SIDEBAR & METRICS
# =============================================================================

def setup_sidebar():
    st.sidebar.title("🎛️ Panneau de Configuration")

    st.sidebar.subheader("🔧 Configuration des Modèles")
    model_type = st.sidebar.selectbox(
        "Type de modèle :",
        ["Régression Linéaire", "Random Forest", "Gradient Boosting", "XGBoost", "AutoML"],
    )

    if model_type in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        n_estimators = st.sidebar.slider("Nombre d'arbres", 50, 500, 100)
        max_depth = st.sidebar.slider("Profondeur max", 3, 10, 5)
        model_params = {"n_estimators": n_estimators, "max_depth": max_depth}
    else:
        model_params = {}

    st.sidebar.subheader("📈 Options d'Analyse")
    include_uncertainty = st.sidebar.checkbox("Inclure les intervalles d'incertitude", True)
    cross_validation = st.sidebar.checkbox("Validation croisée", True)
    sensitivity_analysis = st.sidebar.checkbox("Analyse de sensibilité", False)

    st.sidebar.subheader("🔮 Scénarios")
    default_scenario = st.sidebar.selectbox(
        "Scénario par défaut :",
        [
            "SSP1-2.6 - Développement durable",
            "SSP2-4.5 - Middle of the road",
            "SSP3-7.0 - Régional rivalry",
            "SSP5-8.5 - Développement fossile",
        ],
    )

    return {
        "model_type": model_type,
        "model_params": model_params,
        "include_uncertainty": include_uncertainty,
        "cross_validation": cross_validation,
        "sensitivity_analysis": sensitivity_analysis,
        "default_scenario": default_scenario,
    }


def create_real_time_metrics(df):
    st.subheader("📊 Tableau de Bord des Indicateurs Clés")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_fd = df["Forêts Denses (FD)"].iloc[-1]
        change_fd = current_fd - df["Forêts Denses (FD)"].iloc[0]
        pct_change_fd = change_fd / df["Forêts Denses (FD)"].iloc[0] * 100.0
        trend_icon = "📉" if change_fd < 0 else "📈"

        st.metric(
            "Forêts Denses",
            f"{current_fd:,.0f} Ha {trend_icon}",
            f"{pct_change_fd:+.1f}%",
            delta_color="inverse",
        )

    with col2:
        current_co2 = df["Séquestration CO2"].iloc[-1]
        change_co2 = current_co2 - df["Séquestration CO2"].iloc[0]
        pct_change_co2 = change_co2 / df["Séquestration CO2"].iloc[0] * 100.0
        trend_icon = "🔻" if change_co2 < 0 else "🔺"

        st.metric(
            "Séquestration CO2",
            f"{current_co2:,.0f} T {trend_icon}",
            f"{pct_change_co2:+.1f}%",
            delta_color="inverse",
        )

    with col3:
        deforestation_rate = (
            df["Forêts Denses (FD)"].iloc[0] - df["Forêts Denses (FD)"].iloc[-1]
        ) / (df["Année"].iloc[-1] - df["Année"].iloc[0])

        st.metric(
            "Taux Déforestation Annuel",
            f"{deforestation_rate:,.0f} Ha/an",
            "Moyenne 2000-2024",
        )

    with col4:
        agricultural_pressure = (
            df["Cultures Annuelles (CA)"].iloc[-1]
            / df["Forêts Denses (FD)"].iloc[-1]
            * 100.0
        )
        pressure_trend = "⚠️" if agricultural_pressure > 3 else "✅"

        st.metric(
            "Pression Agricole",
            f"{agricultural_pressure:.2f}% {pressure_trend}",
            "Surface cultivée/forêt",
        )


# =============================================================================
# 6. VISUALISATIONS : PRÉDICTIONS & CORRÉLATIONS
# =============================================================================

def plot_predictions_with_uncertainty(df, model_info, target_var, include_uncertainty=True):
    model = model_info["model"]
    features = model_info["features"]

    X = df[features]
    y = df[target_var]

    y_pred = model.predict(X)

    fig = go.Figure()

    if include_uncertainty and len(df) > 5:
        try:
            y_mean, y_std = calculate_confidence_intervals(model, X, y)

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([df["Année"], df["Année"][::-1]]),
                    y=np.concatenate(
                        [y_mean - 1.96 * y_std, (y_mean + 1.96 * y_std)[::-1]]
                    ),
                    fill="toself",
                    fillcolor="rgba(0,100,80,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Intervalle de confiance 95%",
                    showlegend=True,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df["Année"],
                    y=y_mean,
                    line=dict(color="rgb(0,100,80)", width=3),
                    mode="lines",
                    name="Prédiction moyenne",
                    showlegend=True,
                )
            )
        except Exception as e:
            st.warning(f"Impossible de calculer les intervalles de confiance : {e}")
            fig.add_trace(
                go.Scatter(
                    x=df["Année"],
                    y=y_pred,
                    line=dict(color="rgb(0,100,80)", width=3),
                    mode="lines",
                    name="Prédiction",
                    showlegend=True,
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["Année"],
                y=y_pred,
                line=dict(color="rgb(0,100,80)", width=3),
                mode="lines",
                name="Prédiction",
                showlegend=True,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["Année"],
            y=y,
            mode="markers+lines",
            marker=dict(color="red", size=8),
            line=dict(color="red", width=2, dash="dash"),
            name="Observations",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=f"Prédictions {target_var} avec Intervalles de Confiance",
        xaxis_title="Année",
        yaxis_title=target_var,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_advanced_correlation_matrix(df):
    st.subheader("🔗 Analyse des Corrélations Avancée")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_vars = st.multiselect(
        "Sélectionnez les variables pour l'analyse de corrélation :",
        options=numeric_cols,
        default=numeric_cols[:8],
    )

    if len(selected_vars) < 2:
        st.warning("Veuillez sélectionner au moins 2 variables.")
        return

    corr_data = df[selected_vars]
    corr_matrix = corr_data.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Matrice de Corrélation Interactive",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Corrélations Significatives (|r| > 0.7)")
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_correlations.append(
                    {
                        "Variable 1": corr_matrix.columns[i],
                        "Variable 2": corr_matrix.columns[j],
                        "Corrélation": f"{corr_val:.3f}",
                        "Type": "Forte positive" if corr_val > 0 else "Forte négative",
                    }
                )

    if strong_correlations:
        strong_corr_df = pd.DataFrame(strong_correlations)
        st.dataframe(strong_corr_df, use_container_width=True)
    else:
        st.info("Aucune corrélation forte (|r| > 0.7) identifiée.")


# =============================================================================
# 7. SCÉNARIOS GIEC AVEC INCERTITUDES
# =============================================================================

class ScenarioManager:
    def __init__(self, base_year=2024):
        self.base_year = base_year
        self.scenarios = self.initialize_giec_scenarios()

    def initialize_giec_scenarios(self):
        return {
            "SSP1-2.6 - Développement durable": {
                "pop_growth": 0.008,
                "agri_growth": -0.005,
                "conservation_effort": 0.03,
                "tech_improvement": 0.02,
                "climate_impact": -0.001,
                "economic_growth": 0.025,
                "temp_increase": "1.5-2.0°C",
                "description": "Transition rapide vers la durabilité, forte protection des forêts, économie circulaire",
            },
            "SSP2-4.5 - Middle of the road": {
                "pop_growth": 0.012,
                "agri_growth": 0.008,
                "conservation_effort": 0.01,
                "tech_improvement": 0.01,
                "climate_impact": -0.003,
                "economic_growth": 0.03,
                "temp_increase": "2.0-3.0°C",
                "description": "Continuité des tendances actuelles, mesures environnementales modérées",
            },
            "SSP3-7.0 - Régional rivalry": {
                "pop_growth": 0.018,
                "agri_growth": 0.015,
                "conservation_effort": -0.01,
                "tech_improvement": 0.005,
                "climate_impact": -0.008,
                "economic_growth": 0.02,
                "temp_increase": "3.0-4.0°C",
                "description": "Fortes pressions, faible coopération internationale, fragmentation",
            },
            "SSP5-8.5 - Développement fossile": {
                "pop_growth": 0.015,
                "agri_growth": 0.025,
                "conservation_effort": -0.02,
                "tech_improvement": 0.015,
                "climate_impact": -0.015,
                "economic_growth": 0.035,
                "temp_increase": "4.0-5.0°C",
                "description": "Croissance économique forte basée sur les énergies fossiles, exploitation intensive",
            },
        }

    def simulate_scenario(self, scenario_name, models, df, target_year, n_simulations=100):
        if "fd_best" not in models:
            raise ValueError("Le modèle 'fd_best' n'est pas disponible.")
        if "co2" not in models:
            raise ValueError("Le modèle 'co2' n'est pas disponible.")

        scenario = self.scenarios[scenario_name]
        results = []

        last_year = df["Année"].iloc[-1]
        last_pop = df["Population"].iloc[-1]
        last_agri = df["Cultures Annuelles (CA)"].iloc[-1]
        last_fp = df["Forêts Plantation (FP)"].iloc[-1]
        last_fd = df["Forêts Denses (FD)"].iloc[-1]
        last_temp = df["Temperature"].iloc[-1]
        last_prec = df.get("Precipitation", pd.Series([0])).iloc[-1]

        model_fd = models["fd_best"]["model"]
        features_fd = models["fd_best"]["features"]

        model_co2 = models["co2"]["model"]
        features_co2 = models["co2"]["features"]

        years_ahead = max(0, target_year - last_year)

        for _ in range(n_simulations):
            pop_var = np.random.normal(1, 0.1)
            agri_var = np.random.normal(1, 0.15)
            conserv_var = np.random.normal(1, 0.2)
            tech_var = np.random.normal(1, 0.1)

            future_pop = last_pop * (1 + scenario["pop_growth"] * pop_var) ** years_ahead
            future_agri = last_agri * (1 + scenario["agri_growth"] * agri_var) ** years_ahead
            future_fp = last_fp * (1 + scenario["conservation_effort"] * conserv_var) ** years_ahead

            future_temp = last_temp + years_ahead * 0.02
            future_prec = last_prec  # simplification (on pourrait extrapoler)

            approx_fd_for_pressure = max(last_fd, 1.0)
            future_pressure = (future_pop / approx_fd_for_pressure) * 1_000_000

            # Features FD
            feature_values_fd = []
            for feat in features_fd:
                if feat == "Année":
                    val = target_year
                elif feat == "Population":
                    val = future_pop
                elif feat == "Cultures Annuelles (CA)":
                    val = future_agri
                elif feat == "Forêts Plantation (FP)":
                    val = future_fp
                elif feat == "Pression_Anthropique":
                    val = future_pressure
                elif feat == "Temperature":
                    val = future_temp
                elif feat == "Precipitation":
                    val = future_prec
                else:
                    val = df[feat].iloc[-1] if feat in df.columns else 0.0
                feature_values_fd.append(val)

            X_future_fd = np.array([feature_values_fd])
            future_fd_base = float(model_fd.predict(X_future_fd)[0])

            conservation_impact = scenario["conservation_effort"] * last_fd * years_ahead / 5.0
            tech_impact = scenario["tech_improvement"] * tech_var * last_fd * years_ahead / 20.0
            climate_impact = scenario["climate_impact"] * last_fd * years_ahead

            future_fd_adj = future_fd_base + conservation_impact + tech_impact + climate_impact
            future_fd_adj = max(future_fd_adj, 0.0)

            future_pressure_for_co2 = (future_pop / max(future_fd_adj, 1.0)) * 1_000_000

            # Features CO2
            feature_values_co2 = []
            for feat in features_co2:
                if feat == "Année":
                    val = target_year
                elif feat == "Forêts Denses (FD)":
                    val = future_fd_adj
                elif feat == "Forêts Plantation (FP)":
                    val = future_fp
                elif feat == "Temperature":
                    val = future_temp
                else:
                    val = df[feat].iloc[-1] if feat in df.columns else 0.0
                feature_values_co2.append(val)

            X_future_co2 = np.array([feature_values_co2])
            future_co2 = float(model_co2.predict(X_future_co2)[0])

            results.append(
                {
                    "scenario": scenario_name,
                    "population": future_pop,
                    "agriculture": future_agri,
                    "forest_plantation": future_fp,
                    "forest_dense": future_fd_adj,
                    "co2_sequestration": future_co2,
                    "year": target_year,
                }
            )

        return pd.DataFrame(results)


# =============================================================================
# 8. MODULES CLIMAT : DIAGNOSTICS & MULTI-LOCALITÉS
# =============================================================================

def climate_diagnostics(df_enriched, locality_name=None):
    st.subheader("🌦 Diagnostics climatiques")

    col1, col2 = st.columns(2)
    with col1:
        fig_temp = px.line(
            df_enriched,
            x="Année",
            y="Temperature",
            title=f"Température moyenne annuelle{f' - {locality_name}' if locality_name else ''}",
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        fig_prec = px.line(
            df_enriched,
            x="Année",
            y="Precipitation",
            title=f"Précipitations annuelles{f' - {locality_name}' if locality_name else ''}",
        )
        st.plotly_chart(fig_prec, use_container_width=True)

    st.markdown("**Anomalies climatiques (référence 2000–2010)**")
    ref_mask = (df_enriched["Année"] >= 2000) & (df_enriched["Année"] <= 2010)

    if ref_mask.any():
        ref_temp = df_enriched.loc[ref_mask, "Temperature"].mean()
        ref_prec = df_enriched.loc[ref_mask, "Precipitation"].mean()

        df_anom = df_enriched.copy()
        df_anom["Temp_Anom"] = df_anom["Temperature"] - ref_temp
        df_anom["Prec_Anom_%"] = (df_anom["Precipitation"] - ref_prec) / ref_prec * 100.0

        col3, col4 = st.columns(2)
        with col3:
            fig_anom_t = px.bar(
                df_anom,
                x="Année",
                y="Temp_Anom",
                title="Anomalies de température (°C)",
            )
            st.plotly_chart(fig_anom_t, use_container_width=True)

        with col4:
            fig_anom_p = px.bar(
                df_anom,
                x="Année",
                y="Prec_Anom_%",
                title="Anomalies de précipitations (%)",
            )
            st.plotly_chart(fig_anom_p, use_container_width=True)
    else:
        st.info("Période de référence 2000–2010 non disponible pour les anomalies.")


def multi_locality_dashboard(df_climate, df_forest=None, selected_localities=None):
    if not selected_localities:
        st.info("Veuillez sélectionner au moins une localité dans la barre latérale.")
        return

    st.header("🌍 Analyse Multi-Localités")

    dfc = df_climate[df_climate["Localité"].isin(selected_localities)].copy()

    # Température
    st.subheader("🌡️ Température moyenne annuelle – comparaison")
    fig_temp = px.line(
        dfc,
        x="Année",
        y="Temperature",
        color="Localité",
        line_group="Localité",
        markers=True,
        hover_data=["Source_Climat", "Scenario_Climat"],
        title="Température moyenne annuelle par localité",
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # Précipitations
    st.subheader("🌧️ Précipitations annuelles – comparaison")
    fig_prec = px.line(
        dfc,
        x="Année",
        y="Precipitation",
        color="Localité",
        line_group="Localité",
        markers=True,
        hover_data=["Source_Climat", "Scenario_Climat"],
        title="Précipitations annuelles par localité",
    )
    st.plotly_chart(fig_prec, use_container_width=True)

    # Anomalies
    st.subheader("📊 Anomalies climatiques par rapport à 2000–2010")
    ref_mask = (dfc["Année"] >= 2000) & (dfc["Année"] <= 2010)

    if ref_mask.any():
        ref_means = (
            dfc[ref_mask]
            .groupby("Localité")[["Temperature", "Precipitation"]]
            .mean()
            .rename(columns={"Temperature": "Temp_Ref", "Precipitation": "Prec_Ref"})
        )

        dfc = dfc.merge(ref_means, on="Localité", how="left")
        dfc["Temp_Anom"] = dfc["Temperature"] - dfc["Temp_Ref"]
        dfc["Prec_Anom_%"] = (dfc["Precipitation"] - dfc["Prec_Ref"]) / dfc["Prec_Ref"] * 100.0

        col1, col2 = st.columns(2)
        with col1:
            fig_anom_t = px.bar(
                dfc,
                x="Année",
                y="Temp_Anom",
                color="Localité",
                barmode="group",
                title="Anomalies de température (°C)",
            )
            st.plotly_chart(fig_anom_t, use_container_width=True)
        with col2:
            fig_anom_p = px.bar(
                dfc,
                x="Année",
                y="Prec_Anom_%",
                color="Localité",
                barmode="group",
                title="Anomalies de précipitations (%)",
            )
            st.plotly_chart(fig_anom_p, use_container_width=True)
    else:
        st.info("Période 2000–2010 non couverte dans les données climat.")

    # Forêts (facultatif si tu as un fichier forestier multi-localités)
    if df_forest is not None and "Localité" in df_forest.columns:
        st.subheader("🌳 Forêts denses – comparaison (si disponible)")
        dff = df_forest[df_forest["Localité"].isin(selected_localities)].copy()
        fig_fd = px.line(
            dff,
            x="Année",
            y="Forêts Denses (FD)",
            color="Localité",
            markers=True,
            title="Forêts denses par localité",
        )
        st.plotly_chart(fig_fd, use_container_width=True)


# =============================================================================
# 9. APPLICATION PRINCIPALE
# =============================================================================

def main():
    st.title("🌍 Plateforme Avancée de Modélisation de la Déforestation")
    st.markdown(
        """
    **Analyse scientifique des dynamiques de déforestation intégrant modélisation avancée, 
    scénarios GIEC et analyse d'incertitude pour une prise de décision éclairée.**
    """
    )

    # Session state
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False
    if "current_models" not in st.session_state:
        st.session_state.current_models = None
    if "scenario_results" not in st.session_state:
        st.session_state.scenario_results = {}

    # Mode : une localité / multi-localités
    st.sidebar.title("Mode d'analyse")
    mode_localites = st.sidebar.radio(
        "Sélection du mode :", ["Une localité", "Multi-localités"], index=0
    )

    # -------------------------------------------------------------------------
    # Données forestières
    # -------------------------------------------------------------------------
    st.sidebar.subheader("🌳 Données forestières")
    forest_file = st.sidebar.file_uploader(
        "Fichier forestier (optionnel, CSV)",
        type=["csv"],
        help="Si non fourni, les données par défaut 2000–2024 seront utilisées.",
    )

    if forest_file is not None:
        df_forest_raw = pd.read_csv(forest_file)
        if "Année" not in df_forest_raw.columns:
            st.warning("Le fichier forestier doit contenir une colonne 'Année'.")
            df_forest = load_default_forest_data()
        else:
            df_forest = df_forest_raw.sort_values("Année").reset_index(drop=True)
    else:
        df_forest = load_default_forest_data()

    # -------------------------------------------------------------------------
    # Données climatiques ONACC
    # -------------------------------------------------------------------------
    st.sidebar.subheader("🌦 Données climatiques (ONACC)")
    climate_file = st.sidebar.file_uploader(
        "Fichier climate_data_onacc.csv (optionnel)",
        type=["csv"],
        help="Utilise le standard ONACC proposé (Année, Localité, Precipitation, Temperature, ...).",
    )

    df_climate = None
    selected_localities = None

    if climate_file is not None:
        df_climate = load_onacc_climate_csv(climate_file)
        all_localities = df_climate["Localité"].unique().tolist()

        if mode_localites == "Multi-localités":
            selected_localities = st.sidebar.multiselect(
                "Localités à analyser",
                options=all_localities,
                default=all_localities[: min(3, len(all_localities))],
            )
        else:
            selected_locality_single = st.sidebar.selectbox(
                "Localité à analyser", options=all_localities
            )
            selected_localities = [selected_locality_single]
    else:
        st.sidebar.info("Aucun fichier climat ONACC chargé → climat simulé.")

    # -------------------------------------------------------------------------
    # Mode Multi-localités : seulement vue comparative climat/forêts
    # -------------------------------------------------------------------------
    if mode_localites == "Multi-localités":
        if df_climate is None:
            st.warning(
                "Le mode multi-localités nécessite un fichier climat ONACC. "
                "Veuillez en charger un dans la barre latérale."
            )
            return

        multi_locality_dashboard(df_climate, df_forest=None, selected_localities=selected_localities)
        return

    # -------------------------------------------------------------------------
    # Mode Une localité : pipeline complet (modélisation, scénarios, etc.)
    # -------------------------------------------------------------------------
    if df_climate is not None:
        locality_name = selected_localities[0]
    else:
        locality_name = None

    df_forest = df_forest.sort_values("Année").reset_index(drop=True)

    with st.spinner("🔄 Enrichissement des données..."):
        df_enriched = enrich_dataset(df_forest, df_climate, locality_name)

    # Configuration de la sidebar (modèles, incertitude, scénarios)
    config = setup_sidebar()

    # Dashboard indicateurs
    create_real_time_metrics(df_enriched)

    # Navigation sections
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Sections",
        [
            "📊 Données et Exploration",
            "🤖 Modélisation Avancée",
            "🔮 Scénarios GIEC avec Incertitudes",
            "📈 Analyse de Sensibilité",
            "📋 Rapport Scientifique",
        ],
    )

    # -------------------------------------------------------------------------
    # Section 1 : Données & Exploration
    # -------------------------------------------------------------------------
    if page == "📊 Données et Exploration":
        st.header("📊 Exploration Avancée des Données")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Dataset Enrichi")
            st.dataframe(df_enriched.style.format("{:,.2f}"), use_container_width=True)

            st.subheader("Statistiques Descriptives")
            st.dataframe(df_enriched.describe(), use_container_width=True)

            csv = df_enriched.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger le dataset complet CSV",
                data=csv,
                file_name="donnees_deforestation_enrichies.csv",
                mime="text/csv",
            )

        with col2:
            st.subheader("Visualisations Multiples")

            indicators = st.multiselect(
                "Sélectionnez les indicateurs à visualiser :",
                options=df_enriched.columns[1:],
                default=[
                    "Forêts Denses (FD)",
                    "Population",
                    "Séquestration CO2",
                    "Cultures Annuelles (CA)",
                ],
            )

            if indicators:
                fig = go.Figure()
                colors = px.colors.qualitative.Set3
                for i, indicator in enumerate(indicators):
                    fig.add_trace(
                        go.Scatter(
                            x=df_enriched["Année"],
                            y=df_enriched[indicator],
                            mode="lines+markers",
                            name=indicator,
                            line=dict(color=colors[i % len(colors)], width=3),
                        )
                    )
                fig.update_layout(
                    title="Évolution des Indicateurs Clés",
                    xaxis_title="Année",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Matrice de corrélation
            create_advanced_correlation_matrix(df_enriched)

            # Diagnostics climat
            if df_climate is not None and locality_name is not None:
                climate_diagnostics(df_enriched, locality_name=locality_name)

    # -------------------------------------------------------------------------
    # Section 2 : Modélisation Avancée
    # -------------------------------------------------------------------------
    elif page == "🤖 Modélisation Avancée":
        st.header("🤖 Modélisation Prédictive Avancée")

        if (not st.session_state.models_trained) or st.button("🔄 Réentraîner les modèles"):
            with st.spinner("Entraînement des modèles en cours..."):
                try:
                    models = train_advanced_models(df_enriched, config)
                    st.session_state.current_models = models
                    st.session_state.models_trained = True
                    st.success("✅ Modèles entraînés avec succès !")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement : {e}")
                    return

        if st.session_state.models_trained:
            models = st.session_state.current_models

            st.subheader("📊 Performance des Modèles")

            if (
                config["model_type"] == "AutoML"
                and any(k.startswith("fd_") for k in models.keys())
            ):
                model_comparison = []
                for key, model_info in models.items():
                    if key.startswith("fd_") and key != "fd_best":
                        model_comparison.append(
                            {
                                "Modèle": key.replace("fd_", ""),
                                "R²": model_info["r2"],
                                "RMSE": model_info["rmse"],
                                "MAE": model_info["mae"],
                                "CV Score": model_info["cv_score"],
                            }
                        )
                if model_comparison:
                    comparison_df = pd.DataFrame(model_comparison)
                    st.dataframe(
                        comparison_df.style.format(
                            {
                                "R²": "{:.4f}",
                                "RMSE": "{:,.0f}",
                                "MAE": "{:,.0f}",
                                "CV Score": "{:.4f}",
                            }
                        ),
                        use_container_width=True,
                    )

            if "fd_best" in models:
                best_model_info = models["fd_best"]
                st.subheader(
                    f"🎯 Meilleur Modèle : {best_model_info.get('name', 'Linear Regression')}"
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R²", f"{best_model_info['r2']:.4f}")
                with col2:
                    st.metric("RMSE", f"{best_model_info['rmse']:,.0f}")
                with col3:
                    st.metric("MAE", f"{best_model_info['mae']:,.0f}")
                with col4:
                    st.metric(
                        "Score CV",
                        f"{(best_model_info.get('cv_score') or 0.0):.4f}",
                    )

                st.subheader("📈 Prédictions avec Intervalles de Confiance")
                fig_fd = plot_predictions_with_uncertainty(
                    df_enriched,
                    best_model_info,
                    "Forêts Denses (FD)",
                    config["include_uncertainty"],
                )
                st.plotly_chart(fig_fd, use_container_width=True)

                if hasattr(best_model_info["model"], "feature_importances_"):
                    st.subheader("📊 Importance des Variables")
                    feature_importance = pd.DataFrame(
                        {
                            "Variable": best_model_info["features"],
                            "Importance": best_model_info["model"].feature_importances_,
                        }
                    ).sort_values("Importance", ascending=True)

                    fig_importance = px.bar(
                        feature_importance,
                        x="Importance",
                        y="Variable",
                        orientation="h",
                        title="Importance Relative des Variables",
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)

    # -------------------------------------------------------------------------
    # Section 3 : Scénarios GIEC avec Incertitudes
    # -------------------------------------------------------------------------
    elif page == "🔮 Scénarios GIEC avec Incertitudes":
        st.header("🔮 Simulation de Scénarios GIEC avec Analyse d'Incertitude")

        if not st.session_state.models_trained:
            st.warning("⚠️ Veuillez d'abord entraîner les modèles dans 'Modélisation Avancée'.")
            return

        models = st.session_state.current_models
        scenario_manager = ScenarioManager()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Configuration des Scénarios")
            target_year = st.slider("Horizon temporel", 2025, 2100, 2050)

            selected_scenarios = st.multiselect(
                "Scénarios GIEC à simuler :",
                options=list(scenario_manager.scenarios.keys()),
                default=[config["default_scenario"]],
            )

            n_simulations = st.slider("Nombre de simulations Monte Carlo", 50, 1000, 100)

            if st.button("🚀 Lancer les Simulations"):
                with st.spinner(
                    f"Simulation de {len(selected_scenarios)} scénarios ({n_simulations} tirages chacun)..."
                ):
                    st.session_state.scenario_results = {}
                    for scenario_name in selected_scenarios:
                        results = scenario_manager.simulate_scenario(
                            scenario_name, models, df_enriched, target_year, n_simulations
                        )
                        st.session_state.scenario_results[scenario_name] = results
                    st.success("✅ Simulations terminées !")

        with col2:
            st.subheader("Résultats des Simulations")

            if not st.session_state.scenario_results:
                st.info("Veuillez lancer les simulations pour voir les résultats.")
            else:
                summary_data = []
                for scenario_name, results in st.session_state.scenario_results.items():
                    fd_mean = results["forest_dense"].mean()
                    fd_std = results["forest_dense"].std()
                    co2_mean = results["co2_sequestration"].mean()

                    current_fd = df_enriched["Forêts Denses (FD)"].iloc[-1]
                    current_co2 = df_enriched["Séquestration CO2"].iloc[-1]

                    fd_change_pct = (fd_mean - current_fd) / current_fd * 100.0
                    co2_change_pct = (co2_mean - current_co2) / current_co2 * 100.0

                    summary_data.append(
                        {
                            "Scénario": scenario_name,
                            "Forêts (Moy)": f"{fd_mean:,.0f}",
                            "± Incertitude": f"±{fd_std:,.0f}",
                            "Δ Forêts (%)": f"{fd_change_pct:+.1f}%",
                            "Δ CO2 (%)": f"{co2_change_pct:+.1f}%",
                        }
                    )

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

        if st.session_state.scenario_results:
            st.subheader("📊 Comparaison Visuelle des Scénarios")
            fig_comparison = go.Figure()
            colors = px.colors.qualitative.Bold

            for i, (scenario_name, results) in enumerate(
                st.session_state.scenario_results.items()
            ):
                fig_comparison.add_trace(
                    go.Box(
                        y=results["forest_dense"],
                        name=scenario_name,
                        marker_color=colors[i % len(colors)],
                        boxpoints="outliers",
                    )
                )

            fig_comparison.update_layout(
                title="Distribution des Projections de Forêts Denses par Scénario",
                yaxis_title="Forêts Denses (Ha)",
                height=500,
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

            st.subheader("🕐 Évolution Temporelle des Scénarios")
            selected_scenario = st.selectbox(
                "Scénario pour l'évolution détaillée :",
                options=list(st.session_state.scenario_results.keys()),
            )

            if selected_scenario:
                years_proj = list(range(df_enriched["Année"].iloc[-1], target_year + 1, 5))
                fd_proj = []
                fd_min = []
                fd_max = []

                for year in years_proj:
                    results = scenario_manager.simulate_scenario(
                        selected_scenario, models, df_enriched, year, 50
                    )
                    fd_proj.append(results["forest_dense"].mean())
                    fd_min.append(results["forest_dense"].quantile(0.05))
                    fd_max.append(results["forest_dense"].quantile(0.95))

                fig_evolution = go.Figure()
                fig_evolution.add_trace(
                    go.Scatter(
                        x=years_proj + years_proj[::-1],
                        y=fd_max + fd_min[::-1],
                        fill="toself",
                        fillcolor="rgba(0,100,80,0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="Intervalle de confiance 90%",
                    )
                )
                fig_evolution.add_trace(
                    go.Scatter(
                        x=years_proj,
                        y=fd_proj,
                        line=dict(color="rgb(0,100,80)", width=3),
                        mode="lines+markers",
                        name="Projection moyenne",
                    )
                )
                fig_evolution.add_trace(
                    go.Scatter(
                        x=df_enriched["Année"],
                        y=df_enriched["Forêts Denses (FD)"],
                        line=dict(color="red", width=2),
                        mode="lines+markers",
                        name="Historique",
                    )
                )
                fig_evolution.update_layout(
                    title=f"Évolution des Forêts Denses - {selected_scenario}",
                    xaxis_title="Année",
                    yaxis_title="Forêts Denses (Ha)",
                    height=500,
                )
                st.plotly_chart(fig_evolution, use_container_width=True)

    # -------------------------------------------------------------------------
    # Section 4 : Analyse de Sensibilité
    # -------------------------------------------------------------------------
    elif page == "📈 Analyse de Sensibilité":
        st.header("📈 Analyse de Sensibilité Globale")

        if not st.session_state.models_trained:
            st.warning("⚠️ Veuillez d'abord entraîner les modèles.")
            return

        models = st.session_state.current_models
        if "fd_best" not in models:
            st.warning("Aucun modèle 'fd_best' disponible.")
            return

        model_info = models["fd_best"]
        model = model_info["model"]
        features = model_info["features"]

        st.subheader("🎯 Analyse de l'Impact des Variables d'Entrée")

        base_values = {}
        for feature in features:
            if feature in df_enriched.columns:
                base_values[feature] = df_enriched[feature].iloc[-1]
            else:
                base_values[feature] = 0.0

        X_base = np.array([list(base_values.values())])
        base_prediction = model.predict(X_base)[0]

        sensitivity_results = {}
        perturbations = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]

        for feature in features:
            changes = []
            for pert in perturbations:
                perturbed_values = base_values.copy()
                perturbed_values[feature] *= (1 + pert)

                X_pert = np.array([list(perturbed_values.values())])
                try:
                    prediction = model.predict(X_pert)[0]
                    change_pct = (prediction - base_prediction) / base_prediction * 100.0
                    changes.append(
                        {
                            "Perturbation": pert * 100.0,
                            "Prédiction": prediction,
                            "Changement %": change_pct,
                        }
                    )
                except Exception:
                    continue

            if changes:
                sensitivity_results[feature] = pd.DataFrame(changes)

        if sensitivity_results:
            fig_sensitivity = go.Figure()
            colors = px.colors.qualitative.Set3
            for i, (feature, data) in enumerate(sensitivity_results.items()):
                fig_sensitivity.add_trace(
                    go.Scatter(
                        x=data["Perturbation"],
                        y=data["Changement %"],
                        mode="lines+markers",
                        name=feature,
                        line=dict(color=colors[i % len(colors)], width=3),
                    )
                )
            fig_sensitivity.update_layout(
                title="Analyse de Sensibilité - Impact sur les Forêts Denses",
                xaxis_title="Perturbation des Variables d'Entrée (%)",
                yaxis_title="Changement dans la Prédiction (%)",
                height=500,
            )
            st.plotly_chart(fig_sensitivity, use_container_width=True)

            st.subheader("📋 Sensibilité par Variable")
            sensitivity_summary = []
            for feature, data in sensitivity_results.items():
                max_effect = data["Changement %"].abs().max()
                sensitivity_summary.append(
                    {
                        "Variable": feature,
                        "Impact Max (%)": f"{max_effect:.2f}%",
                        "Sensibilité": "Élevée"
                        if max_effect > 5
                        else "Modérée"
                        if max_effect > 2
                        else "Faible",
                    }
                )
            sensitivity_df = pd.DataFrame(sensitivity_summary)
            st.dataframe(sensitivity_df, use_container_width=True)
        else:
            st.info("Impossible de calculer la sensibilité avec les paramètres actuels.")

    # -------------------------------------------------------------------------
    # Section 5 : Rapport Scientifique
    # -------------------------------------------------------------------------
    else:
        st.header("📋 Rapport Scientifique Complet")

        st.subheader("🎯 Résumé Exécutif")
        col1, col2 = st.columns(2)

        total_deforestation = (
            df_enriched["Forêts Denses (FD)"].iloc[0] - df_enriched["Forêts Denses (FD)"].iloc[-1]
        )
        annual_rate = total_deforestation / (
            df_enriched["Année"].iloc[-1] - df_enriched["Année"].iloc[0]
        )

        with col1:
            st.markdown(
                """
            **Objectifs de la Recherche :**
            - ✅ Analyse multidimensionnelle des dynamiques de déforestation
            - ✅ Modélisation avancée avec validation rigoureuse
            - ✅ Intégration des scénarios GIEC SSP-RCP
            - ✅ Quantification des incertitudes et analyse de sensibilité
            - ✅ Recommandations politiques fondées sur des preuves
            """
            )
            st.metric("Déforestation totale 2000-2024", f"{total_deforestation:,.0f} Ha")
            st.metric("Taux annuel moyen", f"{annual_rate:,.0f} Ha/an")

        with col2:
            st.markdown(
                """
            **Méthodologie Avancée :**
            - 🔬 Interpolation temporelle et enrichissement des données
            - 🤖 Modélisation par ensemble (Random Forest, XGBoost, etc.)
            - 📊 Validation croisée temporelle
            - 🎲 Analyse Monte Carlo pour les incertitudes
            - 📈 Analyse de sensibilité globale
            """
            )
            if st.session_state.models_trained and "fd_best" in st.session_state.current_models:
                best_model = st.session_state.current_models["fd_best"]
                st.metric("Performance du meilleur modèle (R²)", f"{best_model['r2']:.4f}")

        st.subheader("🎯 Recommandations Stratégiques")

        tab1, tab2, tab3 = st.tabs(
            ["🎯 Court Terme (2024-2030)", "📈 Moyen Terme (2031-2040)", "🌳 Long Terme (2041-2050)"]
        )
        with tab1:
            st.markdown(
                """
            **Actions Prioritaires Immédiates :**
            - 🛑 Moratoire ciblé sur la conversion des forêts primaires
            - 🌾 Intensification durable de l'agriculture existante
            - 📊 Système de monitoring en temps réel
            - 💰 Paiements pour services écosystémiques
            - 📚 Programmes d'éducation environnementale
            - 🔄 Diversification des revenus ruraux
            """
            )
        with tab2:
            st.markdown(
                """
            **Stratégies de Transition 2031-2040 :**
            - 🌿 Restauration écologique des zones dégradées
            - 🏙️ Plan d'urbanisation maîtrisé
            - 🔋 Transition énergétique vers les renouvelables
            - 🤝 Coopération régionale pour la gestion des bassins versants
            - 📈 Économie verte créatrice d'emplois
            - 🔬 Innovation technologique agricole et forestière
            """
            )
        with tab3:
            st.markdown(
                """
            **Vision Durable 2041-2050 :**
            - 🌍 Économie décarbonée et circulaire
            - 🏞️ Connectivité écologique restaurée
            - 👥 Gouvernance participative institutionnalisée
            - 🔄 Résilience climatique intégrée aux politiques
            - 💡 Innovation sociale et entrepreneuriat vert
            - 📊 Comptabilité environnementale généralisée
            """
            )

        st.subheader("📄 Export du Rapport Complet")
        if st.button("📊 Générer le Rapport Détaillé"):
            report_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary_metrics": {
                    "total_deforestation": float(total_deforestation),
                    "annual_rate": float(annual_rate),
                    "population_growth": float(
                        df_enriched["Population"].iloc[-1] - df_enriched["Population"].iloc[0]
                    ),
                    "agricultural_expansion": float(
                        df_enriched["Cultures Annuelles (CA)"].iloc[-1]
                        - df_enriched["Cultures Annuelles (CA)"].iloc[0]
                    ),
                },
            }
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                label="📥 Télécharger les données du rapport (JSON)",
                data=report_json,
                file_name=f"rapport_deforestation_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )
            st.success("✅ Rapport généré (structure JSON).")


# =============================================================================
# EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    main()
