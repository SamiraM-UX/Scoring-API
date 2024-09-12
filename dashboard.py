import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Configuration de la page
st.set_page_config(layout="wide")

# Fonction pour ajuster la taille de la police en fonction de la hauteur
def get_title_font_size(height):
    base_size = 12
    scale_factor = height / 600.0
    return base_size * scale_factor

# Génération des annotations pour les graphiques
def generate_annotations(df, x_anchor):
    annotations = []
    for y_val, x_val, feat_val in zip(df["Feature"], df["SHAP Value"], df["Feature Value"]):
        formatted_feat_val = feat_val if pd.isna(feat_val) else (int(feat_val) if feat_val == int(feat_val) else feat_val)
        annotations.append(
            dict(
                x=x_val,
                y=y_val,
                text=f"<b>{formatted_feat_val}</b>",
                showarrow=False,
                xanchor=x_anchor,
                yanchor="middle",
                font=dict(color="white"),
            )
        )
    return annotations

# Fonction pour générer le graphique des SHAP values
def generate_figure(df, title_text, x_anchor, yaxis_categoryorder, yaxis_side):
    fig = go.Figure(data=[go.Bar(y=df["Feature"], x=df["SHAP Value"], orientation="h")])
    annotations = generate_annotations(df, x_anchor)

    title_font_size = get_title_font_size(600)
    fig.update_layout(
        annotations=annotations,
        title_text=title_text,
        title_x=0.25,
        title_y=0.88,
        title_font=dict(size=title_font_size),
        yaxis=dict(categoryorder=yaxis_categoryorder, side=yaxis_side, tickfont=dict(size=14)),
        height=600,
    )
    fig.update_xaxes(title_text="Impact des fonctionnalités")
    return fig

# Fonction pour déterminer la couleur en fonction de la probabilité
def compute_color(value):
    return "green" if value < 48 else "red"

# Fonction pour formater les valeurs affichées
def format_value(val):
    if pd.isna(val):
        return val
    if isinstance(val, (float, int)):
        return int(val) if val == int(val) else round(val, 2)
    return val

# Fonction pour récupérer les états de session
def get_state():
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "data_received": False,
            "data": None,
            "last_sk_id_curr": None,
        }
    return st.session_state["state"]

state = get_state()

# Titre du tableau de bord
st.markdown(
    "<h1 style='text-align: center; color: black;'>Estimation du risque de non-remboursement</h1>",
    unsafe_allow_html=True,
)

# Entrée pour SK_ID_CURR
sk_id_curr = st.text_input("Entrez le SK_ID_CURR:")

col1, col2 = st.columns([1, 20])

# Bouton pour lancer la prédiction
if col1.button("Run") or state["data_received"]:
    if state["last_sk_id_curr"] != sk_id_curr:
        state["data_received"] = False
        state["last_sk_id_curr"] = sk_id_curr

    if not state["data_received"]:
        # Appel à l'API pour la prédiction
        response = requests.get(f"http://localhost:5000/predict?SK_ID_CURR={sk_id_curr}")
        if response.status_code != 200:
            st.error(f"Erreur lors de l'appel à l'API: {response.status_code}")
            st.stop()

        state["data"] = response.json()
        state["data_received"] = True

    data = state["data"]
    
    # Extraction des données retournées par l'API
    proba = data["probability"]
    feature_names = data["feature_names"]
    shap_values = data["shap_values"]
    feature_values = data["feature_values"]

    shap_df = pd.DataFrame(
        list(
            zip(
                feature_names,
                shap_values,
                [format_value(val) for val in feature_values],
            )
        ),
        columns=["Feature", "SHAP Value", "Feature Value"],
    )

    # Calcul de la couleur en fonction de la probabilité
    color = compute_color(proba)

    col2.markdown(
        f"<p style='margin: 10px;'>La probabilité que ce client ne puisse pas rembourser son crédit est de <span style='color:{color}; font-weight:bold;'>{proba:.2f}%</span> (tolérance max: <strong>48%</strong>)</p>",
        unsafe_allow_html=True,
    )

    decision_message = "Le prêt sera accordé." if proba < 48 else "Le prêt ne sera pas accordé."
    st.markdown(
        f"<div style='text-align: center; color:{color}; font-size:30px; border:2px solid {color}; padding:10px;'>{decision_message}</div>",
        unsafe_allow_html=True,
    )

    # Séparation des SHAP values en positifs et négatifs
    top_positive_shap = shap_df.sort_values(by="SHAP Value", ascending=False).head(10)
    top_negative_shap = shap_df.sort_values(by="SHAP Value").head(10)

    # Génération des graphiques
    fig_positive = generate_figure(
        top_positive_shap,
        "Top 10 des fonctionnalités augmentant le risque de non-remboursement",
        "right",
        "total ascending",
        "left",
    )
    fig_negative = generate_figure(
        top_negative_shap,
        "Top 10 des fonctionnalités réduisant le risque de non-remboursement",
        "left",
        "total descending",
        "right",
    )

    # Affichage des graphiques
    col_chart1, col_chart2 = st.columns(2)
    col_chart1.plotly_chart(fig_positive, use_container_width=True)
    col_chart2.plotly_chart(fig_negative, use_container_width=True)
