import os
import sys
import joblib
import pandas as pd
import pytest
from flask import Flask, jsonify, request

# Définir le répertoire courant
current_directory = os.path.dirname(os.path.abspath(__file__))

# Importer l'application Flask depuis main.py
from main import app
print("Importation réussie")

# Créer un client de test pour l'application Flask
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Tester le chargement du modèle
def test_model_loading():
    model_path = os.path.join(current_directory, "saved_model", "best_lgbmb.joblib")
    model = joblib.load(model_path)
    assert model is not None, "Erreur dans le chargement du modèle."

# Tester le chargement du fichier CSV
def test_csv_loading():
    csv_path = os.path.join(current_directory, "Simulations", "Data", "df_train.csv")
    df = pd.read_csv(csv_path)
    assert not df.empty, "Erreur dans le chargement du CSV."

# Tester la fonction de prédiction de l'API
def test_prediction(client):
    # Charger le modèle ici
    model_path = os.path.join(current_directory, "saved_model", "best_lgbmb.joblib")
    model = joblib.load(model_path)

    csv_path = os.path.join(current_directory, "Simulations", "Data", "df_train.csv")
    df = pd.read_csv(csv_path)
    sk_id_curr = df.iloc[0]['SK_ID_CURR']

    response = client.post('/predict', json={'SK_ID_CURR': sk_id_curr})
    data = response.get_json()
    prediction = data['probability']

    assert prediction is not None, "Erreur dans la prédiction."
