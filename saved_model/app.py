import streamlit as st
import joblib
import pandas as pd

st.title("Prédiction avec votre modèle")

# Charger le modèle
model_path = 'saved_model/best_lgbmb_model.joblib'  # Remplacez par le chemin correct
model = joblib.load(model_path)

# Interface utilisateur pour entrer des données
sk_id_curr = st.text_input("Entrez SK_ID_CURR")

if st.button("Prédire"):
    # Charger vos données et faire une prédiction
    data_path = 'saved_model/df_train_smote_corrected_100rows_with_id.joblib'  # Remplacez par le chemin correct
    df = joblib.load(data_path)
    sample = df[df['SK_ID_CURR'] == int(sk_id_curr)]
    
    if sample.empty:
        st.write(f"Aucun échantillon trouvé pour SK_ID_CURR: {sk_id_curr}")
    else:
        prediction = model.predict(sample)
        st.write(f"Prédiction : {prediction}")

