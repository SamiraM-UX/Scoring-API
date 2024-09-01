import os
import joblib
import pandas as pd
import streamlit as st

# Titre de l'application
st.title("Prédiction avec votre modèle")

# Définir le chemin correct du modèle par rapport à la structure du dépôt
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "best_lgbmb_model.joblib")
st.write(f"Chargement du modèle depuis {model_path}")

try:
    model = joblib.load(model_path)
    st.write("Modèle chargé avec succès")
except FileNotFoundError:
    st.error(f"Le fichier {model_path} n'a pas été trouvé.")
    st.stop()

# Interface utilisateur pour entrer des données
sk_id_curr = st.text_input("Entrez SK_ID_CURR")

if st.button("Prédire"):
    # Charger vos données et faire une prédiction
    df_path = os.path.join(current_dir, "df_train_smote_corrected_100rows_with_id.joblib")
    st.write(f"Chargement du DataFrame depuis {df_path}")
    
    try:
        df = joblib.load(df_path)
        st.write("DataFrame chargé avec succès")
    except FileNotFoundError:
        st.error(f"Le fichier {df_path} n'a pas été trouvé.")
        st.stop()
    
    # Renommer les colonnes pour qu'elles correspondent aux colonnes attendues par le modèle
    column_mapping = {i: f'Column_{i}' for i in range(len(df.columns))}
    df.rename(columns=column_mapping, inplace=True)
    
    # Afficher un tableau avec un aperçu des colonnes renommées
    st.write("Aperçu des colonnes après renommage :")
    st.dataframe(df.iloc[:, :10])  # Affiche les 10 premières colonnes sous forme de tableau

    try:
        sample = df[df['SK_ID_CURR'] == int(sk_id_curr)]
        
        if sample.empty:
            st.write(f"Aucun échantillon trouvé pour SK_ID_CURR: {sk_id_curr}")
        else:
            # Liste des colonnes attendues par le modèle
            model_columns = model.booster_.feature_name()
            
            # Afficher les premières colonnes pour le diagnostic
            st.write("Colonnes du DataFrame :")
            st.dataframe(sample.iloc[:, :10])  # Affiche les 10 premières colonnes sous forme de tableau
            st.write("Colonnes attendues par le modèle :")
            st.write(model_columns[:10])  # Afficher les 10 premières colonnes attendues uniquement
            
            try:
                # Convertir les colonnes en types numériques
                sample_for_prediction = sample.loc[:, model_columns].apply(pd.to_numeric, errors='coerce')

                # Vérifier si certaines colonnes n'ont pas pu être converties
                if sample_for_prediction.isnull().any().any():
                    st.write("Certaines colonnes contiennent des données non numériques.")
                    st.write(sample_for_prediction.isnull().sum())
                else:
                    # Obtenir la probabilité prédite
                    prediction_proba = model.predict_proba(sample_for_prediction)
                    probability = prediction_proba[0][1]  # Probabilité de la classe positive
                    st.write(f"Probabilité prédite : {probability * 100:.2f}%")
            except KeyError as e:
                st.write(f"Erreur de colonne manquante ou incorrecte : {e}")
            except ValueError as e:
                st.write(f"Erreur de prédiction : {e}")
    except ValueError:
        st.error("Veuillez entrer un ID valide.")

