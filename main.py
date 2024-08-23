import os
import joblib
import pandas as pd
import shap
from flask import Flask, jsonify, request
import warnings

warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray")

# Initialisation de l'application Flask
app = Flask(__name__)

# Définir le répertoire courant
current_directory = os.path.dirname(os.path.abspath(__file__))

# Charger le modèle en dehors de la clause if __name__ == "__main__":
model_path = os.path.join(current_directory, "saved_model", "best_lgbmb.joblib")
model = joblib.load(model_path)

# Fonction pour trouver l'échantillon dans les segments
def find_sk_id_curr(sk_id_curr):
    for i in range(10):  # Assurez-vous que le nombre de segments correspond à ce que vous avez créé
        part_path = os.path.join(current_directory, "saved_segments", f"df_train_smote_part_{i}.joblib")
        df_part = joblib.load(part_path)
        
        if sk_id_curr in df_part['SK_ID_CURR'].values:
            return df_part[df_part['SK_ID_CURR'] == sk_id_curr]
    
    return pd.DataFrame()  # Retourne un DataFrame vide si SK_ID_CURR n'est trouvé dans aucun segment

@app.route("/")
def home():
    return "API de prédiction avec Flask"

@app.route("/predict", methods=['GET'])
def predict():
    try:
        sk_id_curr = int(request.args.get("SK_ID_CURR"))
    except (TypeError, ValueError):
        return jsonify({"error": "SK_ID_CURR est manquant ou invalide dans la requête"}), 400

    sample = find_sk_id_curr(sk_id_curr)

    if sample.empty:
        return jsonify({"error": f"Aucun échantillon trouvé pour SK_ID_CURR: {sk_id_curr}"}), 404

    # Supprimer la colonne SK_ID_CURR pour la prédiction
    sample_for_prediction = sample.drop(columns=['SK_ID_CURR'])

    # Prédire
    prediction = model.predict_proba(sample_for_prediction)
    proba = prediction[0][1]  # Probabilité de la seconde classe

    # Calculer les valeurs SHAP pour l'échantillon donné
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_for_prediction)

    # Si shap_values est une liste avec un seul élément, utilisez le premier élément
    if isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values = shap_values[0]

    # Limiter les données renvoyées pour la lisibilité (par exemple, les 10 premières features)
    num_features_to_show = 10
    limited_shap_values = shap_values[0][:num_features_to_show].tolist()
    limited_feature_names = sample_for_prediction.columns[:num_features_to_show].tolist()
    limited_feature_values = sample_for_prediction.values[0][:num_features_to_show].tolist()

    return jsonify({
        'probability': round(proba * 100, 2),
        'shap_values': limited_shap_values,
        'feature_names': limited_feature_names,
        'feature_values': limited_feature_values
    })

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=int(port))


