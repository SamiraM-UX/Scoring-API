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

# Charger le modèle en dehors de la clause if __name__ == "__main__"
model_path = os.path.join(current_directory, "saved_model", "best_lgbmb.joblib")
model = joblib.load(model_path)

# Charger les 100 premières lignes du DataFrame en dehors de la clause if __name__ == "__main__"
# Charger le DataFrame à partir du fichier CSV ou Joblib, selon ce qui est nécessaire
csv_filename = os.path.join(current_directory, 'df_train_corrected_100rows.csv')

if os.path.exists(csv_filename):
    df_train_smote = pd.read_csv(csv_filename)
else:
    joblib_filename = os.path.join(current_directory, 'df_train_smote_corrected_100rows.joblib')
    df_train_smote = joblib.load(joblib_filename)

@app.route("/")
def home():
    return "API de prédiction avec Flask"

@app.route("/predict", methods=['GET'])
def predict():
    try:
        sk_id_curr = int(request.args.get("SK_ID_CURR"))
    except (TypeError, ValueError):
        return jsonify({"error": "SK_ID_CURR est manquant ou invalide dans la requête"}), 400

    # Rechercher l'échantillon correspondant à SK_ID_CURR dans df_train_smote
    sample = df_train_smote[df_train_smote['SK_ID_CURR'] == sk_id_curr]

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

    # Si shap_values est une liste avec un seul élément, utiliser le premier élément
    if isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values = shap_values[0]

    # Limiter les données renvoyées pour la lisibilité (par exemple, les 10 premières features)
    num_features_to_show = 10 
    limited_shap_values = shap_values[0][:num_features_to_show].tolist()
    limited_feature_names = sample_for_prediction.columns[:num_features_to_show].tolist()
    limited_feature_values = sample_for_prediction.values[0][:num_features_to_show].tolist()

    return jsonify({
        'probability': round(proba * 100, 2),  # Probabilité arrondie à deux décimales
        'shap_values': limited_shap_values,    # Valeurs SHAP limitées
        'feature_names': limited_feature_names,  # Noms des features limités
        'feature_values': limited_feature_values  # Valeurs des features limitées
    })

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=int(port))
