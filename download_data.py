import os
import requests

# URL du fichier dans GitHub Releases
url = 'https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.0/df_train_smote_corrected.joblib'

# Chemin où le fichier sera enregistré
save_path = os.path.join('saved_model', 'df_train_smote_corrected.joblib')

# Créez le répertoire si nécessaire
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Télécharger le fichier
response = requests.get(url, stream=True)

# Sauvegarder le fichier
with open(save_path, 'wb') as f:
    f.write(response.content)

print(f"Fichier téléchargé et enregistré à {save_path}")
