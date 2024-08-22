import os
import os
import urllib.request

# URL du fichier dans la release GitHub
file_url = "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.0/df_train_smote_corrected.joblib"
file_path = "saved_model/df_train_smote_corrected.joblib"

# Vérifiez si le fichier existe déjà
if not os.path.exists(file_path):
    print("Téléchargement du fichier manquant...")
    urllib.request.urlretrieve(file_url, file_path)
    print("Téléchargement terminé.")

