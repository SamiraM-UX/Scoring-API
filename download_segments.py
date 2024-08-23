import os
import urllib.request

# URL de base du fichier dans la release GitHub
base_url = "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_"

# Répertoire cible pour enregistrer les segments
save_dir = "/home/scoring/Scoring-API/saved_segments/"

# Vérifiez si le répertoire existe, sinon créez-le
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Télécharger chaque segment
for i in range(10):
    file_name = f"df_train_smote_part_{i}.joblib"
    file_url = base_url + str(i) + ".joblib"
    file_path = os.path.join(save_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Téléchargement du {file_name} depuis {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
            print(f"{file_name} téléchargé et enregistré avec succès.")
        except urllib.error.HTTPError as e:
            print(f"Échec du téléchargement de {file_name}. HTTP Error: {e.code} - {e.reason}")
    else:
        print(f"{file_name} existe déjà.")

