import os
import urllib.request

# Liste des URLs correctes pour chaque segment
urls = [
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_0.joblib",
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_1.joblib",
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_2.joblib",
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_3.joblib",
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_4.joblib",
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_5.joblib",
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_6.joblib",
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_7.joblib",
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_8.joblib",
    "https://github.com/SamiraM-UX/Scoring-API/releases/download/v1.1/df_train_smote_part_9.joblib"
]

# Répertoire cible pour enregistrer les segments
save_dir = "/home/scoring/Scoring-API/saved_segments/"

# Vérifiez si le répertoire existe, sinon créez-le
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Télécharger chaque segment
for i, url in enumerate(urls):
    file_name = f"df_train_smote_part_{i}.joblib"
    file_path = os.path.join(save_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Téléchargement du {file_name} depuis {url}...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"{file_name} téléchargé et enregistré avec succès.")
        except urllib.error.HTTPError as e:
            print(f"Échec du téléchargement de {file_name}. HTTP Error: {e.code} - {e.reason}")
    else:
        print(f"{file_name} existe déjà.")
