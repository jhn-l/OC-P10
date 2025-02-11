import os
import pickle
import requests
import zipfile
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit

# 📌 Paramètres
DATA_URL = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+9+-+R%C3%A9alisez+une+application+mobile+de+recommandation+de+contenu/news-portal-user-interactions-by-globocom.zip"
ZIP_FILE = "/tmp/news-portal.zip"
EXTRACTED_FOLDER = "/tmp/"
MODEL_PATH = "/tmp/recommender_model_implicit.pkl"  # Sauvegarde locale du modèle
DATA_FILES = ["clicks_sample.csv"]  # Liste des fichiers nécessaires

# ✅ Télécharger le fichier ZIP
def download_zip_file():
    if not os.path.exists(ZIP_FILE):
        print(f"🔹 Téléchargement de {DATA_URL}...")
        response = requests.get(DATA_URL, stream=True)
        with open(ZIP_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Téléchargement terminé !")
    else:
        print("✅ Fichier ZIP déjà présent, téléchargement ignoré.")

# ✅ Extraire les fichiers du ZIP
def extract_zip_file():
    if not os.path.exists(EXTRACTED_FOLDER):
        os.makedirs(EXTRACTED_FOLDER, exist_ok=True)
    if not all(os.path.exists(os.path.join(EXTRACTED_FOLDER, file)) for file in DATA_FILES):
        print(f"🔹 Décompression de {ZIP_FILE}...")
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(EXTRACTED_FOLDER)
        print("✅ Décompression terminée !")
    else:
        print("✅ Fichiers déjà extraits, extraction ignorée.")

# ✅ Charger les interactions utilisateur-article
def load_interactions():
    if not all(os.path.exists(os.path.join(EXTRACTED_FOLDER, file)) for file in DATA_FILES):
        raise FileNotFoundError("❌ Les fichiers de données ne sont pas disponibles dans /tmp/")
    
    print("🔹 Chargement des interactions utilisateur-article...")
    df_list = [pd.read_csv(os.path.join(EXTRACTED_FOLDER, file)) for file in DATA_FILES]
    interactions_df = pd.concat(df_list, ignore_index=True)
    interactions_df.rename(columns={"click_article_id": "article_id"}, inplace=True)
    interactions_df["article_id"] = interactions_df["article_id"].astype(int)
    interactions_df["user_id"] = interactions_df["user_id"].astype(int)
    print(f"✅ Interactions chargées - Nombre de lignes: {interactions_df.shape[0]}")
    return interactions_df

# ✅ Construire la matrice utilisateur-article sous format sparse
def build_user_item_matrix(interactions_df):
    print("🔹 Construction de la matrice utilisateur-article en format sparse...")
    user_ids = interactions_df["user_id"].astype("category")
    item_ids = interactions_df["article_id"].astype("category")

    user_item_sparse = sparse.coo_matrix(
        (np.ones(len(interactions_df)), 
         (user_ids.cat.codes, item_ids.cat.codes))
    )
    print(f"✅ Matrice utilisateur-article créée : {user_item_sparse.shape[0]} utilisateurs, {user_item_sparse.shape[1]} articles.")

    return user_item_sparse.tocsr()

# ✅ Entraîner le modèle Implicit ALS
def train_implicit_model(user_item_matrix):
    print("🔹 Entraînement du modèle ALS...")
    model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15)
    model.fit(user_item_matrix)
    print("🚀 Modèle ALS entraîné avec succès !")
    return model

# ✅ Sauvegarde du modèle en local
def save_model(model, model_path):
    print(f"📤 Sauvegarde du modèle dans {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("✅ Modèle sauvegardé localement !")

# ✅ Exécution principale
if __name__ == "__main__":
    print("🚀 Début de l'entraînement du modèle...")
    
    download_zip_file()
    extract_zip_file()
    interactions_df = load_interactions()
    user_item_matrix = build_user_item_matrix(interactions_df)
    model = train_implicit_model(user_item_matrix)

    # ✅ Sauvegarde du modèle en local
    save_model(model, MODEL_PATH)

    print("🎯 Fin de l'entraînement et sauvegarde du modèle !")
