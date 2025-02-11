import os
import zipfile
import pickle
import requests
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit

# 📌 Chemins des fichiers
DATA_URL = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+9+-+R%C3%A9alisez+une+application+mobile+de+recommandation+de+contenu/news-portal-user-interactions-by-globocom.zip"
ZIP_FILE = "/tmp/news-portal.zip"
EXTRACTED_FOLDER = "/tmp/"
MODEL_PATH = "/tmp/recommender_model_implicit.pkl"

# ✅ Télécharger le fichier ZIP si nécessaire
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

# ✅ Extraire `clicks.zip` si nécessaire
def extract_clicks_zip():
    zip_path = "/tmp/clicks.zip"
    if not os.path.exists(EXTRACTED_FOLDER):
        os.makedirs(EXTRACTED_FOLDER, exist_ok=True)
    
    if not os.listdir(EXTRACTED_FOLDER):
        print(f"🔹 Décompression de {zip_path} dans {EXTRACTED_FOLDER}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(EXTRACTED_FOLDER)
        print("✅ Décompression terminée !")
    else:
        print("✅ Les fichiers de clicks sont déjà extraits, extraction ignorée.")

# ✅ Charger les interactions utilisateur-article
def load_interactions():
    extract_clicks_zip()
    print("🔹 Chargement des interactions utilisateur-article...")
    all_files = [os.path.join(EXTRACTED_FOLDER, f) for f in os.listdir(EXTRACTED_FOLDER) if f.endswith(".csv")]
    df_list = [pd.read_csv(f) for f in all_files]
    interactions_df = pd.concat(df_list, ignore_index=True)
    interactions_df.rename(columns={"click_article_id": "article_id"}, inplace=True)
    interactions_df["article_id"] = interactions_df["article_id"].astype(int)
    interactions_df["user_id"] = interactions_df["user_id"].astype(int)
    print(f"✅ Interactions chargées - Nombre de lignes: {interactions_df.shape[0]}")
    return interactions_df

# ✅ Construire la matrice utilisateur-article en sparse
def build_user_item_matrix(interactions_df):
    print("🔹 Construction de la matrice utilisateur-article en format sparse...")
    user_ids = interactions_df["user_id"].astype("category")
    item_ids = interactions_df["article_id"].astype("category")
    user_item_sparse = sparse.coo_matrix(
        (np.ones(len(interactions_df)), (user_ids.cat.codes, item_ids.cat.codes))
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

# ✅ Sauvegarder le modèle
def save_model(model, model_path):
    print(f"📤 Sauvegarde du modèle dans {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("✅ Modèle sauvegardé localement !")

# ✅ Exécution principale
if __name__ == "__main__":
    print("🚀 Début de l'entraînement du modèle...")
    download_zip_file()  # 📥 Télécharger les fichiers si nécessaire
    extract_clicks_zip()  # 📂 Décompresser les fichiers
    interactions_df = load_interactions()  # 📊 Charger les interactions
    user_item_matrix = build_user_item_matrix(interactions_df)  # 🔄 Construire la matrice utilisateur-article
    model = train_implicit_model(user_item_matrix)  # 🎯 Entraîner le modèle
    save_model(model, MODEL_PATH)  # 💾 Sauvegarde du modèle
    print("🎯 Fin de l'entraînement et sauvegarde du modèle !")
