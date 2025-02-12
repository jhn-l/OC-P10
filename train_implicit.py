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
EXTRACTED_FOLDER = "/tmp/clicks/"
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
    
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"❌ Le fichier ZIP {ZIP_FILE} n'a pas été téléchargé correctement !")

# ✅ Extraire `news-portal.zip` puis `clicks.zip`
def extract_clicks_zip():
    news_zip_path = ZIP_FILE
    clicks_zip_path = "/tmp/clicks.zip"
    
    if not os.path.exists(news_zip_path):
        raise FileNotFoundError(f"❌ Le fichier ZIP {news_zip_path} n'existe pas !")
    
    with zipfile.ZipFile(news_zip_path, "r") as zip_ref:
        zip_ref.extractall("/tmp/")
    
    if not os.path.exists(clicks_zip_path):
        raise FileNotFoundError(f"❌ Le fichier {clicks_zip_path} n'a pas été extrait correctement !")
    
    print(f"🔹 Décompression de {clicks_zip_path} dans {EXTRACTED_FOLDER}...")
    with zipfile.ZipFile(clicks_zip_path, "r") as zip_ref:
        zip_ref.extractall("/tmp/")
    print("✅ Décompression terminée !")
    
    extracted_files = os.listdir(EXTRACTED_FOLDER)
    print(f"📂 Fichiers trouvés après extraction: {extracted_files}")

# ✅ Charger les interactions utilisateur-article
def load_interactions():
    extract_clicks_zip()
    print("🔹 Chargement des interactions utilisateur-article...")
    all_files = [os.path.join(EXTRACTED_FOLDER, f) for f in os.listdir(EXTRACTED_FOLDER) if f.endswith(".csv")]
    
    if not all_files:
        raise FileNotFoundError(f"❌ Aucun fichier CSV trouvé dans {EXTRACTED_FOLDER} ! Contenu : {os.listdir(EXTRACTED_FOLDER)}")
    
    df_list = [pd.read_csv(f) for f in all_files]
    interactions_df = pd.concat(df_list, ignore_index=True)

    # ✅ Vérifier que `click_timestamp` est bien disponible
    if "click_timestamp" not in interactions_df.columns:
        raise KeyError("❌ La colonne 'click_timestamp' est introuvable dans les données !")

    interactions_df.rename(columns={"click_article_id": "article_id"}, inplace=True)
    interactions_df["article_id"] = interactions_df["article_id"].astype(int)
    interactions_df["user_id"] = interactions_df["user_id"].astype(int)
    interactions_df["click_timestamp"] = interactions_df["click_timestamp"].astype(int)  # S'assurer que c'est bien un entier

    # 📌 Donner plus de poids aux derniers articles consultés (exponentiel)
    interactions_df["weight"] = 1  # Poids de base
    
    def assign_weights(df):
        df = df.sort_values(by="click_timestamp", ascending=False)
        weights = [5, 3, 2]  # Pondération décroissante pour les 3 derniers articles
        df.loc[df.index[:len(weights)], "weight"] = weights[:len(df)]
        return df

    interactions_df = interactions_df.groupby("user_id", group_keys=False).apply(assign_weights)
    
    print(f"✅ Interactions chargées - Nombre de lignes: {interactions_df.shape[0]}")
    return interactions_df


# ✅ Construire la matrice utilisateur-article en sparse
def build_user_item_matrix(interactions_df):
    print("🔹 Construction de la matrice utilisateur-article en format sparse...")
    user_ids = interactions_df["user_id"].astype("category")
    item_ids = interactions_df["article_id"].astype("category")
    user_item_sparse = sparse.coo_matrix(
        (interactions_df["weight"], (user_ids.cat.codes, item_ids.cat.codes))
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
    download_zip_file()
    extract_clicks_zip()
    interactions_df = load_interactions()
    user_item_matrix = build_user_item_matrix(interactions_df)
    model = train_implicit_model(user_item_matrix)
    save_model(model, MODEL_PATH)
    print("🎯 Fin de l'entraînement et sauvegarde du modèle !")
