import os
import pickle
import boto3
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit

# 📌 Paramètres
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "my-recommender-dataset")  # Nom du bucket S3
S3_DATA_PREFIX = "clicks/"  # Dossier des fichiers dans S3
LOCAL_DATA_PATH = "/tmp/clicks/"  # Dossier temporaire dans Lambda
MODEL_PATH = "/tmp/recommender_model_hybrid.pkl" # sauvegarde local du modèle


# ✅ Créer un client S3
s3_client = boto3.client("s3")

# ✅ Télécharger les fichiers depuis S3
def download_data_from_s3():
    print(f"📥 Téléchargement des fichiers depuis S3: s3://{S3_BUCKET}/{S3_DATA_PREFIX} ...")

    # Assurer que le dossier local existe
    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

    # Lister les fichiers S3
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_DATA_PREFIX)
    
    if "Contents" not in response:
        print("❌ Aucun fichier trouvé sur S3.")
        return []

    files_downloaded = []
    for obj in response["Contents"]:
        file_key = obj["Key"]
        local_file_path = os.path.join(LOCAL_DATA_PATH, os.path.basename(file_key))
        
        if file_key.endswith(".csv"):  # On télécharge uniquement les fichiers CSV
            print(f"📥 Téléchargement: {file_key} -> {local_file_path}")
            s3_client.download_file(S3_BUCKET, file_key, local_file_path)
            files_downloaded.append(local_file_path)

    print(f"✅ {len(files_downloaded)} fichiers téléchargés depuis S3.")
    return files_downloaded

# ✅ Charger les interactions utilisateur-article
def load_interactions():
    files = download_data_from_s3()
    if not files:
        raise Exception("❌ Impossible de charger les données : aucun fichier trouvé.")

    print("🔹 Chargement des interactions utilisateur-article...")
    df_list = [pd.read_csv(f) for f in files]
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

# ✅ Sauvegarde du modèle en local pour Docker Lambda
def save_model(model, model_path):
    print(f"📤 Sauvegarde du modèle dans {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("✅ Modèle sauvegardé localement !")

# ✅ Exécution principale
if __name__ == "__main__":
    print("🚀 Début de l'entraînement du modèle...")
    
    interactions_df = load_interactions()
    user_item_matrix = build_user_item_matrix(interactions_df)
    model = train_implicit_model(user_item_matrix)

    # ✅ Sauvegarde du modèle en local pour Docker Lambda
    save_model(model, MODEL_PATH)

    print("🎯 Fin de l'entraînement et sauvegarde du modèle !")
