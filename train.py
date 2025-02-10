import os
import boto3
import pandas as pd
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# 📌 Configurer AWS
S3_BUCKET_NAME = "my-recommender-dataset"
s3 = boto3.client("s3")
MODEL_PATH = "/tmp/recommender_model_hybrid.pkl"
CLICKS_PATH = "clicks/"

# 📌 Charger un fichier depuis S3 en DataFrame
def load_csv_from_s3(file_key):
    obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
    return pd.read_csv(obj["Body"])

# 📌 Charger les interactions utilisateur-article depuis S3
def load_interactions():
    print("🔹 Chargement des interactions utilisateur-article...")
    all_files = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=CLICKS_PATH).get('Contents', [])
    df_list = [load_csv_from_s3(file["Key"]) for file in all_files if file["Key"].endswith(".csv")]
    interactions_df = pd.concat(df_list, ignore_index=True)
    
    interactions_df.rename(columns={"click_article_id": "article_id"}, inplace=True)
    interactions_df["article_id"] = interactions_df["article_id"].astype(int)
    
    print("✅ Interactions chargées - Nombre de lignes:", interactions_df.shape[0])
    return interactions_df

# 📌 Construire une matrice utilisateur-article **sparse**
def build_sparse_matrix(interactions_df):
    print("🔹 Construction de la matrice utilisateur-article éparse...")

    user_ids = interactions_df["user_id"].astype("category")
    article_ids = interactions_df["article_id"].astype("category")

    row = user_ids.cat.codes.values
    col = article_ids.cat.codes.values
    data = np.ones(len(interactions_df))

    sparse_matrix = sp.csr_matrix((data, (row, col)), shape=(user_ids.cat.categories.size, article_ids.cat.categories.size))
    
    print(f"✅ Matrice utilisateur-article construite avec {sparse_matrix.nnz} interactions non nulles.")
    return sparse_matrix, user_ids, article_ids

# 📌 Sélection automatique de `n_components`
def choose_n_components(X, variance_threshold=0.95):
    svd = TruncatedSVD(n_components=min(300, X.shape[1] - 1))
    svd.fit(X)
    explained_variance = np.cumsum(svd.explained_variance_ratio_)

    optimal_n = np.argmax(explained_variance >= variance_threshold) + 1
    print(f"✅ Nombre optimal de composantes SVD : {optimal_n} (Variance expliquée: {explained_variance[optimal_n-1]:.2f})")
    
    return optimal_n

# 📌 Entraîner un modèle de filtrage collaboratif avec TruncatedSVD
def train_collaborative_model(interactions_df):
    print("🔹 Création de la matrice utilisateur-article...")
    sparse_matrix, user_ids, article_ids = build_sparse_matrix(interactions_df)

    # 🔹 Réduction de dimension avec TruncatedSVD
    print("🔹 Entraînement du modèle SVD...")
    n_components = choose_n_components(sparse_matrix)

    svd = TruncatedSVD(n_components=n_components)
    svd.fit(sparse_matrix)

    print(f"✅ Sauvegarde du modèle localement dans {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((svd, user_ids, article_ids), f)  # Stocker les index pour l'inférence

    upload_model_to_s3()
    print("🚀 Modèle entraîné et sauvegardé avec succès sur S3 !")

# 📌 Upload du modèle vers S3
def upload_model_to_s3():
    print(f"🚀 Upload du modèle vers S3: {S3_BUCKET_NAME}/recommender_model_hybrid.pkl...")
    s3.upload_file(MODEL_PATH, S3_BUCKET_NAME, "recommender_model_hybrid.pkl")
    print("✅ Modèle uploadé avec succès sur S3 !")

# 📌 Exécution du script d'entraînement
if __name__ == "__main__":
    interactions_df = load_interactions()
    train_collaborative_model(interactions_df)
