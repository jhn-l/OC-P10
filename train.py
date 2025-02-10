import os
import boto3
import pandas as pd
import pickle
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

# 📌 Configurer AWS
S3_BUCKET_NAME = "my-recommender-dataset"
MODEL_PATH = "/tmp/recommender_model_hybrid.pkl"
CLICKS_PATH = "clicks/"
s3 = boto3.client("s3")

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

# 📌 Transformer les interactions en une matrice creuse pour `implicit`
def transform_to_sparse_matrix(interactions_df):
    user_map = {user: i for i, user in enumerate(interactions_df["user_id"].unique())}
    item_map = {item: i for i, item in enumerate(interactions_df["article_id"].unique())}

    rows = interactions_df["user_id"].map(user_map).values
    cols = interactions_df["article_id"].map(item_map).values
    data = interactions_df["session_size"].values  # Vérifier que cette colonne existe bien

    # ✅ Forcer les types compatibles avec scipy.sparse
    data = data.astype(np.float32)
    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)

    print(f"Data dtype: {data.dtype}, Rows dtype: {rows.dtype}, Cols dtype: {cols.dtype}")  # Debugging

    sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(user_map), len(item_map)))

    return sparse_matrix, user_map, item_map

# 📌 Entraîner un modèle de filtrage collaboratif `implicit`
def train_collaborative_model(interactions_df):
    print("🔹 Transformation des données en format sparse...")
    sparse_matrix, user_map, item_map = transform_to_sparse_matrix(interactions_df)
    
    print("🔹 Entraînement du modèle ALS...")
    model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
    model.fit(sparse_matrix)
    
    print(f"✅ Sauvegarde du modèle localement dans {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, user_map, item_map), f)
    
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
