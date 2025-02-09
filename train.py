import os
import boto3
import pandas as pd
import pickle

from surprise import Dataset, Reader, SVD
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

Dataset.load_builtin = lambda name: None

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

# 📌 Entraîner un modèle de filtrage collaboratif
def train_collaborative_model(interactions_df):
    print("🔹 Préparation des données pour Surprise...")
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(interactions_df[["user_id", "article_id", "session_size"]], reader)
    trainset = surprise_data.build_full_trainset()
    
    print("🔹 Entraînement du modèle SVD...")
    model = SVD()
    model.fit(trainset)
    
    print(f"✅ Sauvegarde du modèle localement dans {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
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
