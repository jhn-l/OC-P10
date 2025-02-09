import os
import pandas as pd
import numpy as np
import pickle
import boto3
import logging
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# 📌 Chemins des fichiers
S3_BUCKET_NAME = "my-recommender-dataset"  # ⚠️ Remplace par ton vrai nom de bucket
CLICKS_FOLDER = "/tmp/clicks/"  # AWS Lambda ne permet l'écriture que dans /tmp/
ARTICLES_METADATA_PATH = "/tmp/articles_metadata.csv"
EMBEDDINGS_PATH = "/tmp/articles_embeddings.pickle"
MODEL_PATH = "/tmp/recommender_model_hybrid.pkl"
DYNAMODB_TABLE = "UserRecommendations"
dynamodb = boto3.resource("dynamodb")

# 📌 Charger les interactions utilisateur-article
def load_interactions():
    print("🔹 Chargement des interactions utilisateur-article...")
    all_files = [os.path.join(CLICKS_FOLDER, f) for f in os.listdir(CLICKS_FOLDER) if f.startswith("clicks_hour_") and f.endswith(".csv")]
    df_list = [pd.read_csv(f) for f in all_files]
    interactions_df = pd.concat(df_list, ignore_index=True)
    interactions_df.rename(columns={"click_article_id": "article_id"}, inplace=True)
    interactions_df["article_id"] = interactions_df["article_id"].astype(int)
    print("✅ Interactions chargées - Nombre de lignes:", interactions_df.shape[0])
    return interactions_df

# 📌 Entraîner le modèle de filtrage collaboratif
def train_collaborative_model(interactions_df):
    print("🔹 Préparation des données pour Surprise...")
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(interactions_df[["user_id", "article_id", "session_size"]], reader)
    
    print("🔹 Entraînement du modèle SVD...")
    trainset, _ = train_test_split(surprise_data, test_size=0.2)
    model = SVD()
    model.fit(trainset)

    print(f"✅ Sauvegarde du modèle dans {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    upload_model_to_s3()
    print("🚀 Modèle entraîné et sauvegardé avec succès !")
    return model

# 📌 Upload du modèle vers S3
def upload_model_to_s3():
    s3_client = boto3.client("s3")
    model_filename = "recommender_model_hybrid.pkl"
    print(f"🚀 Upload du modèle vers S3: {S3_BUCKET_NAME}/{model_filename}...")
    s3_client.upload_file(MODEL_PATH, S3_BUCKET_NAME, model_filename)
    print("✅ Modèle uploadé avec succès sur S3 !")

# 📌 Charger le modèle depuis S3
def load_model_from_s3():
    s3_client = boto3.client("s3")
    model_filename = "recommender_model_hybrid.pkl"
    local_path = MODEL_PATH
    
    print(f"🔄 Téléchargement du modèle depuis S3...")
    s3_client.download_file(S3_BUCKET_NAME, model_filename, local_path)
    
    with open(local_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Modèle chargé avec succès depuis S3")
    return model

# 📌 Sauvegarder les recommandations dans DynamoDB
def save_recommendations_to_dynamodb(user_id, recommendations):
    print(f"💾 Sauvegarde des recommandations pour l'utilisateur {user_id} dans DynamoDB...")
    table = dynamodb.Table(DYNAMODB_TABLE)
    table.put_item(
        Item={
            "user_id": str(user_id),
            "recommendations": recommendations
        }
    )
    print("✅ Recommandations sauvegardées avec succès dans DynamoDB")

# 📌 Fonction Lambda
def lambda_handler(event, context):
    user_id = event.get("user_id")
    if not user_id:
        return {"statusCode": 400, "body": "{\"error\": \"user_id is required\"}"}
    
    interactions_df = load_interactions()
    model = load_model_from_s3()
    recommendations = train_collaborative_model(interactions_df)
    save_recommendations_to_dynamodb(user_id, recommendations)
    
    return {
        "statusCode": 200,
        "body": "{\"user_id\": " + str(user_id) + ", \"recommendations\": " + str(recommendations) + "}"
    }
