import os
os.environ["SURPRISE_DATASET_DIR"] = "/tmp"  # 🔹 Définit le répertoire surprise avant l'import

if not os.path.exists("/tmp/surprise_data"):
    os.makedirs("/tmp/surprise_data")

import json
import boto3
import pandas as pd
import numpy as np
import pickle

import surprise
surprise.dataset.get_dataset_dir = lambda: "/tmp/surprise_data"  # 🔹 Définit le répertoire surprise avant toute utilisation

# 🔹 Modifier `builtin_datasets.py` pour éviter les erreurs d'accès
surprise.builtin_datasets.get_dataset_dir = lambda: "/tmp/surprise_data"

from surprise import Dataset, Reader, SVD
Dataset.load_builtin = lambda name: None  # Désactive le téléchargement automatique de datasets

from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler




# 📌 Paramètres AWS S3 et DynamoDB
S3_BUCKET_NAME = "my-recommender-dataset"
DYNAMODB_TABLE_NAME = "UserRecommendations"
s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

# 📌 Chemins des fichiers (adaptés pour AWS Lambda)
CLICKS_PATH = "clicks/"
ARTICLES_METADATA_PATH = "articles_metadata.csv"
EMBEDDINGS_PATH = "articles_embeddings.pickle"
MODEL_PATH = "/tmp/recommender_model_hybrid.pkl"  # Utilisation de /tmp pour AWS Lambda

# 📌 Charger un fichier depuis S3 en DataFrame
def load_csv_from_s3(file_key):
    obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
    return pd.read_csv(obj["Body"])

# 📌 Charger les interactions utilisateur-article
def load_interactions():
    print("🔹 Chargement des interactions utilisateur-article...")
    all_files = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=CLICKS_PATH).get('Contents', [])
    df_list = [load_csv_from_s3(file["Key"]) for file in all_files if file["Key"].endswith(".csv")]
    interactions_df = pd.concat(df_list, ignore_index=True)
    interactions_df.rename(columns={"click_article_id": "article_id"}, inplace=True)
    interactions_df["article_id"] = interactions_df["article_id"].astype(int)
    print("✅ Interactions chargées - Nombre de lignes:", interactions_df.shape[0])
    return interactions_df

# 📌 Charger les embeddings des articles
def load_articles_embeddings():
    print("🔹 Chargement des embeddings des articles...")
    embeddings_data = s3.get_object(Bucket=S3_BUCKET_NAME, Key=EMBEDDINGS_PATH)["Body"].read()
    embeddings = pickle.loads(embeddings_data)
    print("✅ Embeddings chargés - Nombre d'articles:", len(embeddings))
    return embeddings

# 📌 Entraîner un modèle de filtrage collaboratif
def train_collaborative_model(interactions_df):
    print("🔹 Préparation des données pour Surprise...")
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(interactions_df[["user_id", "article_id", "session_size"]], reader)
    trainset = surprise_data.build_full_trainset()
    
    print("🔹 Entraînement du modèle SVD...")
    model = SVD()
    model.fit(trainset)
    
    print(f"✅ Sauvegarde du modèle dans {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    upload_model_to_s3()
    print("🚀 Modèle entraîné et sauvegardé avec succès !")
    return model

def upload_model_to_s3():
    print(f"🚀 Upload du modèle vers S3: {S3_BUCKET_NAME}/recommender_model_hybrid.pkl...")
    s3.upload_file(MODEL_PATH, S3_BUCKET_NAME, "recommender_model_hybrid.pkl")
    print("✅ Modèle uploadé avec succès sur S3 !")

# 📌 Générer des recommandations hybrides
def hybrid_recommendation(user_id, interactions_df, embeddings, model, top_n=5, alpha=0.5):
    known_articles = interactions_df[interactions_df["user_id"] == user_id]["article_id"].unique()
    all_articles = interactions_df["article_id"].unique()
    unknown_articles = np.setdiff1d(all_articles, known_articles)
    
    user_clicks = interactions_df[interactions_df["user_id"] == user_id].sort_values(by="click_timestamp", ascending=False)
    if user_clicks.empty:
        print("⚠️ Aucun historique de clics trouvé pour cet utilisateur. Utilisation du filtrage basé sur le contenu uniquement.")
        alpha = 1  
    
    # Filtrage collaboratif
    cf_scores = {article: model.predict(user_id, article).est for article in unknown_articles}
    if cf_scores:
        cf_values = np.array(list(cf_scores.values())).reshape(-1, 1)
        cf_values = MinMaxScaler().fit_transform(cf_values).flatten()
        cf_scores = {article: score for article, score in zip(cf_scores.keys(), cf_values)}
    
    # Filtrage basé sur le contenu
    last_article_id = user_clicks["article_id"].iloc[0] if not user_clicks.empty else None
    content_scores = {}
    if last_article_id and last_article_id in embeddings:
        last_article_vector = embeddings[last_article_id].reshape(1, -1)
        article_ids = list(embeddings.keys())
        article_vectors = np.array([embeddings[a] for a in article_ids])
        similarities = cosine_similarity(last_article_vector, article_vectors)[0]
        content_scores = {article_ids[i]: similarities[i] for i in range(len(article_ids))}
    
    if content_scores:
        content_values = np.array(list(content_scores.values())).reshape(-1, 1)
        content_values = MinMaxScaler().fit_transform(content_values).flatten()
        content_scores = {article: score for article, score in zip(content_scores.keys(), content_values)}
    
    # Sélection proportionnelle des recommandations CF et CBF
    num_cf = int(alpha * top_n)
    num_cb = top_n - num_cf
    
    top_cf = sorted(cf_scores, key=cf_scores.get, reverse=True)[:num_cf]
    top_cb = sorted(content_scores, key=content_scores.get, reverse=True)[:num_cb]
    
    recommended_articles = list(dict.fromkeys(top_cf + top_cb))[:top_n]
    print(f"✅ Articles recommandés pour l'utilisateur {user_id} (Hybride) : {recommended_articles}")
    
    store_recommendations_in_dynamodb(user_id, recommended_articles)
    return recommended_articles

# 📌 Stocker les recommandations dans DynamoDB
def store_recommendations_in_dynamodb(user_id, recommendations):
    dynamodb.put_item(
        TableName=DYNAMODB_TABLE_NAME,
        Item={
            "user_id": {"S": str(user_id)},
            "recommendations": {"L": [{"N": str(rec)} for rec in recommendations]}
        }
    )
    print(f"✅ Recommandations stockées pour l'utilisateur {user_id} dans DynamoDB.")

# 📌 Fonction Lambda
def lambda_handler(event, context):
    user_id = event.get("user_id")
    if not user_id:
        return {"statusCode": 400, "body": json.dumps({"error": "user_id is required"})}
    
    interactions_df = load_interactions()
    embeddings = load_articles_embeddings()
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    recommendations = hybrid_recommendation(user_id, interactions_df, embeddings, model)
    return {"statusCode": 200, "body": json.dumps({"user_id": user_id, "recommendations": recommendations})}
