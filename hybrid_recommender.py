import os
import json
import boto3
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import implicit

# 📌 Définir le dossier temporaire pour les modèles et embeddings
os.environ["SURPRISE_DATASET_DIR"] = "/tmp"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 📌 Configuration AWS S3 et DynamoDB
S3_BUCKET_NAME = "my-recommender-dataset"
DYNAMODB_TABLE_NAME = "UserRecommendations"
S3_MODEL_KEY = "recommender_model_hybrid.pkl"
MODEL_LOCAL_PATH = "/tmp/recommender_model_hybrid.pkl"

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

# 📌 Charger un fichier depuis S3 en DataFrame
def load_csv_from_s3(file_key):
    print(f"🔹 Téléchargement de {file_key} depuis S3...")
    obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
    df = pd.read_csv(obj["Body"])
    print(f"✅ Fichier {file_key} chargé avec succès.")
    return df

# 📌 Charger les interactions utilisateur-article
def load_interactions():
    print("🔹 Chargement des interactions utilisateur-article...")
    all_files = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="clicks/").get('Contents', [])
    df_list = [load_csv_from_s3(file["Key"]) for file in all_files if file["Key"].endswith(".csv")]
    interactions_df = pd.concat(df_list, ignore_index=True)
    interactions_df.rename(columns={"click_article_id": "article_id"}, inplace=True)
    interactions_df["article_id"] = interactions_df["article_id"].astype(int)
    print(f"✅ Interactions chargées ({interactions_df.shape[0]} lignes)")
    return interactions_df

# 📌 Charger les embeddings des articles
def load_articles_embeddings():
    print("🔹 Chargement des embeddings des articles...")
    obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key="articles_embeddings.pickle")
    embeddings = pickle.loads(obj["Body"].read())
    print(f"✅ Embeddings chargés ({len(embeddings)} articles)")
    return embeddings

# 📌 Télécharger et charger le modèle depuis S3
def load_model_from_s3():
    print("🔹 Téléchargement du modèle depuis S3...")
    if not os.path.exists(MODEL_LOCAL_PATH):
        obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=S3_MODEL_KEY)
        model_data = obj["Body"].read()
        with open(MODEL_LOCAL_PATH, "wb") as f:
            f.write(model_data)
        print("✅ Modèle téléchargé et sauvegardé localement.")

    print("🔹 Chargement du modèle en mémoire...")
    with open(MODEL_LOCAL_PATH, "rb") as f:
        model, user_mapping, item_mapping = pickle.load(f)
    print("✅ Modèle chargé avec succès.")
    return model, user_mapping, item_mapping

# 📌 Générer des recommandations hybrides
def hybrid_recommendation(user_id, interactions_df, embeddings, model_data, top_n=5, alpha=0.5):
    print(f"🔹 Génération des recommandations pour l'utilisateur {user_id}...")
    model, user_mapping, item_mapping = model_data

    if user_id not in user_mapping:
        print("⚠️ Utilisateur inconnu, recommandations uniquement basées sur le contenu.")
        alpha = 1
        known_articles = []
    else:
        user_idx = user_mapping[user_id]
        recommended_items = model.recommend(user_idx, None, N=top_n)
        known_articles = [item_mapping[i] for i, _ in recommended_items]
    
    all_articles = list(range(len(embeddings)))
    unknown_articles = np.setdiff1d(all_articles, known_articles)

    # 🔹 Filtrage basé sur le contenu
    content_scores = {}
    if known_articles:
        last_article_id = known_articles[-1]
        if last_article_id in embeddings:
            last_article_vector = embeddings[last_article_id].reshape(1, -1)
            article_vectors = np.array([embeddings[a] for a in all_articles])
            similarities = cosine_similarity(last_article_vector, article_vectors)[0]
            content_scores = {all_articles[i]: similarities[i] for i in range(len(all_articles))}
    
    content_values = np.array(list(content_scores.values())).reshape(-1, 1)
    content_values = MinMaxScaler().fit_transform(content_values).flatten()
    content_scores = {article: score for article, score in zip(content_scores.keys(), content_values)}
    
    # 🔹 Fusion des recommandations CF et CBF
    num_cf = int(alpha * top_n)
    num_cb = top_n - num_cf
    
    top_cf = known_articles[:num_cf]
    top_cb = sorted(content_scores, key=content_scores.get, reverse=True)[:num_cb]
    recommended_articles = list(dict.fromkeys(top_cf + top_cb))[:top_n]

    return recommended_articles

# 🚀 Charger les données UNE SEULE FOIS au démarrage de la Lambda
try:
    interactions_df = load_interactions()
    embeddings = load_articles_embeddings()
    model_data = load_model_from_s3()
except Exception as e:
    print(f"❌ Erreur lors du chargement initial : {str(e)}")

# 📌 Fonction Lambda
def lambda_handler(event, context):
    print("🚀 Exécution de la Lambda...")

    user_id = event.get("user_id")
    if not user_id:
        return {"statusCode": 400, "body": json.dumps({"error": "user_id is required"})}

    recommendations = hybrid_recommendation(user_id, interactions_df, embeddings, model_data)

    return {
        "statusCode": 200,
        "body": json.dumps({"user_id": user_id, "recommendations": recommendations})
    }
