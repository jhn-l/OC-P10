import os
import json
import boto3
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

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
MODEL_LOCAL_PATH = "/tmp/recommender_model_hybrid.pkl"  # Utilisation de /tmp pour AWS Lambda

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

# 📌 Télécharger le modèle depuis S3
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
        model = pickle.load(f)
    print("✅ Modèle chargé avec succès.")
    return model

# 📌 Générer des recommandations hybrides
def hybrid_recommendation(user_id, interactions_df, embeddings, model, top_n=5, alpha=0.5):
    print(f"🔹 Génération des recommandations pour l'utilisateur {user_id}...")

    known_articles = interactions_df[interactions_df["user_id"] == user_id]["article_id"].unique()
    all_articles = interactions_df["article_id"].unique()
    unknown_articles = np.setdiff1d(all_articles, known_articles)

    user_clicks = interactions_df[interactions_df["user_id"] == user_id].sort_values(by="click_timestamp", ascending=False)
    if user_clicks.empty:
        print("⚠️ Aucun historique trouvé, utilisation uniquement du filtrage basé sur le contenu.")
        alpha = 1  

    # 🔹 Filtrage collaboratif avec TruncatedSVD
    cf_scores = {}
    if len(unknown_articles) > 0:
        transformed_data = model.transform(interactions_df.pivot_table(index="user_id", columns="article_id", values="session_size", fill_value=0).values)
        for article in unknown_articles:
            cf_scores[article] = transformed_data[user_id][article]

    if cf_scores:
        cf_values = np.array(list(cf_scores.values())).reshape(-1, 1)
        cf_values = MinMaxScaler().fit_transform(cf_values).flatten()
        cf_scores = {article: score for article, score in zip(cf_scores.keys(), cf_values)}

    # 🔹 Filtrage basé sur le contenu
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

    # 🔹 Fusion des recommandations CF et CBF
    num_cf = int(alpha * top_n)
    num_cb = top_n - num_cf
    
    top_cf = sorted(cf_scores, key=cf_scores.get, reverse=True)[:num_cf]
    top_cb = sorted(content_scores, key=content_scores.get, reverse=True)[:num_cb]

    recommended_articles = list(dict.fromkeys(top_cf + top_cb))[:top_n]
    print(f"✅ Articles recommandés pour l'utilisateur {user_id} : {recommended_articles}")

    store_recommendations_in_dynamodb(user_id, recommended_articles)
    return recommended_articles

# 📌 Stocker les recommandations dans DynamoDB
def store_recommendations_in_dynamodb(user_id, recommendations):
    print(f"🚀 Stockage des recommandations pour {user_id} dans DynamoDB...")
    dynamodb.put_item(
        TableName=DYNAMODB_TABLE_NAME,
        Item={
            "user_id": {"S": str(user_id)},
            "recommendations": {"L": [{"N": str(rec)} for rec in recommendations]}
        }
    )
    print("✅ Recommandations sauvegardées avec succès.")
    
# 🚀 Charger les données UNE SEULE FOIS au démarrage de la Lambda
try:
    interactions_df = load_interactions()
    embeddings = load_articles_embeddings()
    model = load_model_from_s3()
except Exception as e:
    print(f"❌ Erreur lors du chargement initial : {str(e)}")
    

# 📌 Fonction Lambda
def lambda_handler(event, context):
    print("🚀 Exécution de la Lambda...")

    user_id = event.get("user_id")
    if not user_id:
        return {"statusCode": 400, "body": json.dumps({"error": "user_id is required"})}

    # interactions_df = load_interactions()
    # embeddings = load_articles_embeddings()
    # model = load_model_from_s3()

    recommendations = hybrid_recommendation(user_id, interactions_df, embeddings, model)

    return {
        "statusCode": 200,
        "body": json.dumps({"user_id": user_id, "recommendations": recommendations})
    }
