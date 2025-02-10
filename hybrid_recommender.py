import os
import json
import pickle
import boto3
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sparse

# 📌 Chemins et paramètres
MODEL_PATH = "/var/task/recommender_model_hybrid.pkl"  # 📥 Modèle ALS stocké dans Docker Lambda
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "my-recommender-dataset")  # 📂 Nom du bucket S3
S3_DATA_PREFIX = "clicks/"  # 📂 Chemin des fichiers sur S3
LOCAL_DATA_PATH = "/tmp/clicks/"  # 📂 Dossier temporaire Lambda

# ✅ Charger le modèle ALS
print("🔹 Chargement du modèle ALS...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modèle non trouvé: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("✅ Modèle ALS chargé avec succès !")

# ✅ Client S3 pour télécharger les fichiers
s3_client = boto3.client("s3")

# 📌 Télécharger les fichiers d'interactions depuis S3
def download_data_from_s3():
    print(f"📥 Téléchargement des fichiers depuis S3: s3://{S3_BUCKET}/{S3_DATA_PREFIX} ...")
    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_DATA_PREFIX)
    if "Contents" not in response:
        print("❌ Aucun fichier trouvé sur S3.")
        return []

    files_downloaded = []
    for obj in response["Contents"]:
        file_key = obj["Key"]
        local_file_path = os.path.join(LOCAL_DATA_PATH, os.path.basename(file_key))
        
        if file_key.endswith(".csv"):
            print(f"📥 Téléchargement: {file_key} -> {local_file_path}")
            s3_client.download_file(S3_BUCKET, file_key, local_file_path)
            files_downloaded.append(local_file_path)

    print(f"✅ {len(files_downloaded)} fichiers téléchargés depuis S3.")
    return files_downloaded

# 📌 Charger les interactions utilisateur-article
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

# 📌 Construire la matrice utilisateur-article sous format sparse
def build_user_item_matrix(interactions_df):
    print("🔹 Construction de la matrice utilisateur-article en format sparse...")
    user_ids = interactions_df["user_id"].astype("category")
    item_ids = interactions_df["article_id"].astype("category")

    user_item_sparse = sparse.coo_matrix(
        (np.ones(len(interactions_df)), 
         (user_ids.cat.codes, item_ids.cat.codes))
    )
    print(f"✅ Matrice utilisateur-article créée : {user_item_sparse.shape[0]} utilisateurs, {user_item_sparse.shape[1]} articles.")

    return user_item_sparse.tocsr(), user_ids, item_ids

# 📌 Recommander des articles avec ALS
def recommend_articles_als(user_id, model, user_item_matrix, user_ids, item_ids, top_n=5):
    # ✅ Vérifier si l'utilisateur existe dans la liste des utilisateurs
    if user_id not in user_ids.to_numpy():
        return {"statusCode": 404, "body": json.dumps({"error": f"Utilisateur {user_id} inconnu"})}

    # ✅ Trouver l’index correct de l’utilisateur dans la matrice utilisateur-article
    user_index = user_ids[user_ids == user_id].index[0]  # Trouver l’index dans user_ids
    user_index = user_ids.cat.codes[user_index]  # Convertir en index numérique

    # ✅ Vérifier que cet index est bien dans la matrice utilisateur-article
    if user_index >= user_item_matrix.shape[0]:
        return {"statusCode": 404, "body": json.dumps({"error": f"Utilisateur {user_id} hors de la plage d'indexation"})}

    # ✅ Récupérer les interactions de l'utilisateur
    user_items = user_item_matrix[user_index]

    # ✅ Générer les recommandations avec ALS
    recommendations = model.recommend(user_index, user_items, N=top_n)

    # ✅ Convertir les indices des articles en `article_id`
    recommended_articles = [item_ids.cat.categories[i] for i in recommendations[0]]

    return recommended_articles

# ✅ Charger les données utilisateur-article au démarrage
print("🔹 Chargement des données utilisateur/article...")
interactions_df = load_interactions()
user_item_matrix, user_ids, item_ids = build_user_item_matrix(interactions_df)

# 📌 Fonction Lambda
def lambda_handler(event, context):
    print("🚀 Exécution de la Lambda...")

    user_id = event.get("user_id")
    if not user_id:
        return {"statusCode": 400, "body": json.dumps({"error": "user_id is required"})}

    # ✅ Convertir user_id en entier pour éviter les problèmes de type
    try:
        user_id = int(user_id)
    except ValueError:
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid user_id format"})}

    # ✅ Optimisation : Convertir user_ids une seule fois en set (accélère la recherche)
    user_ids_set = set(user_ids.to_numpy(dtype=int))

    if user_id not in user_ids_set:
        return {"statusCode": 404, "body": json.dumps({"error": f"Utilisateur {user_id} inconnu"})}

    # ✅ Générer les recommandations ALS pour l'utilisateur
    recommendations = recommend_articles_als(user_id, model, user_item_matrix, user_ids, item_ids)

    return {
        "statusCode": 200,
        "body": json.dumps({"user_id": user_id, "recommendations": recommendations})
    }

