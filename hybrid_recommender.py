import os
import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import implicit
import boto3

# 📌 Paramètres DynamoDB
DYNAMODB_TABLE_NAME = "UserRecommendations"
dynamodb = boto3.client("dynamodb")

class RecommenderSystem:
    def __init__(self, model_path, data_folder, data_files):
        self.model_path = model_path
        self.data_folder = data_folder
        self.data_files = data_files
        self.model = self.load_model()
        self.interactions_df = self.load_interactions()
        self.user_item_matrix, self.user_ids, self.item_ids = self.build_user_item_matrix()

    def check_files_exist(self):
        if not all(os.path.exists(os.path.join(self.data_folder, file)) for file in self.data_files):
            raise FileNotFoundError("❌ Les fichiers de données ne sont pas disponibles dans /tmp/")

    def load_interactions(self):
        self.check_files_exist()
        print("🔹 Chargement des interactions utilisateur-article...")
        df_list = [pd.read_csv(os.path.join(self.data_folder, file)) for file in self.data_files]
        interactions_df = pd.concat(df_list, ignore_index=True)
        interactions_df.rename(columns={"click_article_id": "article_id"}, inplace=True)
        interactions_df["article_id"] = interactions_df["article_id"].astype(int)
        interactions_df["user_id"] = interactions_df["user_id"].astype(int)
        print(f"✅ Interactions chargées - Nombre de lignes: {interactions_df.shape[0]}")
        return interactions_df

    def build_user_item_matrix(self):
        print("🔹 Construction de la matrice utilisateur-article en format sparse...")
        user_ids = self.interactions_df["user_id"].astype("category")
        item_ids = self.interactions_df["article_id"].astype("category")

        user_item_sparse = sparse.coo_matrix(
            (np.ones(len(self.interactions_df)), 
             (user_ids.cat.codes, item_ids.cat.codes))
        )
        print(f"✅ Matrice utilisateur-article créée : {user_item_sparse.shape[0]} utilisateurs, {user_item_sparse.shape[1]} articles.")
        return user_item_sparse.tocsr(), user_ids, item_ids

    def load_model(self):
        print("🔹 Chargement du modèle ALS...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Modèle non trouvé: {self.model_path}")
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Modèle ALS chargé avec succès !")
        return model

    def recommend_articles(self, user_id, top_n=5):
        if user_id not in self.user_ids.cat.categories:
            return {"statusCode": 404, "body": json.dumps({"error": f"Utilisateur {user_id} inconnu"})}

        user_index = self.user_ids[self.user_ids == user_id].index[0]
        user_index = self.user_ids.cat.codes[user_index]

        if user_index >= self.user_item_matrix.shape[0]:
            return {"statusCode": 404, "body": json.dumps({"error": f"Utilisateur {user_id} hors de la plage d'indexation"})}

        if self.user_item_matrix[user_index].nnz == 0:
            return {"statusCode": 404, "body": json.dumps({"error": f"L'utilisateur {user_id} n'a aucune interaction"})}

        recommendations = self.model.recommend(user_index, self.user_item_matrix[user_index], N=top_n)
        recommended_articles = [int(self.item_ids.cat.categories[i]) for i in recommendations[0]]

        return recommended_articles

    def store_recommendations_in_dynamodb(self, user_id, recommendations):
        print(f"🚀 Stockage des recommandations pour {user_id} dans DynamoDB...")
        dynamodb.put_item(
            TableName=DYNAMODB_TABLE_NAME,
            Item={
                "user_id": {"S": str(user_id)},
                "recommendations": {"L": [{"N": str(rec)} for rec in recommendations]}
            }
        )
        print("✅ Recommandations sauvegardées avec succès.")

# Initialisation du système de recommandation
recommender = RecommenderSystem(
    model_path="/var/task/recommender_model_implicit.pkl",
    data_folder="/tmp/clicks/",
    data_files=["clicks_sample.csv"]
)

# 📌 Fonction Lambda
def lambda_handler(event, context):
    print("🚀 Exécution de la Lambda...")

    user_id = event.get("user_id")
    if not user_id:
        return {"statusCode": 400, "body": json.dumps({"error": "user_id is required"})}

    try:
        user_id = int(user_id)
    except ValueError:
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid user_id format"})}

    # ✅ Générer les recommandations ALS pour l'utilisateur
    recommendations = recommender.recommend_articles(user_id)

    # ✅ Vérifier si `recommend_articles()` a retourné une erreur
    if isinstance(recommendations, dict):
        return recommendations  # Retourne directement l'erreur si elle existe

    # ✅ Stocker les recommandations dans DynamoDB
    recommender.store_recommendations_in_dynamodb(user_id, recommendations)

    return {
        "statusCode": 200,
        "body": json.dumps({"user_id": user_id, "recommendations": recommendations})
    }
