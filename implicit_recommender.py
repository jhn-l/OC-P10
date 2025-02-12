import os
import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import implicit
import boto3
import subprocess

# 📌 Paramètres DynamoDB
DYNAMODB_TABLE_NAME = "UserRecommendations"
dynamodb = boto3.client("dynamodb")

class RecommenderSystem:
    def __init__(self, model_path, data_folder):
        self.model_path = model_path
        self.data_folder = data_folder
        self.data_files = self.get_data_files()
        self.model = self.load_model()
        self.interactions_df = self.load_interactions()
        self.user_item_matrix, self.user_ids, self.item_ids = self.build_user_item_matrix()

    def get_data_files(self):
        if not os.path.exists(self.data_folder):
            output = subprocess.run(['ls', '-la', '/var/task'], capture_output=True, text=True)
            print(f"📂 Contenu du répertoire /var/task :\n{output.stdout}")
            raise FileNotFoundError(f"❌ Le dossier {self.data_folder} n'existe pas.")
        return [f for f in os.listdir(self.data_folder) if f.endswith(".csv")]

    def check_files_exist(self):
        if not self.data_files:
            raise FileNotFoundError("❌ Aucun fichier CSV trouvé dans /var/task/")

    def load_interactions(self):
        self.check_files_exist()
        print("🔹 Chargement des interactions utilisateur-article...")
        df_list = [pd.read_csv(os.path.join(self.data_folder, file)) for file in self.data_files]
        interactions_df = pd.concat(df_list, ignore_index=True)

        # ✅ Vérifier que `click_timestamp` est bien disponible
        if "click_timestamp" not in interactions_df.columns:
            raise KeyError("❌ La colonne 'click_timestamp' est introuvable dans les données !")

        interactions_df.rename(columns={"click_article_id": "article_id"}, inplace=True)
        interactions_df["article_id"] = interactions_df["article_id"].astype(int)
        interactions_df["user_id"] = interactions_df["user_id"].astype(int)
        interactions_df["click_timestamp"] = interactions_df["click_timestamp"].astype(int)  # S'assurer que c'est bien un entier

        # 📌 Donner plus de poids au dernier article visité
        interactions_df["weight"] = 1  # Poids normal
        interactions_df.loc[interactions_df.groupby("user_id")["click_timestamp"].idxmax(), "weight"] = 5  # Booster le dernier article
        
        print(f"✅ Interactions chargées - Nombre de lignes: {interactions_df.shape[0]}")
        return interactions_df

    def build_user_item_matrix(self):
        print("🔹 Construction de la matrice utilisateur-article en format sparse...")
        user_ids = self.interactions_df["user_id"].astype("category")
        item_ids = self.interactions_df["article_id"].astype("category")

        user_item_sparse = sparse.coo_matrix(
            (self.interactions_df["weight"], 
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

        recommendations = self.model.recommend(user_index, self.user_item_matrix[user_index], N=5)
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
    data_folder="/var/task/clicks"
)

# 📌 Fonction Lambda
import json

# 📌 Fonction Lambda
import json

import json

def lambda_handler(event, context):
    print("🚀 Exécution de la Lambda...")
    print(f"🚀 Événement reçu par Lambda : {json.dumps(event)}")

    try:
        user_id = event.get("user_id")
        if not user_id:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "user_id is required"})
            }

        user_id = int(user_id)
        print(f"🔍 Génération des recommandations pour user_id : {user_id}")

        # ✅ Générer les recommandations ALS pour l'utilisateur
        recommendations = recommender.recommend_articles(user_id)

        # ✅ Vérifier si `recommend_articles()` a retourné une erreur
        if isinstance(recommendations, dict):
            return {
                "statusCode": 400,
                "body": json.dumps(recommendations)
            }

        # ✅ Stocker les recommandations dans DynamoDB
        recommender.store_recommendations_in_dynamodb(user_id, recommendations)

        response = {
            "statusCode": 200,
            "body": json.dumps({
                "user_id": user_id,
                "recommendations": recommendations
            }),
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }

        print(f"🚀 Réponse envoyée à API Gateway : {json.dumps(response)}")
        return response

    except Exception as e:
        print(f"❌ Erreur inattendue : {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal Server Error", "details": str(e)}),
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }


