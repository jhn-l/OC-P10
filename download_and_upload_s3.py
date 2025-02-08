import os
import requests
import zipfile
import boto3

# 📌 Configuration des variables
DATA_URL = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+9+-+R%C3%A9alisez+une+application+mobile+de+recommandation+de+contenu/news-portal-user-interactions-by-globocom.zip"
ZIP_FILE = "news-portal.zip"
EXTRACTED_FOLDER = "."  # Extraction dans le répertoire courant
S3_BUCKET_NAME = "my-recommender-dataset"

# 📌 Configuration AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")

# 📌 Initialisation du client S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

def file_exists_in_s3(s3_key):
    """Vérifie si un fichier existe déjà sur S3."""
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return True
    except:
        return False

def download_zip_file():
    """Télécharge le fichier ZIP depuis l'URL fournie si non présent"""
    if not os.path.exists(ZIP_FILE):
        print(f"🔹 Téléchargement de {DATA_URL}...")
        response = requests.get(DATA_URL, stream=True)
        with open(ZIP_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Téléchargement terminé !")
    else:
        print("✅ Fichier ZIP déjà présent, téléchargement ignoré.")

def extract_zip_file(zip_path, extract_to):
    """Décompresse un fichier ZIP si non extrait"""
    if not os.path.exists(extract_to + "/articles_metadata.csv"):
        print(f"🔹 Décompression de {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Décompression terminée !")
    else:
        print("✅ Fichiers déjà extraits, extraction ignorée.")

def upload_to_s3(file_path, s3_key):
    """Téléverse un fichier vers AWS S3 si non présent"""
    if not file_exists_in_s3(s3_key):
        print(f"🔹 Upload de {file_path} vers S3 ({S3_BUCKET_NAME}/{s3_key})...")
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        print(f"✅ {file_path} uploadé avec succès !")
    else:
        print(f"✅ {file_path} déjà présent sur S3, upload ignoré.")

if __name__ == "__main__":
    # 📌 Étape 1 : Télécharger le fichier ZIP si nécessaire
    download_zip_file()

    # 📌 Étape 2 : Décompresser si nécessaire
    extract_zip_file(ZIP_FILE, EXTRACTED_FOLDER)

    # 📌 Étape 3 : Vérifier et extraire clicks.zip si présent
    click_zip_file = os.path.join(EXTRACTED_FOLDER, "clicks.zip")
    if os.path.exists(click_zip_file):
        extract_zip_file(click_zip_file, EXTRACTED_FOLDER)
    else:
        print("⚠️ clicks.zip n'a pas été trouvé !")

    # 📌 Étape 4 : Upload des fichiers principaux vers S3 si nécessaire
    files_to_upload = [
        "articles_metadata.csv",
        "articles_embeddings.pickle",
        "clicks_sample.csv",
    ]

    for file_path in files_to_upload:
        if os.path.exists(file_path):
            upload_to_s3(file_path, file_path)  # Envoie avec le même nom
        else:
            print(f"⚠️ Fichier introuvable : {file_path}")

    # 📌 Étape 5 : Upload des fichiers horaires "clicks/clicks_hour_XXX.csv" si nécessaires
    clicks_folder = "clicks"
    if os.path.exists(clicks_folder):
        for file in os.listdir(clicks_folder):
            file_path = os.path.join(clicks_folder, file)
            if os.path.isfile(file_path):
                upload_to_s3(file_path, f"clicks/{file}")  # Stocker dans un dossier clicks/ sur S3
    else:
        print("⚠️ Aucun dossier 'clicks/' trouvé, aucun fichier supplémentaire à uploader.")

    print("🚀 Processus terminé !")
