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

def download_zip_file():
    """Télécharge le fichier ZIP depuis l'URL fournie"""
    print(f"🔹 Téléchargement de {DATA_URL}...")
    response = requests.get(DATA_URL, stream=True)
    with open(ZIP_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Téléchargement terminé !")

def extract_zip_file(zip_path, extract_to):
    """Décompresse un fichier ZIP et affiche son contenu"""
    print(f"🔹 Décompression de {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_files = zip_ref.namelist()  # Liste des fichiers extraits
    print("✅ Décompression terminée !")
    print(f"📂 Contenu extrait :\n{extracted_files}")

def upload_to_s3(file_path, s3_key):
    """Téléverse un fichier vers AWS S3"""
    print(f"🔹 Upload de {file_path} vers S3 ({S3_BUCKET_NAME}/{s3_key})...")
    s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
    print(f"✅ {file_path} uploadé avec succès !")

if __name__ == "__main__":
    # 📌 Étape 1 : Télécharger le fichier ZIP
    download_zip_file()

    # 📌 Étape 2 : Décompresser directement dans le dossier courant
    extract_zip_file(ZIP_FILE, EXTRACTED_FOLDER)

    # 📌 Étape 3 : Vérifier et extraire clicks.zip si présent
    click_zip_file = os.path.join(EXTRACTED_FOLDER, "clicks.zip")
    if os.path.exists(click_zip_file):
        extract_zip_file(click_zip_file, EXTRACTED_FOLDER)
    else:
        print("⚠️ clicks.zip n'a pas été trouvé !")

    # 📌 Étape 4 : Upload des fichiers extraits vers S3
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

    print("🚀 Processus terminé !")
