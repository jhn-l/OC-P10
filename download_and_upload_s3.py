import os
import requests
import zipfile
import boto3
import json
import shutil

def get_dynamodb_client():
    return boto3.client('dynamodb', region_name=os.getenv("AWS_REGION", "eu-north-1"))

def create_dynamodb_table():
    dynamodb = get_dynamodb_client()
    try:
        dynamodb.create_table(
            TableName='UserRecommendations',
            KeySchema=[{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': 'user_id', 'AttributeType': 'S'}],
            BillingMode='PAY_PER_REQUEST'
        )
        print("✅ Table UserRecommendations créée avec succès !")
    except dynamodb.exceptions.ResourceInUseException:
        print("✅ Table UserRecommendations existe déjà.")

def insert_sample_data():
    dynamodb = get_dynamodb_client()
    sample_data = [
        {"user_id": {"S": "115523"}, "recommendations": {"L": [{"N": "149738"}, {"N": "103086"}, {"N": "156457"}, {"N": "103074"}, {"N": "103137"}]}},
        {"user_id": {"S": "10234"}, "recommendations": {"L": [{"N": "394"}, {"N": "3144"}, {"N": "3145"}, {"N": "3232"}, {"N": "3434"}]}}
    ]
    for item in sample_data:
        dynamodb.put_item(TableName='UserRecommendations', Item=item)
    print("✅ Données de test insérées avec succès !")

def create_lambda_function():
    lambda_client = boto3.client('lambda', region_name=os.getenv("AWS_REGION", "eu-north-1"))
    role_arn = os.getenv("AWS_LAMBDA_ROLE_ARN")
    
    lambda_code = '''
import json
import boto3

def lambda_handler(event, context):
    dynamodb = boto3.client('dynamodb')
    user_id = event.get('user_id')
    
    if not user_id:
        return {"statusCode": 400, "body": "user_id is required"}

    response = dynamodb.get_item(TableName='UserRecommendations', Key={'user_id': {'S': user_id}})
    
    if 'Item' not in response:
        return {"statusCode": 404, "body": json.dumps({"error": "User not found"})}

    # Extraction propre des recommandations
    recommendations = [int(item['N']) for item in response['Item']['recommendations']['L']]

    return {
        "statusCode": 200,
        "body": json.dumps({
            "user_id": user_id,
            "recommendations": recommendations
        })
    }
'''
    
    os.makedirs("lambda_code", exist_ok=True)
    with open("lambda_code/lambda_function.py", "w") as f:
        f.write(lambda_code)
    
    shutil.make_archive("lambda_package", 'zip', "lambda_code")
    
    with open("lambda_package.zip", "rb") as f:
        zip_data = f.read()
    
    try:
        lambda_client.create_function(
            FunctionName='GetUserRecommendations',
            Runtime='python3.8',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_data},
            Timeout=15,
            MemorySize=128
        )
        print('✅ AWS Lambda déployée avec succès.')
    except lambda_client.exceptions.ResourceConflictException:
        print('✅ AWS Lambda existe déjà.')

def deploy_project():
    create_dynamodb_table()
    insert_sample_data()
    create_lambda_function()

deploy_project()

# 📌 Ajout des processus de téléchargement et upload des fichiers
DATA_URL = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+9+-+R%C3%A9alisez+une+application+mobile+de+recommandation+de+contenu/news-portal-user-interactions-by-globocom.zip"
ZIP_FILE = "news-portal.zip"
EXTRACTED_FOLDER = "."
S3_BUCKET_NAME = "my-recommender-dataset"
s3_client = boto3.client('s3')

def file_exists_in_s3(s3_key):
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return True
    except:
        return False

def download_zip_file():
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
    if not os.path.exists(os.path.join(extract_to, "articles_metadata.csv")):
        print(f"🔹 Décompression de {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Décompression terminée !")
    else:
        print("✅ Fichiers déjà extraits, extraction ignorée.")

def upload_to_s3(file_path, s3_key):
    if not file_exists_in_s3(s3_key):
        print(f"🔹 Upload de {file_path} vers S3 ({S3_BUCKET_NAME}/{s3_key})...")
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        print(f"✅ {file_path} uploadé avec succès !")
    else:
        print(f"✅ {file_path} déjà présent sur S3, upload ignoré.")

download_zip_file()
extract_zip_file(ZIP_FILE, EXTRACTED_FOLDER)
click_zip_file = os.path.join(EXTRACTED_FOLDER, "clicks.zip")
if os.path.exists(click_zip_file):
    extract_zip_file(click_zip_file, EXTRACTED_FOLDER)
else:
    print("⚠️ clicks.zip n'a pas été trouvé !")
files_to_upload = ["articles_metadata.csv", "articles_embeddings.pickle", "clicks_sample.csv"]
for file_path in files_to_upload:
    if os.path.exists(file_path):
        upload_to_s3(file_path, file_path)
    else:
        print(f"⚠️ Fichier introuvable : {file_path}")
clicks_folder = "clicks"
if os.path.exists(clicks_folder):
    for file in os.listdir(clicks_folder):
        file_path = os.path.join(clicks_folder, file)
        if os.path.isfile(file_path):
            upload_to_s3(file_path, f"clicks/{file}")
else:
    print("⚠️ Aucun dossier 'clicks/' trouvé, aucun fichier supplémentaire à uploader.")
print("🚀 Processus terminé !")
