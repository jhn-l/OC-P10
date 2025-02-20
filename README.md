# Projet P10 - API de Recommandation

## Dépôt GitHub

Retrouvez le code source du projet sur GitHub : [OC-P10 Repository](https://github.com/jhn-l/OC-P10/)

## Fonctionnement Général du Projet

Le projet repose sur un système de recommandation basé sur le modèle ALS d'`implicit`.

### Exploration des Méthodes de Recommandation

Dans cette étude, deux approches de recommandation ont été explorées :

1. **Filtrage collaboratif implicite avec ALS (Alternating Least Squares)** : utilisé pour recommander des articles en fonction des interactions des utilisateurs.
2. **Méthode alternative basée sur des heuristiques ou d'autres modèles** (expliqué dans `etude-des-modeles.ipynb`).

### Architecture retenue

- **Backend Flask** : Gestion de l'API REST et interface web.
- **Modèle ALS** : Entraînement et recommandation des articles.
- **DynamoDB & AWS Lambda** : Stockage des recommandations et exécution serverless.
- **Pipeline CI/CD avec GitHub Actions** : Déploiement et mise à jour automatique de l'infrastructure.

### Principaux Fichiers

- `app.py` : Serveur Flask pour fournir l'interface utilisateur et récupérer les recommandations via AWS Lambda.
- `implicit_recommender.py` : Chargement des interactions et exécution du modèle de recommandation ALS.
- `train_implicit.py` : Entraînement du modèle de recommandation à partir des données utilisateur.
- `DockerfileAWS` : Fichier Docker pour empaqueter et déployer le modèle sur AWS Lambda.
- `download_and_upload.yml` : Workflow GitHub Actions pour automatiser le déploiement.
- `index.html` : Interface utilisateur permettant aux utilisateurs de tester les recommandations.
- `etude-des-modeles.ipynb` : Analyse et comparaison des différentes approches de recommandation.

## Endpoint de l'API REST

L'API de recommandation est accessible via le endpoint suivant :

```
https://qtf8d0dzlk.execute-api.eu-north-1.amazonaws.com/dev/recommend
```

## Tester l'API

Vous pouvez tester le fonctionnement de l'API en utilisant la commande `curl` suivante :

```bash
curl -X POST "https://qtf8d0dzlk.execute-api.eu-north-1.amazonaws.com/dev/recommend"      -H "Content-Type: application/json"      -d '{"user_id": 115523}'
```

### Réponse attendue

Si la requête est correcte, l'API retourne une réponse au format JSON contenant les recommandations pour l'utilisateur :

```json
{
  "statusCode": 200,
  "body": "{"user_id": 115523, "recommendations": [234267, 271262, 225010, 234269, 160142]}",
  "headers": {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*"
  }
}
```

## Remarque

- Assurez-vous d'avoir une connexion internet active pour effectuer la requête.
- Vous pouvez utiliser des outils comme [Postman](https://www.postman.com/) pour tester l'API de manière plus conviviale.
- Pour toute question ou problème, consultez le dépôt GitHub du projet.
