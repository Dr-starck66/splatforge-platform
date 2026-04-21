# SPLAT·FORGE Platform — Guide de Déploiement GPU

Ce projet est une plateforme de reconstruction 3D (3D Gaussian Splatting) utilisant FastAPI, Docker et COLMAP. Pour fonctionner correctement, il nécessite un serveur équipé d'un **GPU NVIDIA**.

## 🚀 Prérequis du Serveur

- **OS** : Ubuntu 22.04 LTS (recommandé)
- **GPU** : NVIDIA (minimum 8GB VRAM recommandé)
- **Pilotes** : NVIDIA Drivers installés
- **Outils** : Docker et Docker Compose

## 🛠️ Installation étape par étape

### 1. Installation Automatisée (Recommandé)

Si votre serveur est vierge, utilisez le script d'installation fourni pour configurer Docker, les pilotes NVIDIA et le NVIDIA Container Toolkit en une seule commande :

```bash
chmod +x install_gpu_runtime.sh
./install_gpu_runtime.sh
```

*Note : Un redémarrage du serveur peut être nécessaire après l'installation des pilotes.*

### 2. Clonage et Lancement

```bash
# Cloner votre dépôt
git clone https://github.com/Dr-starck66/splatforge-platform.git
cd splatforge-platform

# Créer les dossiers de données
mkdir -p data/uploads data/outputs

# Lancer l'application avec Docker Compose
docker compose up -d --build
```

## 🌐 Accès à l'application

Une fois lancé, l'application est accessible sur :
- **Interface Web** : `http://<IP_DE_VOTRE_SERVEUR>:8000`
- **Documentation API** : `http://<IP_DE_VOTRE_SERVEUR>:8000/api/docs`

## ⚠️ Note Importante sur `pipeline.py`

Le fichier `app/pipeline.py` est actuellement manquant dans ce dépôt. Ce fichier est crucial car il contient la logique de traitement COLMAP et 3DGS. Assurez-vous de l'ajouter dans le dossier `app/` pour que le traitement des images fonctionne.

## 🐳 Gestion des conteneurs

- **Voir les logs** : `docker compose logs -f api`
- **Arrêter l'application** : `docker compose down`
- **Vérifier l'état du GPU** : `docker exec -it splatforge-api nvidia-smi`
