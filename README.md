# SPLAT·FORGE Platform — Guide de Déploiement GPU

Ce projet est une plateforme de reconstruction 3D (3D Gaussian Splatting) utilisant FastAPI, Docker et COLMAP. Pour fonctionner correctement, il nécessite un serveur équipé d'un **GPU NVIDIA**.

## 🚀 Prérequis du Serveur

- **OS** : Ubuntu 22.04 LTS (recommandé)
- **GPU** : NVIDIA (minimum 8GB VRAM recommandé)
- **Pilotes** : NVIDIA Drivers installés
- **Outils** : Docker et Docker Compose

## 🛠️ Installation étape par étape

### 1. Installation des pilotes NVIDIA et Docker

Si votre serveur est vierge, exécutez ces commandes pour installer les composants nécessaires :

```bash
# Mise à jour du système
sudo apt update && sudo apt upgrade -y

# Installation de Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Installation du NVIDIA Container Toolkit (indispensable pour le GPU dans Docker)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/nginx/conf.d/nvidia-container-toolkit.list \
  && sudo apt-get update \
  && sudo apt-get install -y nvidia-container-toolkit

# Redémarrer Docker pour appliquer les changements
sudo systemctl restart docker
```

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
