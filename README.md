# SPLAT·FORGE Platform — Guide de Déploiement GPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dr-starck66/splatforge-platform/blob/main/SplatForge_Colab.ipynb)

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

Le fichier `app/pipeline.py` est présent et lance le pipeline COLMAP. **Un entraînement 3DGS réel n'est pas inclus dans ce dépôt** et nécessite un script externe installé sur un GPU NVIDIA compatible. Configurer la variable `SPLATFORGE_TRAIN_SCRIPT` avec le chemin de ce script (interface attendue `-s SOURCE -m MODEL_DIR`). Sans script d'entraînement, le moteur échoue explicitement avec `REAL_3DGS_UNAVAILABLE` : **il ne crée plus de faux modèle `.splat` et n'annonce plus de réussite artificielle**. Un script sans modèle produit `REAL_3DGS_OUTPUT_MISSING`. Tests de non-régression : `python -m unittest discover -s tests -p 'test_pipeline_real_output.py' -v` (les tests simulent COLMAP et ne prouvent pas un rendu GPU). Le test GPU bout en bout reste nécessaire avant PASS fonctionnel.

## 🐳 Gestion des conteneurs

- **Voir les logs** : `docker compose logs -f api`
- **Arrêter l'application** : `docker compose down`
- **Vérifier l'état du GPU** : `docker exec -it splatforge-api nvidia-smi`
