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


## ✅ Chemin CPU sans GPU ajouté le 17 octobre 2026

Un moteur expérimental PyTorch **réel** accepte maintenant des photographies accompagnées des poses de caméras et points 3D reconstruits par COLMAP. Il optimise les gaussiennes par descente de gradient et exporte \`3dgs/scene.splat\` (32 octets par gaussienne), directement lisible dans le visualiseur existant. Il ne prétend PAS avoir la même qualité, performance ou fidélité qu'un entraînement CUDA natif gsplat.

Le code d'application \`app/main.py\` a été corrigé pour exécuter le pipeline en tâche de fond côté serveur et sauvegarder les sessions sur disque. Les sessions interrompues sont relancées au redémarrage lorsque les images d'entrée persistent.

### Serveur CPU existant, sans nouvelle infrastructure GPU

\`\`\`bash
docker compose -f docker-compose.cpu.yml up -d --build
\`\`\`

Le conteneur CPU utilise COLMAP avec SIFT CPU, PyTorch CPU, et \`tools/train_colmap_cpu.py\` par défaut. Le Dockerfile NVIDIA précédent reste disponible et accepte \`SPLATFORGE_TRAIN_SCRIPT\` pour un moteur externe compatible. Les images d'entrée et sorties doivent être conservées dans \`./data\` sur un volume persistant. Aucune nouvelle machine ni dépense n'est automatiquement engagée par ce dépôt.

**Validation exécutée localement** (pas sur un site public) : 5 images synthétiques + 5 poses COLMAP-text + 12 points 3D, optimisation réelle 80 itérations, MSE \`0.0073640458 -> 0.0000105124\`, fichier .splat de 384 octets, SHA256 \`71e40a1c1195be06022f232716f098782e6de7e84501f357207e6a2f490fba4d\`, 4 tests unitaires Python réussis. Les données COLMAP sont une fixture synthétique ; le binaire COLMAP, les photos terrain, les tests API et le déploiement conteneur doivent encore être validés séparément pour un PASS global.

Commandes reproductibles :

\`\`\`bash
python -m unittest discover -s tests -p 'test_cpu_colmap_trainer.py' -v
python -m unittest discover -s tests -p 'test_api_background.py' -v
python -m unittest discover -s tests -p 'test_pipeline_real_output.py' -v
\`\`\`

### Limites explicites

Le chemin CPU utilise une approximation de rasterisation gaussienne et un nombre borné de vues et points pour limiter RAM/CPU. Les reconstructions échouent proprement lorsque COLMAP ne retrouve pas assez de caméras/points ou que l'optimisation ne converge pas ; aucun faux succès n'est émis. Le workflow natif MuJoCo/PyBullet/CUDA est disponible dans \`tools/astra_native_engines_smoke.py\`, mais il requiert son propre environnement de test et ne peut pas être présumé PASS.


## ⚠️ Note Importante sur `pipeline.py`

Le chemin CPU 3DGS expérimental est maintenant intégré (`tools/train_colmap_cpu.py`) ; pour un moteur GPU natif haute fidélité, configurer `SPLATFORGE_TRAIN_SCRIPT` et vérifier indépendamment le matériel CUDA. Le script échoue explicitement si l'entraînement ou les preuves ne sont pas valides.

## 🐳 Gestion des conteneurs

- **Voir les logs** : `docker compose logs -f api`
- **Arrêter l'application** : `docker compose down`
- **Vérifier l'état du GPU** : `docker exec -it splatforge-api nvidia-smi`
