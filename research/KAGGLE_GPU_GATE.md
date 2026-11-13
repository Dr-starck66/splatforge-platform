# SplatForge — Kaggle GPU availability gate

Goal: use the existing Kaggle free GPU quota to evaluate SplatForge experimentation without buying infrastructure.
Existing CPU pipeline (tools/train_colmap_cpu.py), native-engine smoke and SplatForge_Colab.ipynb remain unchanged.

Run in Kaggle with a GPU accelerator: python tools/kaggle_gpu_probe.py --output /kaggle/working/splatforge-gpu-probe.json
Tests on any CPU: python -m unittest discover -s tests -p test_kaggle_gpu_probe.py -v

A CUDA matrix multiplication PASS only establishes that the runtime executes a small CUDA operation. It does NOT prove actual COLMAP reconstruction, gsplat training, image provenance, 3D quality, persistence or a deployable API.
Without CUDA, fail closed. No cron, new deployment or paid GPU is requested.
For full reconstruction, attach only legally reusable images and verify generated geometry and output in an independent test.