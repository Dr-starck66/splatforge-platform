#!/bin/bash

# ══════════════════════════════════════════════════════════
# SPLAT·FORGE — GPU Runtime Setup Script
# Installs Docker, NVIDIA Drivers, and NVIDIA Container Toolkit
# ══════════════════════════════════════════════════════════

set -e

echo "🚀 Starting SPLAT·FORGE GPU Runtime Setup..."

# 1. Update System
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed."
else
    echo "✅ Docker is already installed."
fi

# 3. Install NVIDIA Drivers (if not present)
if ! command -v nvidia-smi &> /dev/null; then
    echo "🏎️ Installing NVIDIA Drivers..."
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    echo "⚠️ NVIDIA Drivers installed. A REBOOT might be required after this script."
else
    echo "✅ NVIDIA Drivers detected."
fi

# 4. Install NVIDIA Container Toolkit
echo "🛠️ Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && sudo apt-get update \
  && sudo apt-get install -y nvidia-container-toolkit

# 5. Configure Docker to use NVIDIA Runtime
echo "⚙️ Configuring Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 6. Verification
echo "🔍 Verifying installation..."
if sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "🎉 SUCCESS: GPU is accessible from Docker!"
else
    echo "❌ ERROR: GPU test failed. You might need to REBOOT your server."
fi

echo "--------------------------------------------------------"
echo "✅ Setup complete! You can now run:"
echo "   docker compose up -d --build"
echo "--------------------------------------------------------"
