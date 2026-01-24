#!/bin/bash
# Script para instalar dependencias del sistema y Python necesarias para los experimentos

set -e

echo "Instalando dependencias..."

# Actualizar índice de paquetes
sudo apt-get update -qq

# Instalar paquetes desde requirements-apt.txt
if [ -f requirements/requirements-apt.txt ]; then
    echo "📦 Instalando paquetes APT desde requirements/requirements-apt.txt..."
    grep -v '^#' requirements/requirements-apt.txt | grep -v '^$' | xargs sudo apt-get install -y -qq
    echo "✅ Dependencias del sistema instaladas"
fi

echo ""

# Instalar paquetes Python desde requirements.txt
if [ -f requirements/requirements-pip.txt ]; then
    echo "📦 Instalando paquetes Python desde requirements/requirements-pip.txt..."
    pip install -r requirements/requirements-pip.txt
    echo "✅ Dependencias de Python instaladas"
fi

echo ""
echo "🔍 Verificando NVIDIA drivers y CUDA..."

# Verificar nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  ADVERTENCIA: nvidia-smi no encontrado"
    echo "   Instala NVIDIA drivers manualmente"
else
    nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs -I {} echo "✅ GPU detectada: {}"
fi

# Verificar nvcc
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  ADVERTENCIA: nvcc no encontrado"
    echo "   Instala CUDA toolkit manualmente"
else
    nvcc --version | grep "release" | awk '{print $5}' | xargs -I {} echo "✅ CUDA version: {}"
fi
