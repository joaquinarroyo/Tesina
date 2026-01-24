# Tesina

## 📋 Requisitos del Sistema

### Hardware
- GPU NVIDIA con soporte CUDA (Compute Capability ≥ 3.5)
- Mínimo 8GB RAM
- 20GB espacio en disco

### Software Base
- Ubuntu 20.04+ (o distribución basada en Debian)
- Python 3.10+
- NVIDIA Drivers ≥ 550.0
- CUDA Toolkit 12.6

## Instalación

### 1. Preparar Entorno Python

```bash
# Clonar repositorio
git clone <repo-url>
cd Tesina

# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Ejecutar script de instalación (requiere sudo para paquetes del sistema)
chmod +x setup_system.sh
./setup_system.sh
```

El script instalará automáticamente:
- ✅ Dependencias del sistema (git, openssh-client, task-spooler, nvtop)
- ✅ Dependencias de Python (PyTorch 2.10.0, torchvision, nvidia-ml-py, pandas)
- ⚠️ Verificará NVIDIA drivers y CUDA toolkit (deben estar preinstalados)

### 2. Instalar NVIDIA Drivers (si no están instalados)

Los **drivers NVIDIA** permiten al sistema operativo comunicarse con la GPU.

```bash
# Verificar si ya tienes drivers
nvidia-smi

# Si falla, instalar drivers (ejemplo con driver 550)
sudo apt update
sudo apt install -y nvidia-driver-550

# IMPORTANTE: Reiniciar el sistema después de instalar drivers
# En servidores remotos (GCP, AWS, etc.), usar la consola web para reiniciar
# En sistemas locales: sudo reboot
```

Después del reinicio, `nvidia-smi` debe mostrar información de tu GPU:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
...
```

### 3. Instalar CUDA Toolkit (si no está instalado)

**CUDA Toolkit** contiene:
- **nvcc**: Compilador CUDA (compila código GPU)
- **cuDNN**: Bibliotecas optimizadas para deep learning
- **cuBLAS, cuFFT**: Bibliotecas matemáticas para GPU
- Headers y herramientas de desarrollo

```bash
cd ~/Tesina

# Descargar CUDA 12.6.3
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

# Instalar SOLO toolkit (--toolkit mantiene los drivers existentes)
sudo sh cuda_12.6.3_560.35.05_linux.run --silent --toolkit --override

# Agregar CUDA al PATH del sistema
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verificar instalación
nvcc --version

# Opcional: Eliminar instalador (libera ~4GB de espacio)
rm cuda_12.6.3_560.35.05_linux.run
```

**Salida esperada:**
```
Cuda compilation tools, release 12.6, V12.6.85
```

**Nota:** `--override` permite instalar CUDA en sistemas con gcc 14+ (CUDA 12.6 soporta oficialmente hasta gcc 12).

### 4. Verificar Instalación Completa

```bash
# 1. Verificar drivers NVIDIA
nvidia-smi

# 2. Verificar CUDA toolkit
nvcc --version

# 3. Verificar PyTorch con CUDA (debe estar en venv activado)
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Salida esperada del test de PyTorch:**
```
PyTorch: 2.10.0+cu126
CUDA disponible: True
CUDA version: 12.6
GPU: Tesla T4 (o tu GPU)
```

Si todo muestra correctamente, la instalación está completa y lista para ejecutar experimentos.

## 🧪 Ejecutar Experimentos

### Modo 1: Experimentos Secuenciales (Baseline)

```bash
# Experimento E1 (MNIST + SimpleCNN)
python -m experiments.sequential --exp E1 --out ./runs --repeat 3 --seed 42 --gpu-index 0

# Experimento E3 (CIFAR-10 + ResNet-18)
python -m experiments.sequential --exp E3 --out ./runs --repeat 3 --seed 100 --gpu-index 0
```

### Modo 2: Experimentos MPS (Paralelo)

```bash
# 1. Iniciar MPS daemon
source setup_mps.sh

# 2. Ejecutar experimentos paralelos
python -m experiments.mps --exp E1 --parallel 2 --out ./runs --seed 1000 --mps-group MPS_E1_x2

# 3. Monitorear GPU
nvidia-smi dmon -s pucvmet -c 100  # Monitoreo continuo
nvtop  # Monitor interactivo

# 4. Al terminar, detener MPS
echo quit | nvidia-cuda-mps-control
```

## 📊 Estructura de Resultados

```
runs/
├── master_runs.csv                    # Resumen de todos los experimentos
└── E1_mnist_simplecnn_mps_<timestamp>_pid<pid>/
    ├── run.json                       # Configuración del experimento
    ├── summary_row.csv                # Métricas finales
    ├── epochs.csv                     # Métricas por época
    └── gpu_samples.csv                # Muestras de utilización GPU
```