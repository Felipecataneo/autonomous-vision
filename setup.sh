#!/bin/bash

# ======================================
# Autonomous Vision System - Setup
# ======================================

set -e

echo "========================================"
echo "🚗 Autonomous Vision System Setup"
echo "========================================"

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função de log
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verifica Python
log_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
log_info "Python $python_version found"

# Verifica se é Python 3.8+
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    log_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi

# Cria ambiente virtual
log_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_info "Virtual environment created"
else
    log_warn "Virtual environment already exists"
fi

# Ativa ambiente
log_info "Activating virtual environment..."
source venv/bin/activate

# Atualiza pip
log_info "Upgrading pip..."
pip install --upgrade pip -q

# Instala dependências
log_info "Installing dependencies..."
pip install -r requirements.txt -q

log_info "✅ Python dependencies installed"

# Verifica Ollama
log_info "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    log_info "✅ Ollama found"
    
    # Verifica se Ollama está rodando
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_info "✅ Ollama is running"
        
        # Verifica se Qwen3-VL está instalado
        if ollama list | grep -q "qwen3-vl"; then
            log_info "✅ Qwen3-VL model found"
        else
            log_warn "Qwen3-VL not found"
            read -p "Download Qwen3-VL (8GB)? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Downloading Qwen3-VL (this may take a while)..."
                ollama pull qwen3-vl:8b
                log_info "✅ Qwen3-VL downloaded"
            fi
        fi
    else
        log_warn "Ollama is not running"
        log_info "Start Ollama with: ollama serve"
    fi
else
    log_warn "Ollama not found"
    log_info "Install Ollama from: https://ollama.com/install"
    log_info "Or run: curl -fsSL https://ollama.com/install.sh | sh"
fi

# Baixa YOLO se necessário
log_info "Checking YOLO models..."
if [ ! -f "yolo11n.pt" ]; then
    log_info "Downloading YOLO11n..."
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
    log_info "✅ YOLO11n downloaded"
else
    log_info "✅ YOLO11n found"
fi

# Cria diretório de saída
log_info "Creating output directory..."
mkdir -p outputs
log_info "✅ Output directory created"

# Testa importações
log_info "Testing imports..."
python3 -c "
import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import requests
print('✅ All imports successful')
" 2>&1

if [ $? -eq 0 ]; then
    log_info "✅ Import test passed"
else
    log_error "Import test failed"
    exit 1
fi

# Resumo
echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Start Ollama: ollama serve"
echo "  3. Run system: python main.py"
echo ""
echo "Examples:"
echo "  python example.py 1  # Webcam"
echo "  python example.py 2  # Video file"
echo "  python example.py 3  # High performance"
echo ""
echo "Documentation: README.md"
echo "========================================"