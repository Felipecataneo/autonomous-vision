# 🚀 Quick Start Guide

## Instalação (5 minutos)

### 1. Clone e Setup

```bash
# Clone o repositório
git clone https://github.com/felipecataneo/autonomous-vision.git
cd autonomous-vision

# Execute setup automático
chmod +x setup.sh
./setup.sh
```

### 2. Ativa Ambiente

```bash
source venv/bin/activate
```

### 3. Inicia Ollama

```bash
# Terminal separado
ollama serve
```

## Uso Básico

### Webcam

```bash
python example.py 1
```

### Vídeo

```bash
python example.py 2
```

### High Performance

```bash
python example.py 3
```

## Customização Rápida

### Arquivo: `quick_run.py`

```python
from core.config import SystemConfig
from main import AutonomousVisionSystem

config = SystemConfig()
config.video.source = "seu_video.mp4"
config.video.save_output = True

system = AutonomousVisionSystem(config)
system.run()
```

Execute:
```bash
python quick_run.py
```

## Configurações Comuns

### 1. Mudar fonte de vídeo

```python
config.video.source = "0"           # Webcam
config.video.source = "video.mp4"   # Arquivo
config.video.source = "rtsp://..."  # Stream IP
```

### 2. Ajustar performance

```python
# Mais rápido
config.yolo.model_path = "yolo11n.pt"
config.kalman.use_acceleration = False
config.analyzer.analysis_interval = 5.0

# Mais preciso
config.yolo.model_path = "yolo11m.pt"
config.kalman.use_acceleration = True
config.analyzer.analysis_interval = 2.0
```

### 3. Salvar/Mostrar output

```python
config.video.save_output = True   # Salva vídeo
config.video.show_window = False  # Headless (sem janela)
```

## Troubleshooting

### Ollama não conecta

```bash
# Verifica status
curl http://localhost:11434/api/tags

# Reinicia
pkill ollama
ollama serve
```

### FPS baixo

```python
# Use modelo menor
config.yolo.model_path = "yolo11n.pt"

# Desativa aceleração
config.kalman.use_acceleration = False

# Menos análises
config.analyzer.analysis_interval = 10.0
```

### Erro de importação

```bash
# Reinstala dependências
pip install -r requirements.txt --force-reinstall
```

## Próximos Passos

1. **Leia README.md** para features completas
2. **Teste example.py** para ver todas as configurações
3. **Customize core/config.py** para suas necessidades
4. **Exporte para ONNX** para inferência mais rápida

## Comandos Úteis

```bash
# Lista modelos Ollama
ollama list

# Baixa modelo diferente
ollama pull qwen3-vl:14b

# Exporta YOLO para ONNX
python example.py 6

# Testa componentes
python example.py 7

# Benchmark
python example.py 8
```

## Suporte

- 📖 Documentação: `README.md`
- 🐛 Issues: GitHub Issues
- 💬 Discussões: GitHub Discussions

---

**Pronto para rodar! 🚗💨**