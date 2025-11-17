"""
Configurações centralizadas do sistema
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class YOLOConfig:
    """Configurações do YOLO"""
    model_path: str = "yolo11n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    tracker: str = "bytetrack.yaml"
    device: str = "auto"  # auto, cuda, cpu, mps
    

@dataclass
class KalmanConfig:
    """Configurações do Filtro de Kalman"""
    use_acceleration: bool = True
    measurement_noise: float = 10.0
    process_noise: float = 0.1
    initial_covariance: float = 1000.0
    

@dataclass
class CollisionConfig:
    """Configurações de detecção de colisão"""
    iou_threshold: float = 0.3
    time_horizon: int = 60  # frames
    prediction_step: int = 5  # frames
    

@dataclass
class AnalyzerConfig:
    """Configurações do analisador de cena"""
    provider: Literal["ollama", "openai"] = "ollama"
    model: str = "qwen3-vl:8b"
    base_url: str = "http://localhost:11434"
    analysis_interval: float = 3.0
    adaptive_interval: bool = True
    max_timeout: int = 45
    image_max_size: int = 512
    image_quality: int = 80
    

@dataclass
class VideoConfig:
    """Configurações de vídeo"""
    source: str = "0"
    save_output: bool = False
    show_window: bool = True
    output_fps: int = 30
    

@dataclass
class SystemConfig:
    """Configuração completa do sistema"""
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    
    use_threading: bool = True
    enable_logging: bool = True
    log_file: str = "system.log"