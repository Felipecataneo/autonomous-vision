"""
Configurações centralizadas do sistema - Otimizado para RTX 4070
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class YOLOConfig:
    """Configurações do YOLO"""
    # MUDANÇA: Usando o modelo 'medium'. Sua 4070 aguenta sem problemas.
    # A precisão de detecção será drasticamente maior.
    # (Você precisará baixar este modelo, mas o sistema deve fazer isso automaticamente)
    model_path: str = "yolo11m.pt"
    
    # MUDANÇA: Menor threshold. Com um modelo maior, podemos confiar mais em
    # detecções com menor confiança, capturando mais objetos distantes/ocultos.
    conf_threshold: float = 0.20
    
    iou_threshold: float = 0.45
    tracker: str = "bytetrack.yaml"
    
    # MUDANÇA: Forçar o uso de CUDA. Com 'auto', ele já escolheria, mas
    # especificar 'cuda:0' é uma boa prática para garantir o uso da GPU principal.
    device: str = "cuda:0"
    

@dataclass
class KalmanConfig:
    """Configurações do Filtro de Kalman"""
    use_acceleration: bool = True
    measurement_noise: float = 10.0
    
    # MUDANÇA: Aumentamos o ruído de processo. Como o sistema roda mais rápido,
    # queremos que o filtro reaja mais rapidamente a mudanças de direção.
    # A trajetória ficará um pouco menos "suave", mas muito mais reativa e realista.
    process_noise: float = 0.3
    
    initial_covariance: float = 1000.0
    

@dataclass
class CollisionConfig:
    """Configurações de detecção de colisão"""
    # MUDANÇA: Um threshold um pouco mais baixo para ser mais sensível a
    # sobreposições iminentes.
    iou_threshold: float = 0.25
    
    time_horizon: int = 60
    prediction_step: int = 5
    

@dataclass
class AnalyzerConfig:
    """Configurações do analisador de cena"""
    provider: Literal["ollama", "openai"] = "ollama"
    model: str = "qwen3-vl:8b"
    base_url: str = "http://localhost:11434"
    
    # MUDANÇA: Análise muito mais frequente. Sua GPU acelera o Ollama,
    # então a latência de inferência será baixa. Podemos consultar a IA com mais frequência.
    analysis_interval: float = 1.5
    
    adaptive_interval: bool = True
    max_timeout: int = 45
    
    # MUDANÇA: Imagens maiores para a IA. Isso dá muito mais contexto
    # e detalhes para a análise, resultando em respostas de maior qualidade.
    image_max_size: int = 768
    
    image_quality: int = 90
    

@dataclass
class VideoConfig:
    """Configurações de vídeo"""
    source: str = "video.mp4" # Mude para "0" para webcam
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