"""
Interfaces base do sistema de visão autônoma
"""
from typing import Protocol, List, Tuple, Dict, Optional
import numpy as np


class VisionAnalyzer(Protocol):
    """Interface para analisadores de cena (Ollama, OpenAI, etc)"""
    
    def analyze_scene(
        self, 
        frame: np.ndarray, 
        detections: List[Dict], 
        collisions: List[Dict]
    ) -> Optional[Dict]:
        """Analisa cena e retorna predições"""
        ...
    
    def should_analyze(
        self, 
        detections: List[Dict], 
        collisions: List[Dict]
    ) -> bool:
        """Decide se deve executar análise"""
        ...


class TrajectoryPredictor(Protocol):
    """Interface para predição de trajetórias"""
    
    def update(self, track_id: int, position: Tuple[int, int]) -> None:
        """Atualiza modelo com nova posição"""
        ...
    
    def predict_trajectory(
        self, 
        track_id: int, 
        steps: int, 
        step_size: int
    ) -> List[Tuple[int, int]]:
        """Prediz trajetória futura"""
        ...
    
    def get_velocity(self, track_id: int) -> Tuple[float, float]:
        """Retorna velocidade estimada"""
        ...


class CollisionDetectorInterface(Protocol):
    """Interface para detecção de colisões"""
    
    @staticmethod
    def will_collide(
        bbox1: List[int],
        bbox2: List[int],
        traj1: List[Tuple[int, int]], 
        traj2: List[Tuple[int, int]], 
        threshold: float
    ) -> Tuple[bool, Optional[int], Optional[Tuple[int, int]]]:
        """Verifica se haverá colisão"""
        ...
    
    @staticmethod
    def check_all_collisions(detections: List[Dict]) -> List[Dict]:
        """Verifica todas as combinações"""
        ...


class ObjectDetector(Protocol):
    """Interface para detectores de objetos"""
    
    def detect_and_track(
        self, 
        frame: np.ndarray
    ) -> List[Dict]:
        """Detecta e rastreia objetos"""
        ...