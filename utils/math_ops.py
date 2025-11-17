"""
Operações matemáticas otimizadas com NumPy e Numba
"""
import numpy as np
from typing import Tuple, List

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func


@njit
def fast_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Distância euclidiana rápida com Numba"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def vectorized_distances(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Calcula distâncias entre dois conjuntos de pontos
    
    Args:
        points1: Array (N, 2)
        points2: Array (N, 2)
    
    Returns:
        Array (N,) com distâncias
    """
    return np.linalg.norm(points1 - points2, axis=1)


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calcula IoU (Intersection over Union) entre duas bboxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU score [0, 1]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def scale_bbox(bbox: List[int], center: Tuple[int, int]) -> List[int]:
    """
    Move bbox para nova posição mantendo tamanho
    
    Args:
        bbox: [x1, y1, x2, y2]
        center: (cx, cy) nova posição do centro
    
    Returns:
        Nova bbox [x1, y1, x2, y2]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    return [
        center[0] - w // 2,
        center[1] - h // 2,
        center[0] + w // 2,
        center[1] + h // 2
    ]


def compute_velocity(
    pos1: Tuple[float, float], 
    pos2: Tuple[float, float], 
    dt: float = 1.0
) -> Tuple[float, float]:
    """Calcula velocidade entre duas posições"""
    return ((pos2[0] - pos1[0]) / dt, (pos2[1] - pos1[1]) / dt)


def compute_acceleration(
    vel1: Tuple[float, float], 
    vel2: Tuple[float, float], 
    dt: float = 1.0
) -> Tuple[float, float]:
    """Calcula aceleração entre duas velocidades"""
    return ((vel2[0] - vel1[0]) / dt, (vel2[1] - vel1[1]) / dt)