"""
Cálculos físicos para análise de colisão
"""
import numpy as np
from typing import Tuple, Optional


def compute_ttc(
    pos1: Tuple[float, float],
    vel1: Tuple[float, float],
    pos2: Tuple[float, float],
    vel2: Tuple[float, float]
) -> float:
    """
    Time To Collision (TTC) entre dois objetos
    
    Args:
        pos1, pos2: Posições (x, y)
        vel1, vel2: Velocidades (vx, vy)
    
    Returns:
        TTC em frames (inf se não colidem)
    """
    rel_pos = np.array(pos2) - np.array(pos1)
    rel_vel = np.array(vel2) - np.array(vel1)
    
    # Se estão se afastando
    if np.dot(rel_pos, rel_vel) >= 0:
        return float('inf')
    
    # TTC = -dot(rel_pos, rel_vel) / dot(rel_vel, rel_vel)
    vel_squared = np.dot(rel_vel, rel_vel)
    if vel_squared < 1e-6:
        return float('inf')
    
    ttc = -np.dot(rel_pos, rel_vel) / vel_squared
    return max(0.0, ttc)


def compute_minimum_distance(
    pos1: Tuple[float, float],
    vel1: Tuple[float, float],
    pos2: Tuple[float, float],
    vel2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calcula distância mínima de aproximação (MAD)
    
    Returns:
        (distancia_minima, tempo_até_distancia_minima)
    """
    rel_pos = np.array(pos2) - np.array(pos1)
    rel_vel = np.array(vel2) - np.array(vel1)
    
    vel_squared = np.dot(rel_vel, rel_vel)
    
    if vel_squared < 1e-6:
        # Velocidades iguais ou nulas
        return np.linalg.norm(rel_pos), 0.0
    
    # Tempo até distância mínima
    t_min = -np.dot(rel_pos, rel_vel) / vel_squared
    t_min = max(0.0, t_min)
    
    # Posição no tempo t_min
    future_pos1 = np.array(pos1) + t_min * np.array(vel1)
    future_pos2 = np.array(pos2) + t_min * np.array(vel2)
    
    min_dist = np.linalg.norm(future_pos2 - future_pos1)
    
    return min_dist, t_min


def compute_collision_severity(
    mass1: float,
    vel1: Tuple[float, float],
    mass2: float,
    vel2: Tuple[float, float]
) -> float:
    """
    Estima severidade de colisão baseado em momentum
    
    Returns:
        Energia cinética relativa (proxy para severidade)
    """
    v1 = np.linalg.norm(vel1)
    v2 = np.linalg.norm(vel2)
    
    # Energia cinética relativa
    ke = 0.5 * mass1 * v1**2 + 0.5 * mass2 * v2**2
    
    return ke


def extrapolate_position(
    pos: Tuple[float, float],
    vel: Tuple[float, float],
    acc: Tuple[float, float],
    t: float
) -> Tuple[float, float]:
    """
    Extrapola posição futura com aceleração
    
    x(t) = x0 + v0*t + 0.5*a*t^2
    """
    x = pos[0] + vel[0] * t + 0.5 * acc[0] * t * t
    y = pos[1] + vel[1] * t + 0.5 * acc[1] * t * t
    
    return (x, y)