"""Trajectory prediction and collision detection"""
from .kalman import KalmanTrajectoryPredictor
from .collision import CollisionDetector
from .physics import (
    compute_ttc,
    compute_minimum_distance,
    compute_collision_severity,
    extrapolate_position
)

__all__ = [
    'KalmanTrajectoryPredictor',
    'CollisionDetector',
    'compute_ttc',
    'compute_minimum_distance',
    'compute_collision_severity',
    'extrapolate_position'
]
