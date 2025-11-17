"""
Preditor de trajetórias com Filtro de Kalman
Suporta modelo de velocidade constante e aceleração constante
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Tuple, List, Dict
import time


class KalmanTrajectoryPredictor:
    """Predição de trajetórias com Kalman Filter otimizado"""
    
    def __init__(
        self, 
        use_acceleration: bool = True,
        measurement_noise: float = 10.0,
        process_noise: float = 0.1,
        initial_covariance: float = 1000.0,
        dt: float = 1.0
    ):
        """
        Args:
            use_acceleration: Se True, usa modelo com aceleração
            measurement_noise: Ruído de medição (R)
            process_noise: Ruído de processo (Q)
            initial_covariance: Covariância inicial (P)
            dt: Delta de tempo entre frames
        """
        self.use_acceleration = use_acceleration
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.initial_covariance = initial_covariance
        self.dt = dt
        
        self.filters: Dict[int, KalmanFilter] = {}
        self.last_update: Dict[int, float] = {}
        self.last_position: Dict[int, Tuple[float, float]] = {}
    
    def _create_filter(self, track_id: int) -> KalmanFilter:
        """Cria novo filtro de Kalman"""
        
        if self.use_acceleration:
            # Modelo com aceleração: [x, y, vx, vy, ax, ay]
            kf = KalmanFilter(dim_x=6, dim_z=2)
            dt = self.dt
            dt2 = 0.5 * dt * dt
            
            # Matriz de transição (aceleração constante)
            kf.F = np.array([
                [1, 0, dt, 0,  dt2, 0],
                [0, 1, 0,  dt, 0,   dt2],
                [0, 0, 1,  0,  dt,  0],
                [0, 0, 0,  1,  0,   dt],
                [0, 0, 0,  0,  1,   0],
                [0, 0, 0,  0,  0,   1]
            ], dtype=float)
            
            # Matriz de observação (só medimos x, y)
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ], dtype=float)
            
            # Ruído de processo (Q): Maior incerteza na aceleração
            kf.Q = np.eye(6) * self.process_noise
            kf.Q[4:, 4:] *= 10
            
        else:
            # Modelo simples: [x, y, vx, vy]
            kf = KalmanFilter(dim_x=4, dim_z=2)
            dt = self.dt
            
            kf.F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0,  dt],
                [0, 0, 1,  0],
                [0, 0, 0,  1]
            ], dtype=float)
            
            kf.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=float)
            
            kf.Q = np.eye(4) * self.process_noise
        
        # Ruído de medição (R) - Confiança na detecção do YOLO
        kf.R = np.eye(2) * self.measurement_noise
        
        # Covariância inicial (P)
        kf.P *= self.initial_covariance
        
        return kf
    
    def get_filter(self, track_id: int) -> KalmanFilter:
        """Retorna filtro existente ou cria novo"""
        if track_id not in self.filters:
            self.filters[track_id] = self._create_filter(track_id)
        return self.filters[track_id]
    
    def update(self, track_id: int, position: Tuple[int, int]) -> None:
        """
        Atualiza filtro com nova medição
        """
        kf = self.get_filter(track_id)
        pos_array = np.array([float(position[0]), float(position[1])])
        
        if track_id not in self.last_update:
            # Primeira medição: inicializa estado
            state_size = 6 if self.use_acceleration else 4
            kf.x = np.zeros(state_size)
            kf.x[0], kf.x[1] = pos_array[0], pos_array[1]
        else:
            # Predição + atualização
            kf.predict()
            kf.update(pos_array)
        
        self.last_update[track_id] = time.time()
        self.last_position[track_id] = position
    
    def predict_trajectory(
        self, 
        track_id: int, 
        steps: int = 60, 
        step_size: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Prediz trajetória futura
        """
        if track_id not in self.filters:
            return []
        
        kf = self.get_filter(track_id)
        trajectory = []
        
        state = kf.x.copy()
        F = kf.F.copy()
        
        for _ in range(0, steps, step_size):
            state = F @ state
            trajectory.append((int(state[0]), int(state[1])))
        
        return trajectory
    
    def get_velocity(self, track_id: int) -> Tuple[float, float]:
        """Retorna velocidade estimada (vx, vy)"""
        if track_id not in self.filters:
            return (0.0, 0.0)
        
        kf = self.filters[track_id]
        return (float(kf.x[2]), float(kf.x[3]))
    
    def get_acceleration(self, track_id: int) -> Tuple[float, float]:
        """Retorna aceleração estimada (ax, ay)"""
        if not self.use_acceleration or track_id not in self.filters:
            return (0.0, 0.0)
        
        kf = self.filters[track_id]
        return (float(kf.x[4]), float(kf.x[5]))
    
    def get_uncertainty(self, track_id: int) -> np.ndarray:
        """Retorna matriz de covariância da posição (2x2)"""
        if track_id not in self.filters:
            return np.eye(2) * self.initial_covariance
        
        kf = self.filters[track_id]
        return kf.P[:2, :2]
    
    def cleanup_old_tracks(self, max_age: float = 5.0) -> None:
        """Remove tracks que não foram atualizados recentemente"""
        now = time.time()
        old_tracks = [
            tid for tid, t in self.last_update.items() 
            if now - t > max_age
        ]
        
        for tid in old_tracks:
            del self.filters[tid]
            del self.last_update[tid]
            if tid in self.last_position:
                del self.last_position[tid]