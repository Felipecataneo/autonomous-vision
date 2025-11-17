"""
Sistema de visualização HUD
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import time


class HUDDrawer:
    """Desenha interface visual do sistema"""
    
    def __init__(self):
        self.track_colors = {}
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Cores padrão
        self.COLORS = {
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'orange': (0, 165, 255),
            'red': (0, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        
        self.RISK_COLORS = {
            'low': self.COLORS['green'],
            'medium': self.COLORS['yellow'],
            'high': self.COLORS['orange'],
            'critical': self.COLORS['red']
        }
    
    def update_fps(self):
        """Atualiza cálculo de FPS"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Retorna cor única por track"""
        if track_id not in self.track_colors:
            np.random.seed(track_id)
            color_list = [
                self.COLORS['cyan'],
                self.COLORS['magenta'],
                self.COLORS['yellow'],
                self.COLORS['green'],
                self.COLORS['orange']
            ]
            self.track_colors[track_id] = color_list[track_id % len(color_list)]
        return self.track_colors[track_id]
    
    def draw_trajectory(
        self,
        frame: np.ndarray,
        start_pos: Tuple[int, int],
        trajectory: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        is_critical: bool = False
    ):
        """Desenha trajetória com gradiente de opacidade"""
        if not trajectory:
            return
        
        points = np.array([start_pos] + trajectory, dtype=np.int32)
        
        # Desenha linhas com fade
        for i in range(len(points) - 1):
            alpha = 1.0 - (i / len(points))
            overlay = frame.copy()
            
            thickness = 3 if is_critical else 2
            line_color = self.COLORS['red'] if is_critical else color
            
            cv2.line(overlay, tuple(points[i]), tuple(points[i+1]), line_color, thickness)
            cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, frame)
        
        # Marca final da trajetória
        if len(trajectory) > 0:
            end = trajectory[-1]
            cv2.circle(frame, end, 6, self.COLORS['red'] if is_critical else color, -1)
            cv2.circle(frame, end, 8, self.COLORS['white'], 1)
            
            # Label de tempo
            cv2.putText(
                frame, "t+2s", (end[0] + 10, end[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['white'], 1
            )
    
    def draw_detection_box(
        self,
        frame: np.ndarray,
        detection: Dict,
        is_critical: bool = False
    ):
        """Desenha bbox com informações"""
        x1, y1, x2, y2 = detection['box']
        center = detection['center']
        color = self.COLORS['red'] if is_critical else self.get_track_color(detection['id'])
        
        # Box principal
        thickness = 3 if is_critical else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Cantos decorativos
        corner_len = 15
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
        
        # Centro
        cv2.circle(frame, center, 3, color, -1)
        cv2.circle(frame, center, 5, self.COLORS['white'], 1)
        
        # Informações
        velocity = detection.get('velocity', (0, 0))
        speed = np.linalg.norm(velocity)
        
        info = f"{detection['label'].upper()} #{detection['id']} | {speed:.1f}px/f"
        
        # Background do texto
        (text_w, text_h), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), self.COLORS['black'], -1)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, 2)
        
        cv2.putText(frame, info, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['white'], 1)
    
    def draw_collision_warning(self, frame: np.ndarray, collision: Dict):
        """Desenha aviso de colisão iminente"""
        point = collision['point']
        time_frames = collision['time']
        time_s = time_frames * 5 / 30  # Assumindo 30 FPS
        
        # Círculo pulsante
        pulse = int(20 + 10 * np.sin(time.time() * 5))
        cv2.circle(frame, point, pulse, self.COLORS['red'], 3)
        cv2.circle(frame, point, pulse + 5, self.COLORS['white'], 1)
        
        # Texto de aviso
        severity = collision.get('severity', 'unknown').upper()
        text = f"COLLISION: {time_s:.1f}s ({severity})"
        
        cv2.putText(frame, text, (point[0] - 80, point[1] - pulse - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['red'], 2)
    
    def draw_main_hud(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        predictions: Dict,
        collisions: List[Dict],
        analysis_count: int
    ) -> np.ndarray:
        """Desenha HUD principal"""
        h, w = frame.shape[:2]
        
        # Header com transparência
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (5, 5, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, 65), (w, 65), self.COLORS['cyan'], 2)
        
        # Título
        cv2.putText(frame, "AUTONOMOUS VISION | QWEN3-VL", (15, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLORS['cyan'], 2)
        
        # Risk level
        risk = predictions.get('risk_level', 'unknown').upper()
        risk_color = self.RISK_COLORS.get(risk.lower(), self.COLORS['white'])
        
        cv2.putText(frame, f"RISK: {risk}", (15, 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)
        
        # Stats à direita
        stats = f"FPS: {self.fps:.1f} | OBJ: {len(detections)} | INF: {analysis_count}"
        cv2.putText(frame, stats, (w - 380, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        scene = predictions.get('scene_type', 'unknown').upper()
        cv2.putText(frame, f"SCENE: {scene}", (w - 380, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Collision warning
        if collisions:
            warn_y = 75
            closest = min(collisions, key=lambda c: c['time'])
            time_s = closest['time'] * 5 / 30
            
            cv2.rectangle(overlay, (10, warn_y), (w - 10, warn_y + 35), (0, 0, 100), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (10, warn_y), (w - 10, warn_y + 35), self.COLORS['red'], 3)
            
            text = f"⚠ COLLISION: {closest['labels'][0]} vs {closest['labels'][1]} in {time_s:.1f}s"
            cv2.putText(frame, text, (25, warn_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['red'], 2)
        
        return frame
    
    def draw_uncertainty_ellipse(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        covariance: np.ndarray,
        color: Tuple[int, int, int] = None
    ):
        """Desenha elipse de incerteza da posição (Kalman)"""
        if color is None:
            color = self.COLORS['yellow']
        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(covariance)
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
            
            # 3-sigma (99.7% confiança)
            axes = (
                int(np.sqrt(eigenvalues[0]) * 3),
                int(np.sqrt(eigenvalues[1]) * 3)
            )
            
            cv2.ellipse(frame, center, axes, angle, 0, 360, color, 1)
        except:
            pass  # Ignora se covariância inválida