"""
Detector de colisões com física realista
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from utils.math_ops import compute_iou, scale_bbox, vectorized_distances
from .physics import compute_ttc, compute_minimum_distance


class CollisionDetector:
    """Detecta colisões usando geometria e física"""
    
    def __init__(self, iou_threshold: float = 0.3):
        """
        Args:
            iou_threshold: IoU mínimo para considerar colisão
        """
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def will_collide_iou(
        bbox1: List[int],
        bbox2: List[int],
        traj1: List[Tuple[int, int]],
        traj2: List[Tuple[int, int]]
    ) -> Tuple[bool, Optional[int], Optional[Tuple[int, int]]]:
        """
        Detecta colisão baseado em IoU de bboxes futuras
        
        Returns:
            (vai_colidir, frame_colisão, ponto_colisão)
        """
        if not traj1 or not traj2:
            return False, None, None
        
        max_iou = 0.0
        collision_frame = None
        collision_point = None
        
        # Verifica IoU em cada passo da trajetória
        for i, (p1, p2) in enumerate(zip(traj1, traj2)):
            # Projeta bboxes para posições futuras
            future_box1 = scale_bbox(bbox1, p1)
            future_box2 = scale_bbox(bbox2, p2)
            
            iou = compute_iou(future_box1, future_box2)
            
            if iou > max_iou:
                max_iou = iou
                collision_frame = i
                collision_point = (
                    (p1[0] + p2[0]) // 2,
                    (p1[1] + p2[1]) // 2
                )
        
        will_collide = max_iou > 0.3
        return will_collide, collision_frame, collision_point if will_collide else None
    
    @staticmethod
    def will_collide_distance(
        traj1: List[Tuple[int, int]],
        traj2: List[Tuple[int, int]],
        threshold: float = 50.0
    ) -> Tuple[bool, Optional[int], Optional[Tuple[int, int]]]:
        """
        Detecta colisão por distância (método rápido, menos preciso)
        
        Returns:
            (vai_colidir, frame_colisão, ponto_colisão)
        """
        if not traj1 or not traj2:
            return False, None, None
        
        # Converte para numpy para vetorização
        traj1_np = np.array(traj1)
        traj2_np = np.array(traj2)
        
        # Calcula todas as distâncias de uma vez
        distances = vectorized_distances(traj1_np, traj2_np)
        
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        if min_dist < threshold:
            point = (
                (traj1[min_dist_idx][0] + traj2[min_dist_idx][0]) // 2,
                (traj1[min_dist_idx][1] + traj2[min_dist_idx][1]) // 2
            )
            return True, int(min_dist_idx), point
        
        return False, None, None
    
    def check_all_collisions(
        self,
        detections: List[Dict],
        use_iou: bool = True
    ) -> List[Dict]:
        """
        Verifica colisões entre todas as combinações de objetos
        
        Args:
            detections: Lista de detecções com 'box', 'trajectory', etc
            use_iou: Se True, usa método IoU (mais preciso)
        
        Returns:
            Lista de colisões detectadas
        """
        collisions = []
        
        for i, det1 in enumerate(detections):
            for det2 in detections[i+1:]:
                traj1 = det1.get('trajectory', [])
                traj2 = det2.get('trajectory', [])
                
                if use_iou:
                    will_collide, frame, point = self.will_collide_iou(
                        det1['box'],
                        det2['box'],
                        traj1,
                        traj2
                    )
                else:
                    will_collide, frame, point = self.will_collide_distance(
                        traj1,
                        traj2,
                        threshold=60.0
                    )
                
                if will_collide:
                    # Calcula TTC e MAD para informação adicional
                    vel1 = det1.get('velocity', (0, 0))
                    vel2 = det2.get('velocity', (0, 0))
                    pos1 = det1['center']
                    pos2 = det2['center']
                    
                    ttc = compute_ttc(pos1, vel1, pos2, vel2)
                    mad, t_mad = compute_minimum_distance(pos1, vel1, pos2, vel2)
                    
                    collisions.append({
                        'objects': [det1['id'], det2['id']],
                        'labels': [det1['label'], det2['label']],
                        'time': frame,
                        'point': point,
                        'ttc': ttc,
                        'min_distance': mad,
                        'severity': 'high' if ttc < 30 else 'medium' if ttc < 60 else 'low'
                    })
        
        # Ordena por tempo (colisões mais iminentes primeiro)
        collisions.sort(key=lambda c: c['time'])
        
        return collisions