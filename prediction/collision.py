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
        traj2: List[Tuple[int, int]],
        iou_threshold: float
    ) -> Tuple[bool, Optional[int], Optional[Tuple[int, int]]]:
        """
        Detecta colisão baseado em IoU de bboxes futuras
        """
        if not traj1 or not traj2:
            return False, None, None
        
        max_iou = 0.0
        collision_frame = None
        collision_point = None
        
        for i, (p1, p2) in enumerate(zip(traj1, traj2)):
            future_box1 = scale_bbox(bbox1, p1)
            future_box2 = scale_bbox(bbox2, p2)
            
            iou = compute_iou(future_box1, future_box2)
            
            if iou > max_iou:
                max_iou = iou
            
            if iou > iou_threshold:
                collision_frame = i
                collision_point = (
                    (p1[0] + p2[0]) // 2,
                    (p1[1] + p2[1]) // 2
                )
                return True, collision_frame, collision_point
        
        return False, None, None
    
    def check_all_collisions(
        self,
        detections: List[Dict],
        use_iou: bool = True
    ) -> List[Dict]:
        """
        Verifica colisões entre todas as combinações de objetos
        """
        collisions = []
        
        for i, det1 in enumerate(detections):
            for det2 in detections[i+1:]:
                traj1 = det1.get('trajectory', [])
                traj2 = det2.get('trajectory', [])
                
                will_collide, frame, point = self.will_collide_iou(
                    det1['box'],
                    det2['box'],
                    traj1,
                    traj2,
                    self.iou_threshold
                )
                
                if will_collide:
                    pos1, vel1 = det1['center'], det1.get('velocity', (0, 0))
                    pos2, vel2 = det2['center'], det2.get('velocity', (0, 0))
                    
                    ttc = compute_ttc(pos1, vel1, pos2, vel2)
                    mad, t_mad = compute_minimum_distance(pos1, vel1, pos2, vel2)
                    
                    # Define severidade baseada no tempo para colisão (em frames)
                    if ttc < 30: # ~1s
                        severity = 'critical'
                    elif ttc < 90: # ~3s
                        severity = 'high'
                    else:
                        severity = 'medium'

                    collisions.append({
                        'objects': [det1['id'], det2['id']],
                        'labels': [det1['label'], det2['label']],
                        'time': frame,
                        'point': point,
                        'ttc': ttc,
                        'min_distance': mad,
                        'severity': severity
                    })
        
        collisions.sort(key=lambda c: c['time'])
        return collisions