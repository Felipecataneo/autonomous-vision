"""
Detector e rastreador de objetos com YOLO
"""
import numpy as np
from typing import List, Dict
import torch


class YOLODetector:
    """Wrapper para YOLO com tracking"""
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        tracker: str = "bytetrack.yaml",
        device: str = "auto"
    ):
        """
        Args:
            model_path: Caminho do modelo YOLO
            conf_threshold: Confiança mínima
            iou_threshold: IoU para NMS
            tracker: Algoritmo de tracking
            device: cuda, cpu, mps ou auto
        """
        print(f"🔍 Loading YOLO: {model_path}")
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"⚠ Error loading {model_path}: {e}")
            print("   Fallback to yolov8n.pt")
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker = tracker
        
        # Auto-detecta melhor device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"   Device: {self.device}")
        print(f"   Tracker: {self.tracker}")
    
    def detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """
        Detecta e rastreia objetos no frame
        
        Args:
            frame: Frame BGR do OpenCV
        
        Returns:
            Lista de detecções com formato:
            {
                'box': [x1, y1, x2, y2],
                'center': (cx, cy),
                'label': str,
                'conf': float,
                'id': int
            }
        """
        if frame is None or frame.size == 0:
            return []
        
        # YOLO inference + tracking
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker,
            device=self.device
        )
        
        detections = []
        
        if len(results) == 0:
            return detections
        
        boxes = results[0].boxes
        
        if boxes is None or boxes.id is None:
            return detections
        
        # Extrai dados
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        
        for b, c, cf, tid in zip(xyxy, cls, conf, ids):
            x1, y1, x2, y2 = b
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            detections.append({
                'box': b.tolist(),
                'center': center,
                'label': self.model.names[c],
                'conf': float(cf),
                'id': int(tid)
            })
        
        return detections
    
    def export_to_onnx(self, output_path: str = "model.onnx"):
        """Exporta modelo para ONNX para inferência rápida"""
        print(f"📦 Exporting to ONNX: {output_path}")
        self.model.export(format='onnx', simplify=True)
        print("   ✅ Done")