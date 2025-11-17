"""
Sistema de Visão Autônoma - Main
Integração completa de todos os módulos
"""
import sys
import time
from pathlib import Path

# Imports dos módulos
from core.config import SystemConfig
from detection.yolo_detector import YOLODetector
from prediction.kalman import KalmanTrajectoryPredictor
from prediction.collision import CollisionDetector
from analysis.ollama import OllamaVisionAnalyzer
from visualization.hud import HUDDrawer
from utils.video import VideoManager, VideoDisplay


class AutonomousVisionSystem:
    """Sistema principal de visão autônoma"""
    
    def __init__(self, config: SystemConfig = None):
        if config is None: config = SystemConfig()
        self.config = config
        
        print("="*60 + "\n🚗 AUTONOMOUS VISION SYSTEM v2.0\n" + "="*60)
        self._init_components()
    
    def _init_components(self):
        """Inicializa todos os componentes"""
        print("\n[1/5] Initializing YOLO Detector...")
        self.detector = YOLODetector(
            model_path=self.config.yolo.model_path,
            conf_threshold=self.config.yolo.conf_threshold,
            iou_threshold=self.config.yolo.iou_threshold,
            tracker=self.config.yolo.tracker,
            device=self.config.yolo.device
        )
        
        print("\n[2/5] Initializing Kalman Predictor...")
        self.predictor = KalmanTrajectoryPredictor(
            use_acceleration=self.config.kalman.use_acceleration
        )
        print(f"   Mode: {'Acceleration' if self.config.kalman.use_acceleration else 'Velocity'}")
        
        print("\n[3/5] Initializing Collision Detector...")
        self.collision_detector = CollisionDetector(
            iou_threshold=self.config.collision.iou_threshold
        )
        
        print("\n[4/5] Initializing Scene Analyzer...")
        self.analyzer = OllamaVisionAnalyzer(
            use_threading=self.config.use_threading,
            adaptive_interval=self.config.analyzer.adaptive_interval,
            analysis_interval=self.config.analyzer.analysis_interval
        )
        
        print("\n[5/5] Initializing HUD...")
        self.hud = HUDDrawer()
        print("\n✅ All systems ready\n")
    
    def process_frame(self, frame):
        """Processa um frame completo"""
        # 1. Detecção e tracking
        detections = self.detector.detect_and_track(frame)
        
        # 2. Atualiza Kalman e prediz trajetórias
        for det in detections:
            self.predictor.update(det['id'], det['center'])
            det['velocity'] = self.predictor.get_velocity(det['id'])
            det['trajectory'] = self.predictor.predict_trajectory(
                det['id'],
                steps=self.config.collision.time_horizon,
                step_size=self.config.collision.prediction_step
            )
            if self.config.kalman.use_acceleration:
                det['acceleration'] = self.predictor.get_acceleration(det['id'])
        
        # 3. Detecção de colisões
        collisions = self.collision_detector.check_all_collisions(detections)
        
        # 4. Análise de cena com IA
        new_analysis = self.analyzer.analyze_scene(frame, detections, collisions)
        analysis = new_analysis if new_analysis else self.analyzer.get_latest_prediction()
        
        # 5. Desenha visualizações
        critical_ids = analysis.get('critical_objects', [])
        
        for collision in collisions:
            self.hud.draw_collision_warning(frame, collision)
        
        for det in detections:
            is_critical = det['id'] in critical_ids or str(det['id']) in critical_ids
            self.hud.draw_detection_box(frame, det, is_critical)
            self.hud.draw_trajectory(
                frame,
                det['center'],
                det['trajectory'],
                self.hud.get_track_color(det['id']),
                is_critical
            )
        
        frame = self.hud.draw_main_hud(
            frame, detections, analysis, collisions, self.analyzer.analysis_count
        )
        
        return frame, detections, collisions, analysis
    
    def run(self):
        """Loop principal do sistema"""
        video_mgr = VideoManager(self.config.video.source)
        if not video_mgr.open(): return
        
        if self.config.video.save_output: video_mgr.setup_writer()
        display = VideoDisplay("Autonomous Vision") if self.config.video.show_window else None
        
        print("="*60 + "\n🟢 SYSTEM RUNNING | Press 'q' to quit\n" + "="*60)
        
        try:
            while video_mgr.is_opened():
                frame = video_mgr.read()
                if frame is None: break
                
                processed_frame, _, _, _ = self.process_frame(frame)
                
                self.hud.update_fps()
                if self.config.video.save_output: video_mgr.write(processed_frame)
                if display and not display.show(processed_frame): break
                
                if video_mgr.frame_count % 300 == 0:
                    self.predictor.cleanup_old_tracks()
        
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        
        finally:
            video_mgr.release()
            if display: display.close()
            
            print("\n" + "="*60 + "\n📊 FINAL STATISTICS\n" + "="*60)
            print(f"Total frames: {video_mgr.frame_count}")
            print(f"Average FPS: {self.hud.fps:.2f}")
            print(f"AI inferences: {self.analyzer.analysis_count}")
            print(f"Cost: $0.00 (LOCAL)")
            print("="*60)


if __name__ == "__main__":
    config = SystemConfig()
    # Exemplo: config.video.source = "video.mp4"
    system = AutonomousVisionSystem(config)
    system.run()