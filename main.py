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
        """
        Args:
            config: Configuração do sistema (usa defaults se None)
        """
        if config is None:
            config = SystemConfig()
        
        self.config = config
        
        print("="*60)
        print("🚗 AUTONOMOUS VISION SYSTEM v2.0")
        print("="*60)
        
        # Inicializa componentes
        self._init_components()
    
    def _init_components(self):
        """Inicializa todos os componentes"""
        
        # Detector YOLO
        print("\n[1/5] Initializing YOLO Detector...")
        self.detector = YOLODetector(
            model_path=self.config.yolo.model_path,
            conf_threshold=self.config.yolo.conf_threshold,
            iou_threshold=self.config.yolo.iou_threshold,
            tracker=self.config.yolo.tracker,
            device=self.config.yolo.device
        )
        
        # Preditor Kalman
        print("\n[2/5] Initializing Kalman Predictor...")
        self.predictor = KalmanTrajectoryPredictor(
            use_acceleration=self.config.kalman.use_acceleration,
            measurement_noise=self.config.kalman.measurement_noise,
            process_noise=self.config.kalman.process_noise,
            initial_covariance=self.config.kalman.initial_covariance
        )
        print(f"   Mode: {'Acceleration' if self.config.kalman.use_acceleration else 'Velocity'}")
        
        # Detector de colisão
        print("\n[3/5] Initializing Collision Detector...")
        self.collision_detector = CollisionDetector(
            iou_threshold=self.config.collision.iou_threshold
        )
        
        # Analisador de cena
        print("\n[4/5] Initializing Scene Analyzer...")
        if self.config.analyzer.provider == "ollama":
            self.analyzer = OllamaVisionAnalyzer(
                model=self.config.analyzer.model,
                base_url=self.config.analyzer.base_url,
                analysis_interval=self.config.analyzer.analysis_interval,
                adaptive_interval=self.config.analyzer.adaptive_interval,
                max_timeout=self.config.analyzer.max_timeout,
                image_max_size=self.config.analyzer.image_max_size,
                image_quality=self.config.analyzer.image_quality,
                use_threading=self.config.use_threading
            )
        else:
            raise NotImplementedError(f"Provider '{self.config.analyzer.provider}' not implemented")
        
        # HUD
        print("\n[5/5] Initializing HUD...")
        self.hud = HUDDrawer()
        
        print("\n✅ All systems ready\n")
    
    def process_frame(self, frame):
        """
        Processa um frame completo
        
        Returns:
            frame_processado, detections, collisions, predictions
        """
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
        collisions = self.collision_detector.check_all_collisions(
            detections,
            use_iou=True
        )
        
        # 4. Análise de cena com IA
        new_predictions = self.analyzer.analyze_scene(frame, detections, collisions)
        predictions = new_predictions if new_predictions else self.analyzer.get_latest_prediction()
        
        # 5. Desenha visualizações
        critical_ids = predictions.get('critical_objects', [])
        
        # Avisos de colisão
        for collision in collisions:
            self.hud.draw_collision_warning(frame, collision)
        
        # Detecções e trajetórias
        for det in detections:
            is_critical = str(det['id']) in critical_ids or det['id'] in critical_ids
            
            self.hud.draw_detection_box(frame, det, is_critical)
            self.hud.draw_trajectory(
                frame,
                det['center'],
                det['trajectory'],
                self.hud.get_track_color(det['id']),
                is_critical
            )
            
            # Opcional: desenha elipse de incerteza
            # uncertainty = self.predictor.get_uncertainty(det['id'])
            # self.hud.draw_uncertainty_ellipse(frame, det['center'], uncertainty)
        
        # HUD principal
        frame = self.hud.draw_main_hud(
            frame,
            detections,
            predictions,
            collisions,
            self.analyzer.analysis_count
        )
        
        return frame, detections, collisions, predictions
    
    def run(self):
        """Loop principal do sistema"""
        
        # Setup vídeo
        video_mgr = VideoManager(self.config.video.source)
        if not video_mgr.open():
            return
        
        if self.config.video.save_output:
            video_mgr.setup_writer()
        
        display = None
        if self.config.video.show_window:
            display = VideoDisplay("Autonomous Vision - Qwen3-VL")
        
        print("="*60)
        print("🎬 SYSTEM RUNNING")
        print("="*60)
        print("Press 'q' to quit\n")
        
        try:
            while video_mgr.is_opened():
                frame = video_mgr.read()
                if frame is None:
                    break
                
                # Processa frame
                processed_frame, detections, collisions, predictions = self.process_frame(frame)
                
                # Atualiza FPS
                self.hud.update_fps()
                
                # Salva se configurado
                if self.config.video.save_output:
                    video_mgr.write(processed_frame)
                
                # Mostra se configurado
                if display:
                    if not display.show(processed_frame):
                        break
                
                # Cleanup de tracks antigos periodicamente
                if video_mgr.frame_count % 300 == 0:
                    self.predictor.cleanup_old_tracks()
        
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        
        finally:
            # Cleanup
            video_mgr.release()
            if display:
                display.close()
            
            # Stats finais
            print("\n" + "="*60)
            print("📊 FINAL STATISTICS")
            print("="*60)
            print(f"Total frames: {video_mgr.frame_count}")
            print(f"Average FPS: {self.hud.fps:.2f}")
            print(f"AI inferences: {self.analyzer.analysis_count}")
            print(f"Cost: $0.00 (LOCAL)")
            print("="*60)


def main():
    """Entry point"""
    
    # Configuração customizada (opcional)
    config = SystemConfig()
    
    # Exemplos de customização:
    # config.yolo.model_path = "yolo11n.pt"
    # config.analyzer.analysis_interval = 2.0
    # config.video.source = "video.mp4"
    # config.video.save_output = True
    # config.video.show_window = False
    
    # Cria e executa sistema
    system = AutonomousVisionSystem(config)
    system.run()


if __name__ == "__main__":
    main()