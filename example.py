"""
Exemplos de uso do Autonomous Vision System
"""
from core.config import SystemConfig, YOLOConfig, AnalyzerConfig, VideoConfig
from main import AutonomousVisionSystem


def example_1_webcam():
    """Exemplo 1: Webcam em tempo real"""
    print("="*60)
    print("EXEMPLO 1: Webcam em tempo real")
    print("="*60)
    
    config = SystemConfig()
    config.video.source = "0"
    config.video.save_output = False
    config.video.show_window = True
    
    system = AutonomousVisionSystem(config)
    system.run()


def example_2_video_file():
    """Exemplo 2: Processa vídeo e salva resultado"""
    print("="*60)
    print("EXEMPLO 2: Processa vídeo e salva")
    print("="*60)
    
    config = SystemConfig()
    config.video.source = "video.mp4"
    config.video.save_output = True
    config.video.show_window = False  # Headless
    
    system = AutonomousVisionSystem(config)
    system.run()


def example_3_high_performance():
    """Exemplo 3: Configuração para máxima performance"""
    print("="*60)
    print("EXEMPLO 3: High Performance Mode")
    print("="*60)
    
    config = SystemConfig()
    
    # YOLO mais leve
    config.yolo.model_path = "yolo11n.pt"
    config.yolo.conf_threshold = 0.3  # Mais conservador
    
    # Kalman sem aceleração (mais rápido)
    config.kalman.use_acceleration = False
    
    # Análise menos frequente
    config.analyzer.analysis_interval = 5.0
    config.analyzer.adaptive_interval = False
    config.analyzer.image_max_size = 384
    config.analyzer.image_quality = 70
    
    # Threading ativo
    config.use_threading = True
    
    config.video.source = "0"
    
    system = AutonomousVisionSystem(config)
    system.run()


def example_4_high_accuracy():
    """Exemplo 4: Máxima precisão (sacrifica FPS)"""
    print("="*60)
    print("EXEMPLO 4: High Accuracy Mode")
    print("="*60)
    
    config = SystemConfig()
    
    # YOLO maior e mais preciso
    config.yolo.model_path = "yolo11m.pt"
    config.yolo.conf_threshold = 0.15  # Detecta mais objetos
    config.yolo.iou_threshold = 0.5
    
    # Kalman com aceleração
    config.kalman.use_acceleration = True
    config.kalman.measurement_noise = 5.0  # Mais confiança na medição
    
    # Análise frequente com adaptação
    config.analyzer.analysis_interval = 2.0
    config.analyzer.adaptive_interval = True
    config.analyzer.image_max_size = 640  # Imagem maior
    config.analyzer.image_quality = 95
    
    config.video.source = "video.mp4"
    config.video.save_output = True
    
    system = AutonomousVisionSystem(config)
    system.run()


def example_5_custom_tracker():
    """Exemplo 5: Tracker customizado"""
    print("="*60)
    print("EXEMPLO 5: Custom Tracker")
    print("="*60)
    
    config = SystemConfig()
    
    # Experimenta com BoT-SORT
    config.yolo.tracker = "botsort.yaml"
    
    config.video.source = "0"
    
    system = AutonomousVisionSystem(config)
    system.run()


def example_6_export_onnx():
    """Exemplo 6: Exporta modelo para ONNX"""
    print("="*60)
    print("EXEMPLO 6: Export YOLO to ONNX")
    print("="*60)
    
    from detection.yolo_detector import YOLODetector
    
    detector = YOLODetector("yolo11n.pt")
    detector.export_to_onnx("yolo11n.onnx")
    
    print("\n✅ Agora use:")
    print("   config.yolo.model_path = 'yolo11n.onnx'")


def example_7_test_components():
    """Exemplo 7: Testa componentes individualmente"""
    print("="*60)
    print("EXEMPLO 7: Test Individual Components")
    print("="*60)
    
    import cv2
    import numpy as np
    
    # Teste Kalman
    from prediction.kalman import KalmanTrajectoryPredictor
    
    predictor = KalmanTrajectoryPredictor(use_acceleration=True)
    
    # Simula movimento
    positions = [(100, 100), (110, 105), (120, 112), (130, 121)]
    
    for i, pos in enumerate(positions):
        predictor.update(track_id=1, position=pos)
        trajectory = predictor.predict_trajectory(1, steps=60, step_size=5)
        velocity = predictor.get_velocity(1)
        
        print(f"Frame {i}: pos={pos}, vel={velocity}, traj_len={len(trajectory)}")
    
    # Teste Collision Detector
    from prediction.collision import CollisionDetector
    
    detector = CollisionDetector(iou_threshold=0.3)
    
    det1 = {
        'id': 1,
        'box': [100, 100, 150, 150],
        'center': (125, 125),
        'label': 'car',
        'trajectory': [(130, 130), (135, 135), (140, 140)]
    }
    
    det2 = {
        'id': 2,
        'box': [200, 100, 250, 150],
        'center': (225, 125),
        'label': 'car',
        'trajectory': [(220, 130), (215, 135), (210, 140)]
    }
    
    collisions = detector.check_all_collisions([det1, det2], use_iou=True)
    
    print(f"\nCollisions detected: {len(collisions)}")
    for c in collisions:
        print(f"  {c['labels'][0]} vs {c['labels'][1']} in {c['time']} frames")


def example_8_benchmark():
    """Exemplo 8: Benchmark de diferentes configurações"""
    print("="*60)
    print("EXEMPLO 8: Benchmark Mode")
    print("="*60)
    
    import time
    
    configs = [
        ("YOLO11n + No Accel", {"model": "yolo11n.pt", "accel": False}),
        ("YOLO11n + Accel", {"model": "yolo11n.pt", "accel": True}),
        ("YOLO11s + Accel", {"model": "yolo11s.pt", "accel": True}),
    ]
    
    for name, params in configs:
        print(f"\n🔍 Testing: {name}")
        
        config = SystemConfig()
        config.yolo.model_path = params["model"]
        config.kalman.use_acceleration = params["accel"]
        config.video.source = "video.mp4"
        config.video.save_output = False
        config.video.show_window = False
        config.analyzer.analysis_interval = 999  # Desabilita análise
        
        try:
            system = AutonomousVisionSystem(config)
            
            start = time.time()
            # Processa apenas 300 frames para teste
            # TODO: Implementar limite de frames no VideoManager
            system.run()
            elapsed = time.time() - start
            
            fps = system.hud.fps
            print(f"   FPS: {fps:.2f} | Time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    examples = {
        "1": example_1_webcam,
        "2": example_2_video_file,
        "3": example_3_high_performance,
        "4": example_4_high_accuracy,
        "5": example_5_custom_tracker,
        "6": example_6_export_onnx,
        "7": example_7_test_components,
        "8": example_8_benchmark,
    }
    
    if len(sys.argv) < 2:
        print("\n🚀 Autonomous Vision System - Examples\n")
        print("Usage: python example.py <number>\n")
        print("Available examples:")
        for num, func in examples.items():
            doc = func.__doc__ or "No description"
            print(f"  {num}. {doc}")
        print()
    else:
        example_num = sys.argv[1]
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"❌ Example {example_num} not found")
            print(f"Available: {', '.join(examples.keys())}")