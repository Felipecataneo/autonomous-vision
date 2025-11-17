import cv2
import time
import base64
import openai
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime
from collections import defaultdict, deque
import json
from filterpy.kalman import KalmanFilter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ================================================================
# PREDITOR DE TRAJETÓRIAS COM KALMAN FILTER
# ================================================================

class KalmanTrajectoryPredictor:
    """Predição realista usando Kalman Filter ao invés de movimento linear"""
    
    def __init__(self):
        self.filters = {}
        self.last_update = {}
    
    def get_filter(self, track_id):
        if track_id not in self.filters:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            
            # Estado: [x, y, vx, vy]
            kf.F = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)
            
            # Matriz de observação (só medimos x, y)
            kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=float)
            
            # Ruído de medição (confiança na detecção)
            kf.R = np.eye(2) * 10
            
            # Ruído de processo (modelo de movimento)
            kf.Q = np.eye(4) * 0.1
            
            # Covariância inicial
            kf.P *= 1000
            
            self.filters[track_id] = kf
        
        return self.filters[track_id]
    
    def update(self, track_id, position):
        """Atualiza filtro com nova medição"""
        kf = self.get_filter(track_id)
        
        # Primeira medição: inicializa estado
        if track_id not in self.last_update:
            kf.x = np.array([position[0], position[1], 0, 0], dtype=float)
        else:
            kf.predict()
            kf.update(np.array(position, dtype=float))
        
        self.last_update[track_id] = time.time()
    
    def predict_trajectory(self, track_id, steps=60, step_size=5):
        """Prediz trajetória futura usando o modelo de Kalman"""
        if track_id not in self.filters:
            return []
        
        kf = self.get_filter(track_id)
        trajectory = []
        
        # Copia estado atual
        state = kf.x.copy()
        F = kf.F.copy()
        
        for i in range(0, steps, step_size):
            state = F @ state
            trajectory.append((int(state[0]), int(state[1])))
        
        return trajectory
    
    def get_velocity(self, track_id):
        """Retorna velocidade estimada"""
        if track_id not in self.filters:
            return (0, 0)
        
        kf = self.filters[track_id]
        return (float(kf.x[2]), float(kf.x[3]))

# ================================================================
# DETECTOR DE COLISÕES GEOMÉTRICO
# ================================================================

class CollisionDetector:
    """Detecta colisões reais entre trajetórias"""
    
    @staticmethod
    def will_collide(traj1, traj2, threshold=50):
        """
        Verifica se duas trajetórias vão colidir
        Retorna: (vai_colidir, tempo_ate_colisao, ponto_colisao)
        """
        if not traj1 or not traj2:
            return False, None, None
        
        min_dist = float('inf')
        collision_time = None
        collision_point = None
        
        # Verifica distância em cada passo temporal
        for i, (p1, p2) in enumerate(zip(traj1, traj2)):
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                collision_time = i
                collision_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        
        will_collide = min_dist < threshold
        return will_collide, collision_time, collision_point if will_collide else None
    
    @staticmethod
    def check_all_collisions(detections):
        """Verifica todas as combinações de trajetórias"""
        collisions = []
        
        for i, det1 in enumerate(detections):
            for det2 in detections[i+1:]:
                will_collide, time_step, point = CollisionDetector.will_collide(
                    det1.get('trajectory', []),
                    det2.get('trajectory', []),
                    threshold=60
                )
                
                if will_collide:
                    collisions.append({
                        'objects': [det1['id'], det2['id']],
                        'time': time_step,
                        'point': point,
                        'labels': [det1['label'], det2['label']]
                    })
        
        return collisions

# ================================================================
# ANALISADOR COM OPENAI (SEM LOOP DESNECESSÁRIO)
# ================================================================

class AutonomousAnalyzer:
    def __init__(self, api_key, analysis_interval=5.0):
        self.client = openai.OpenAI(api_key=api_key)
        self.last_analysis_time = 0
        self.analysis_interval = analysis_interval  # Aumentado para 5s
        self.predictions = {}
        self.analysis_count = 0
    
    def encode_frame(self, frame, max_size=640):  # Reduzido de 800
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Reduzido de 80
        return base64.b64encode(buffer).decode('utf-8')
    
    def should_analyze(self, detections, collisions):
        """Decide se deve chamar API baseado em condições"""
        now = time.time()
        
        # Não analisa se intervalo não passou
        if now - self.last_analysis_time < self.analysis_interval:
            return False
        
        # Só analisa se houver algo relevante
        if not detections:
            return False
        
        # Analisa sempre se detectar colisão iminente
        if collisions:
            return True
        
        # Analisa se tiver objetos em movimento
        moving_objects = sum(1 for d in detections if np.linalg.norm(d.get('velocity', (0,0))) > 2)
        if moving_objects > 0:
            return True
        
        return False
    
    def analyze_scene(self, frame, detections, collisions):
        """Análise contextual SOMENTE quando necessário"""
        
        if not self.should_analyze(detections, collisions):
            return None
        
        self.last_analysis_time = time.time()
        self.analysis_count += 1
        
        # Contexto compacto
        context = []
        for d in detections[:8]:  # Aumentado de 6
            vel = d.get('velocity', (0, 0))
            speed = np.linalg.norm(vel)
            context.append(
                f"{d['label']} ID{d['id']}: pos={d['center']}, v={speed:.1f}px/f"
            )
        
        # Contexto de colisões
        collision_info = ""
        if collisions:
            collision_info = f"\n⚠ COLLISION DETECTED: {len(collisions)} potential collisions\n"
            for c in collisions[:2]:
                collision_info += f"- {c['labels'][0]} ID{c['objects'][0]} vs {c['labels'][1]} ID{c['objects'][1]} in {c['time']*5//30:.1f}s\n"
        
        try:
            img_b64 = self.encode_frame(frame)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-11-20",  # Modelo mais recente
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}",
                                    "detail": "low"
                                }
                            },
                            {
                                "type": "text",
                                "text": f"""Autonomous vehicle vision system. Analyze risks briefly.

Objects tracked:
{chr(10).join(context)}
{collision_info}

Return ONLY JSON (no markdown):
{{
  "scene_type": "urban/highway/parking/residential",
  "risk_level": "low/medium/high/critical",
  "critical_objects": [list of IDs at risk],
  "collision_risk": "brief description (10 words max)",
  "predicted_events": ["event in 1-3s (8 words max)"],
  "attention_zones": [{{"x": int, "y": int, "radius": int, "type": "danger/caution"}}]
}}

Focus: collision trajectories, sudden movements, priority: pedestrians > vehicles."""
                            }
                        ]
                    }
                ],
                max_tokens=250,  # Reduzido de 300
                temperature=0.1  # Mais determinístico
            )
            
            text = response.choices[0].message.content.strip()
            
            # Limpa markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(text)
            self.predictions = result
            
            print(f"🧠 API Call #{self.analysis_count} | Risk: {result.get('risk_level', 'N/A').upper()}")
            
            return result
            
        except Exception as e:
            print(f"⚠ Analysis error: {e}")
            return None

# ================================================================
# HUD MODERNIZADO
# ================================================================

class AutonomousVisionHUD:
    def __init__(self, model_path="yolo11n.pt"):  # YOLO11 por padrão
        print("🔍 Loading YOLO11...")
        
        # Tenta YOLO11, fallback para v8
        try:
            self.model = YOLO(model_path)
        except:
            print("⚠ YOLO11 not found, using YOLOv8n")
            self.model = YOLO("yolov8n.pt")
        
        print("🧠 Initializing AI systems...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found")
        
        self.analyzer = AutonomousAnalyzer(api_key, analysis_interval=5.0)
        self.predictor = KalmanTrajectoryPredictor()
        self.collision_detector = CollisionDetector()
        
        self.predictions = {}
        self.detected_collisions = []
        
        self.track_colors = {}
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def get_color(self, track_id):
        if track_id not in self.track_colors:
            np.random.seed(track_id)
            colors = [(0,255,255), (255,0,255), (255,255,0), (0,255,0), (255,128,0)]
            self.track_colors[track_id] = colors[track_id % len(colors)]
        return self.track_colors[track_id]
    
    def draw_collision_warning(self, frame, collision):
        """Desenha aviso de colisão iminente"""
        point = collision['point']
        time_s = collision['time'] * 5 / 30  # Converte frames para segundos
        
        # Círculo pulsante
        pulse = int(20 + 10 * np.sin(time.time() * 5))
        cv2.circle(frame, point, pulse, (0, 0, 255), 3)
        cv2.circle(frame, point, pulse + 5, (255, 255, 255), 1)
        
        # Texto de aviso
        text = f"COLLISION: {time_s:.1f}s"
        cv2.putText(frame, text, (point[0] - 60, point[1] - pulse - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Linha conectando objetos
        return frame
    
    def draw_trajectory(self, frame, pos, trajectory, color, is_critical=False):
        """Trajetória com gradiente"""
        if not trajectory:
            return
        
        points = np.array([pos] + trajectory, dtype=np.int32)
        
        for i in range(len(points) - 1):
            alpha = 1.0 - (i / len(points))
            overlay = frame.copy()
            
            thickness = 3 if is_critical else 2
            line_color = (0, 0, 255) if is_critical else color
            
            cv2.line(overlay, tuple(points[i]), tuple(points[i+1]), line_color, thickness)
            cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, frame)
        
        # Ponto final
        if len(trajectory) > 0:
            end = trajectory[-1]
            cv2.circle(frame, end, 6, (0, 0, 255) if is_critical else color, -1)
            cv2.circle(frame, end, 8, (255, 255, 255), 1)
            cv2.putText(frame, "t+2s", (end[0] + 10, end[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def draw_detection_box(self, frame, det, is_critical=False):
        """Box com informações"""
        x1, y1, x2, y2 = det['box']
        center = det['center']
        color = (0, 0, 255) if is_critical else self.get_color(det['id'])
        
        thickness = 3 if is_critical else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Cantos decorativos
        corner = 15
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 3)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, 3)
        
        # Centro
        cv2.circle(frame, center, 3, color, -1)
        cv2.circle(frame, center, 5, (255, 255, 255), 1)
        
        # Info compacta
        velocity = det.get('velocity', (0, 0))
        speed = np.linalg.norm(velocity)
        
        info = f"{det['label'].upper()} #{det['id']} | {speed:.1f}px/f"
        
        # Background
        (text_w, text_h), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, 2)
        
        cv2.putText(frame, info, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_hud(self, frame, detections, predictions, collisions):
        """HUD principal"""
        h, w = frame.shape[:2]
        
        # Header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (5, 5, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, 65), (w, 65), (0, 255, 255), 2)
        
        # Título
        cv2.putText(frame, "AUTONOMOUS VISION v2", (15, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Risk level
        risk = predictions.get('risk_level', 'unknown').upper()
        risk_colors = {'LOW': (0, 255, 0), 'MEDIUM': (0, 255, 255), 
                       'HIGH': (0, 165, 255), 'CRITICAL': (0, 0, 255)}
        risk_color = risk_colors.get(risk, (200, 200, 200))
        
        cv2.putText(frame, f"RISK: {risk}", (15, 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)
        
        # Stats direita
        stats = f"FPS: {self.fps:.1f} | OBJ: {len(detections)} | API: {self.analyzer.analysis_count}"
        cv2.putText(frame, stats, (w - 350, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        scene = predictions.get('scene_type', 'unknown').upper()
        cv2.putText(frame, f"SCENE: {scene}", (w - 350, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Collision warning crítico
        if collisions:
            warn_y = 75
            closest = min(collisions, key=lambda c: c['time'])
            time_s = closest['time'] * 5 / 30
            
            cv2.rectangle(overlay, (10, warn_y), (w - 10, warn_y + 30), (0, 0, 100), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (10, warn_y), (w - 10, warn_y + 30), (0, 0, 255), 3)
            
            text = f"⚠ COLLISION IMMINENT: {closest['labels'][0]} vs {closest['labels'][1]} in {time_s:.1f}s"
            cv2.putText(frame, text, (25, warn_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def run(self, source=0, save_output=False, show_window=True):
        print(f"📹 Starting: {source}")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open: {source}")
        
        writer = None
        if save_output:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = f"autonomous_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Saving to: {output_path}")
        
        print("✅ System active. Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS calculation
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed
            
            # YOLO tracking com configurações otimizadas
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.25,  # Threshold de confiança
                iou=0.45,   # IoU para NMS
                tracker="botsort.yaml"  # Tracker melhorado
            )
            
            boxes = results[0].boxes
            
            detections = []
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                
                for b, c, cf, tid in zip(xyxy, cls, conf, ids):
                    x1, y1, x2, y2 = b
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Atualiza Kalman Filter
                    self.predictor.update(tid, center)
                    
                    # Obtém velocidade e trajetória
                    velocity = self.predictor.get_velocity(tid)
                    trajectory = self.predictor.predict_trajectory(tid, steps=60, step_size=5)
                    
                    detections.append({
                        'box': b.tolist(),
                        'center': center,
                        'label': self.model.names[c],
                        'conf': float(cf),
                        'id': int(tid),
                        'velocity': velocity,
                        'trajectory': trajectory
                    })
            
            # Detecção geométrica de colisões
            self.detected_collisions = self.collision_detector.check_all_collisions(detections)
            
            # Análise com IA (SÓ quando necessário)
            new_predictions = self.analyzer.analyze_scene(frame, detections, self.detected_collisions)
            if new_predictions:
                self.predictions = new_predictions
            
            # Desenha avisos de colisão
            for collision in self.detected_collisions:
                self.draw_collision_warning(frame, collision)
            
            # Desenha detecções
            critical_ids = self.predictions.get('critical_objects', [])
            for det in detections:
                is_critical = str(det['id']) in critical_ids or det['id'] in critical_ids
                self.draw_detection_box(frame, det, is_critical)
                self.draw_trajectory(frame, det['center'], det['trajectory'], 
                                   self.get_color(det['id']), is_critical)
            
            # HUD
            frame = self.draw_hud(frame, detections, self.predictions, self.detected_collisions)
            
            if writer:
                writer.write(frame)
            
            if show_window:
                cv2.namedWindow("Autonomous Vision v2", cv2.WINDOW_NORMAL)
                cv2.imshow("Autonomous Vision v2", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if writer:
            writer.release()
            print(f"\n✅ Video saved: {output_path}")
        if show_window:
            cv2.destroyAllWindows()
        
        print(f"\n📊 Total API calls: {self.analyzer.analysis_count}")
        print(f"💰 Estimated cost: ${self.analyzer.analysis_count * 0.02:.2f}")

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    try:
        system = AutonomousVisionHUD()
        
        # Modo headless recomendado
        system.run("video.mp4", save_output=True, show_window=False)
        
    except KeyboardInterrupt:
        print("\n⏹ Interrupted")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()