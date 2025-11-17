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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ================================================================
# ANALISADOR DE TRAJETÓRIAS E PREDIÇÕES
# ================================================================
class AutonomousAnalyzer:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.last_time = 0
        self.interval = 3
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # 30 frames de histórico
        self.predictions = {}
        
    def encode_frame(self, frame, max_size=800):
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')
    
    def calculate_velocity(self, track_id, current_pos):
        """Calcula velocidade baseado no histórico"""
        history = self.track_history[track_id]
        history.append(current_pos)
        
        if len(history) < 2:
            return (0, 0)
        
        # Velocidade média dos últimos 10 frames
        recent = list(history)[-10:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        
        return (dx / len(recent), dy / len(recent))
    
    def predict_trajectory(self, pos, velocity, steps=60):
        """Prediz trajetória futura (60 frames = ~2s a 30fps)"""
        trajectory = []
        x, y = pos
        vx, vy = velocity
        
        for i in range(0, steps, 5):  # A cada 5 frames
            future_x = int(x + vx * i)
            future_y = int(y + vy * i)
            trajectory.append((future_x, future_y))
        
        return trajectory
    
    def analyze_scene(self, frame, detections):
        """Análise contextual com IA"""
        now = time.time()
        if now - self.last_time < self.interval or not detections:
            return None
        
        self.last_time = now
        
        # Contexto para a IA
        context = []
        for d in detections[:6]:
            vel = d.get('velocity', (0, 0))
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            context.append(
                f"- {d['label']} ID:{d['id']} pos:{d['center']} velocidade:{speed:.1f}px/frame"
            )
        
        try:
            img_b64 = self.encode_frame(frame)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
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
                                "text": f"""Você é o sistema de visão de um carro autônomo. Analise a cena e identifique riscos.

Objetos detectados com velocidade:
{chr(10).join(context)}

Retorne APENAS JSON (sem markdown):
{{
  "scene_type": "urban/highway/parking/pedestrian",
  "risk_level": "low/medium/high/critical",
  "critical_objects": ["ID do objeto em risco"],
  "collision_risk": "descrição curta do risco (max 15 palavras)",
  "predicted_events": ["evento que vai acontecer em 1-3s (max 12 palavras)"],
  "attention_zones": [
    {{"x": 100, "y": 200, "radius": 50, "type": "danger/caution/interest"}}
  ]
}}

Foque em: trajetórias de colisão, comportamentos inesperados, objetos em movimento rápido."""
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            text = response.choices[0].message.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(text)
            self.predictions = result
            return result
            
        except Exception as e:
            print(f"⚠ Erro na análise: {e}")
            return None


# ================================================================
# HUD ESTILO AUTONOMOUS VEHICLE
# ================================================================
class AutonomousVisionHUD:
    def __init__(self, model_path="yolov8n.pt"):
        print("🔍 Carregando YOLO...")
        self.model = YOLO(model_path)
        
        print("🧠 Inicializando Autonomous Vision System...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY não encontrada")
        
        self.analyzer = AutonomousAnalyzer(api_key)
        self.predictions = {}
        
        self.track_colors = {}
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def get_color(self, track_id):
        if track_id not in self.track_colors:
            np.random.seed(track_id)
            colors = [(0,255,255), (255,0,255), (255,255,0), (0,255,0)]
            self.track_colors[track_id] = colors[track_id % len(colors)]
        return self.track_colors[track_id]
    
    def draw_trajectory(self, frame, pos, trajectory, color, is_critical=False):
        """Desenha trajetória predita estilo Tesla"""
        if not trajectory:
            return
        
        # Linha de trajetória
        points = np.array([pos] + trajectory, dtype=np.int32)
        
        # Gradiente de opacidade
        for i in range(len(points) - 1):
            alpha = 1.0 - (i / len(points))
            overlay = frame.copy()
            
            thickness = 3 if is_critical else 2
            line_color = (0, 0, 255) if is_critical else color
            
            cv2.line(overlay, tuple(points[i]), tuple(points[i+1]), line_color, thickness)
            cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)
        
        # Ponto final com tempo
        if len(trajectory) > 0:
            end = trajectory[-1]
            cv2.circle(frame, end, 8, (0, 0, 255) if is_critical else color, -1)
            cv2.circle(frame, end, 10, (255, 255, 255), 1)
            cv2.putText(frame, "t+2s", (end[0] + 12, end[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_attention_zones(self, frame, zones):
        """Desenha zonas de atenção/perigo"""
        if not zones:
            return
        
        overlay = frame.copy()
        
        for zone in zones:
            x, y = zone.get('x', 0), zone.get('y', 0)
            radius = zone.get('radius', 50)
            zone_type = zone.get('type', 'interest')
            
            if zone_type == 'danger':
                color = (0, 0, 255)
                alpha = 0.4
            elif zone_type == 'caution':
                color = (0, 165, 255)
                alpha = 0.3
            else:
                color = (0, 255, 255)
                alpha = 0.2
            
            cv2.circle(overlay, (x, y), radius, color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.circle(frame, (x, y), radius, color, 2)
    
    def draw_detection_box(self, frame, det, is_critical=False):
        """Box estilo autonomous vehicle"""
        x1, y1, x2, y2 = det['box']
        center = det['center']
        color = (0, 0, 255) if is_critical else self.get_color(det['id'])
        
        # Box principal
        thickness = 3 if is_critical else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Cantos decorativos
        corner = 20
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 4)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, 4)
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, 4)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, 4)
        
        # Centro + ID
        cv2.circle(frame, center, 4, color, -1)
        cv2.circle(frame, center, 6, (255, 255, 255), 1)
        
        # Info box
        velocity = det.get('velocity', (0, 0))
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        info = [
            f"{det['label'].upper()} #{det['id']}",
            f"{det['conf']:.2f}",
            f"{speed:.1f}px/f"
        ]
        
        # Background
        box_h = 20 + len(info) * 18
        cv2.rectangle(frame, (x1, y1 - box_h - 5), (x1 + 150, y1), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1 - box_h - 5), (x1 + 150, y1), color, 2)
        
        # Texto
        for i, text in enumerate(info):
            cv2.putText(frame, text, (x1 + 8, y1 - box_h + 20 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    def draw_timeline(self, frame, predictions):
        """Timeline de eventos preditos"""
        h, w = frame.shape[:2]
        
        events = predictions.get('predicted_events', [])
        if not events:
            return
        
        # Background
        timeline_h = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, h - timeline_h - 20), (w - 20, h - 20), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (20, h - timeline_h - 20), (w - 20, h - 20), (0, 255, 255), 2)
        
        # Título
        cv2.putText(frame, "PREDICTED EVENTS", (35, h - timeline_h),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Eventos
        for i, event in enumerate(events[:2]):
            icon = "▸"
            cv2.putText(frame, f"{icon} t+{i+1}s: {event}", 
                       (35, h - timeline_h + 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_hud(self, frame, detections, predictions):
        """HUD principal"""
        h, w = frame.shape[:2]
        
        # Header minimalista
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (5, 5, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.line(frame, (0, 70), (w, 70), (0, 255, 255), 2)
        
        # Status
        risk = predictions.get('risk_level', 'unknown').upper()
        risk_colors = {'LOW': (0, 255, 0), 'MEDIUM': (0, 255, 255), 
                       'HIGH': (0, 165, 255), 'CRITICAL': (0, 0, 255)}
        risk_color = risk_colors.get(risk, (200, 200, 200))
        
        cv2.putText(frame, "AUTONOMOUS VISION", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(frame, f"RISK: {risk}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
        
        # Stats
        stats = f"FPS: {self.fps:.1f} | OBJECTS: {len(detections)}"
        cv2.putText(frame, stats, (w - 280, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        scene = predictions.get('scene_type', 'unknown').upper()
        cv2.putText(frame, f"SCENE: {scene}", (w - 280, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Collision warning
        collision = predictions.get('collision_risk')
        if collision:
            warn_y = 90
            cv2.rectangle(overlay, (15, warn_y), (w - 15, warn_y + 35), (0, 0, 100), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (15, warn_y), (w - 15, warn_y + 35), (0, 0, 255), 3)
            cv2.putText(frame, f"⚠ COLLISION RISK: {collision}", (30, warn_y + 23),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def run(self, source=0, save_output=False, show_window=True):
        print(f"📹 Iniciando: {source}")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"❌ Erro ao abrir: {source}")
        
        writer = None
        if save_output:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = f"autonomous_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Salvando em: {output_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
        
        if not show_window:
            print("⚠ Modo headless ativado")
        
        print("✅ Sistema ativo. Pressione 'q' para sair.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed
                
                if not show_window and total_frames > 0:
                    progress = (self.frame_count / total_frames) * 100
                    bar_len = 40
                    filled = int(bar_len * self.frame_count / total_frames)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    print(f"\r⏳ [{bar}] {progress:.1f}% | {self.frame_count}/{total_frames} | {self.fps:.1f} FPS", end='', flush=True)
            
            # YOLO
            results = self.model.track(frame, persist=True, verbose=False)
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
                    
                    # Calcula velocidade e trajetória
                    velocity = self.analyzer.calculate_velocity(tid, center)
                    trajectory = self.analyzer.predict_trajectory(center, velocity)
                    
                    detections.append({
                        'box': b.tolist(),
                        'center': center,
                        'label': self.model.names[c],
                        'conf': float(cf),
                        'id': int(tid),
                        'velocity': velocity,
                        'trajectory': trajectory
                    })
            
            # Análise com IA
            new_predictions = self.analyzer.analyze_scene(frame, detections)
            if new_predictions:
                self.predictions = new_predictions
                print(f"\n🧠 Risco: {new_predictions.get('risk_level', 'N/A').upper()}")
            
            # Desenha zonas de atenção
            zones = self.predictions.get('attention_zones', [])
            self.draw_attention_zones(frame, zones)
            
            # Desenha detecções e trajetórias
            critical_ids = self.predictions.get('critical_objects', [])
            for det in detections:
                is_critical = str(det['id']) in critical_ids or det['id'] in critical_ids
                self.draw_detection_box(frame, det, is_critical)
                self.draw_trajectory(frame, det['center'], det['trajectory'], 
                                   self.get_color(det['id']), is_critical)
            
            # HUD e timeline
            frame = self.draw_hud(frame, detections, self.predictions)
            self.draw_timeline(frame, self.predictions)
            
            if writer:
                writer.write(frame)
            
            if show_window:
                cv2.namedWindow("Autonomous Vision System", cv2.WINDOW_NORMAL)
                cv2.imshow("Autonomous Vision System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if writer:
            writer.release()
            print(f"\n✅ Vídeo salvo: {output_path}")
        if show_window:
            cv2.destroyAllWindows()


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    try:
        system = AutonomousVisionHUD()
        
        # MODO HEADLESS (recomendado)
        system.run("video.mp4", save_output=True, show_window=False)
        
    except KeyboardInterrupt:
        print("\n⏹ Interrompido")
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()