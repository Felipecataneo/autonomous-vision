"""
Analisador de cena com Ollama + Qwen3-VL (CORRIGIDO)
"""
import cv2
import time
import base64
import json
import re
import numpy as np
import requests
from typing import List, Dict, Optional
from queue import Queue
import threading


class OllamaVisionAnalyzer:
    """Análise de cena com Qwen3-VL via Ollama (local)"""
    
    def __init__(
        self,
        model: str = "qwen3-vl:8b",
        base_url: str = "http://localhost:11434",
        analysis_interval: float = 3.0,
        adaptive_interval: bool = True,
        max_timeout: int = 45,
        image_max_size: int = 512,
        image_quality: int = 80,
        use_threading: bool = True
    ):
        self.model = model
        self.base_url = base_url
        self.analysis_interval = analysis_interval
        self.adaptive_interval = adaptive_interval
        self.max_timeout = max_timeout
        self.image_max_size = image_max_size
        self.image_quality = image_quality
        
        self.last_analysis_time = 0
        self.predictions = {}
        self.analysis_count = 0
        self.current_risk = "unknown"
        
        # Threading
        self.use_threading = use_threading
        if use_threading:
            self.queue = Queue(maxsize=1)
            self.result_queue = Queue()
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
        
        self._check_ollama()
    
    def _check_ollama(self):
        """Verifica se Ollama está rodando e modelo instalado"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                
                if not any('qwen3-vl' in m.lower() for m in models):
                    print("⚠ Qwen3-VL não encontrado. Instale com:")
                    print("   ollama pull qwen3-vl:8b")
                    raise RuntimeError("Modelo não instalado")
                
                print(f"✅ Ollama conectado | Modelos: {len(models)}")
            else:
                raise RuntimeError("Ollama não responde")
                
        except requests.exceptions.ConnectionError:
            print("❌ Ollama offline. Inicie com: ollama serve")
            raise RuntimeError("Ollama offline")
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame para base64 com otimizações"""
        if frame is None or frame.size == 0:
            print("⚠ Frame vazio")
            return ""
        
        h, w = frame.shape[:2]
        
        # Redimensiona se necessário
        if max(h, w) > self.image_max_size:
            scale = self.image_max_size / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Encode WebP (melhor compressão que JPEG)
        success, buffer = cv2.imencode(
            '.webp', 
            frame, 
            [cv2.IMWRITE_WEBP_QUALITY, self.image_quality]
        )
        
        if not success:
            # Fallback para JPEG
            success, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            )
        
        b64 = base64.b64encode(buffer).decode('utf-8')
        return b64
    
    def should_analyze(self, detections: List[Dict], collisions: List[Dict]) -> bool:
        """Decide se deve executar análise"""
        now = time.time()
        
        # Intervalo adaptativo baseado em risco
        if self.adaptive_interval:
            if self.current_risk == "critical":
                interval = 0.5
            elif self.current_risk == "high":
                interval = 1.5
            elif self.current_risk == "medium":
                interval = 2.5
            else:
                interval = self.analysis_interval
        else:
            interval = self.analysis_interval
        
        if now - self.last_analysis_time < interval:
            return False
        
        if not detections:
            return False
        
        # Sempre analisa se houver colisão
        if collisions:
            return True
        
        # Analisa se houver movimento significativo
        moving = sum(1 for d in detections if np.linalg.norm(d.get('velocity', (0,0))) > 2)
        if moving > 0:
            return True
        
        # Analisa se houver objetos próximos à borda
        # (podem colidir em breve ao entrar no frame)
        return False
    
    def _parse_ollama_response(self, response_json: dict) -> Optional[Dict]:
        """
        Parse robusto de resposta do Ollama
        Trata casos onde JSON vem em 'response' ou 'thinking'
        """
        # Tenta pegar response direto
        result_text = response_json.get('response', '').strip()
        
        # Se vazio, extrai do thinking
        if not result_text:
            thinking = response_json.get('thinking', '')
            
            if not thinking:
                print("   ⚠ Response e thinking vazios")
                return None
            
            # Tenta extrair JSON do thinking via regex
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', thinking, re.DOTALL)
            
            if match:
                result_text = match.group(0)
                print(f"   🔧 JSON extraído do thinking")
            else:
                # Fallback: constrói JSON baseado em keywords
                print(f"   🤖 Construindo JSON de texto natural")
                
                thinking_lower = thinking.lower()
                
                # Detecta scene_type
                if "highway" in thinking_lower or "freeway" in thinking_lower:
                    scene = "highway"
                elif "residential" in thinking_lower or "neighborhood" in thinking_lower:
                    scene = "residential"
                else:
                    scene = "urban"
                
                # Detecta risk_level
                if "critical" in thinking_lower or "danger" in thinking_lower:
                    risk = "critical"
                elif "high" in thinking_lower:
                    risk = "high"
                elif "medium" in thinking_lower or "moderate" in thinking_lower:
                    risk = "medium"
                else:
                    risk = "low"
                
                # Extrai IDs críticos
                ids = re.findall(r'(?:ID|id|#)\s*(\d+)', thinking)
                critical = list(set(int(x) for x in ids))[:3]
                
                # Extrai descrição de risco (última frase)
                sentences = [s.strip() for s in thinking.split('.') if s.strip()]
                collision_desc = sentences[-1][:80] if sentences else "Monitoring scene"
                
                result_text = json.dumps({
                    "scene_type": scene,
                    "risk_level": risk,
                    "critical_objects": critical,
                    "collision_risk": collision_desc
                })
        
        # Remove markdown se existir
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        try:
            result = json.loads(result_text)
            
            # Validação de campos obrigatórios
            required = ['scene_type', 'risk_level']
            if not all(k in result for k in required):
                print(f"   ⚠ Campos faltando: {required}")
                return None
            
            # Campos opcionais com defaults
            result.setdefault('critical_objects', [])
            result.setdefault('collision_risk', 'No immediate risk')
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"   ⚠ JSON inválido: {e}")
            print(f"   Raw: {result_text[:150]}...")
            return None
    
    def _analyze_sync(
        self, 
        frame: np.ndarray, 
        detections: List[Dict], 
        collisions: List[Dict]
    ) -> Optional[Dict]:
        """Análise síncrona (bloqueante)"""
        
        # Contexto compacto
        context = []
        for d in detections[:8]:
            vel = d.get('velocity', (0, 0))
            speed = np.linalg.norm(vel)
            context.append(
                f"{d['label']} ID{d['id']}: pos={d['center']}, speed={speed:.1f}px/f"
            )
        
        # Info de colisões
        collision_info = ""
        if collisions:
            collision_info = f"\n⚠ {len(collisions)} COLLISION(S) DETECTED:\n"
            for c in collisions[:2]:
                time_s = c['time'] * 5 / 30
                collision_info += f"- {c['labels'][0]} vs {c['labels'][1]} in {time_s:.1f}s (severity: {c.get('severity', 'unknown')})\n"
        
        try:
            img_b64 = self.encode_frame(frame)
            
            if not img_b64:
                return None
            
            # Prompt otimizado
            prompt = f"""Analyze this driving scene. Detected: {len(detections)} objects.
{collision_info if collisions else "No collisions detected."}

Objects: {', '.join(context[:5])}

Return ONLY valid JSON (no markdown, no thinking):
{{
  "scene_type": "urban|highway|residential",
  "risk_level": "low|medium|high|critical",
  "critical_objects": [1, 2],
  "collision_risk": "brief description"
}}"""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 400,
                    "num_ctx": 4096,
                    "stop": ["</think>", "```"]
                }
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.max_timeout
            )
            inference_time = time.time() - start_time
            
            if response.status_code != 200:
                print(f"   ⚠ Ollama error: {response.status_code}")
                return None
            
            # Parse com tratamento robusto
            result = self._parse_ollama_response(response.json())
            
            if result:
                self.current_risk = result.get('risk_level', 'unknown')
                self.predictions = result
                self.analysis_count += 1
                
                print(f"🧠 Analysis #{self.analysis_count} ({inference_time:.2f}s) | Risk: {result['risk_level'].upper()}")
            
            return result
            
        except requests.exceptions.Timeout:
            print(f"   ⚠ Timeout (>{self.max_timeout}s)")
            return None
        except Exception as e:
            print(f"   ⚠ Error: {e}")
            return None
    
    def _worker(self):
        """Thread worker para análise assíncrona"""
        while True:
            frame, detections, collisions = self.queue.get()
            result = self._analyze_sync(frame, detections, collisions)
            if result:
                self.result_queue.put(result)
    
    def analyze_scene(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        collisions: List[Dict]
    ) -> Optional[Dict]:
        """
        Analisa cena (síncrono ou assíncrono)
        
        Returns:
            Dict com análise ou None
        """
        if not self.should_analyze(detections, collisions):
            return None
        
        self.last_analysis_time = time.time()
        
        if self.use_threading:
            # Análise assíncrona
            if self.queue.full():
                return None  # Pula se já tem análise rodando
            
            self.queue.put((frame.copy(), detections, collisions))
            
            # Retorna resultado anterior se disponível
            if not self.result_queue.empty():
                return self.result_queue.get()
            
            return None
        else:
            # Análise síncrona (bloqueia)
            return self._analyze_sync(frame, detections, collisions)
    
    def get_latest_prediction(self) -> Dict:
        """Retorna última predição válida"""
        return self.predictions