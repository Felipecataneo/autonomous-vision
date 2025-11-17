"""
Analisador de cena com Ollama + Qwen3-VL (CORRIGIDO E ROBUSTO)
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
        image_max_size: int = 768, # Mantendo a configuração agressiva
        image_quality: int = 90,
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
                    print("🔴 Qwen3-VL não encontrado. Instale com: ollama pull qwen3-vl:8b")
                    raise RuntimeError("Modelo não instalado")
                print(f"✅ Ollama conectado | Modelos: {len(models)}")
            else:
                raise RuntimeError("Ollama não responde")
        except requests.exceptions.ConnectionError:
            print("❌ Ollama offline. Inicie com: ollama serve")
            raise RuntimeError("Ollama offline")
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame para base64 com otimizações"""
        if frame is None or frame.size == 0: return ""
        
        h, w = frame.shape[:2]
        if max(h, w) > self.image_max_size:
            scale = self.image_max_size / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        success, buffer = cv2.imencode('.webp', frame, [cv2.IMWRITE_WEBP_QUALITY, self.image_quality])
        if not success:
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        
        return base64.b64encode(buffer).decode('utf-8')
    
    def should_analyze(self, detections: List[Dict], collisions: List[Dict]) -> bool:
        """Decide se deve executar análise"""
        now = time.time()
        
        interval = self.analysis_interval
        if self.adaptive_interval:
            risk_intervals = {"critical": 0.5, "high": 1.0, "medium": 2.0}
            interval = risk_intervals.get(self.current_risk, self.analysis_interval)
        
        if now - self.last_analysis_time < interval: return False
        if not detections: return False
        if collisions: return True
        
        return any(np.linalg.norm(d.get('velocity', (0,0))) > 2 for d in detections)

    def _parse_ollama_response(self, response_json: dict) -> Optional[Dict]:
        """Parse robusto de resposta do Ollama para extrair JSON"""
        result_text = response_json.get('response', '').strip()
        
        if not result_text or (not result_text.startswith('{') and 'thinking' in response_json):
            thinking = response_json.get('thinking', '')
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', thinking, re.DOTALL)
            if match:
                result_text = match.group(0)
                print("   [INFO] JSON extraído do 'thinking'")

        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```").strip()
        elif "```" in result_text:
            result_text = result_text.split("```").split("```")[0].strip()

        try:
            result = json.loads(result_text)
            required = ['scene_type', 'risk_level']
            if not all(k in result for k in required):
                print(f"   [ERROR] Campos obrigatórios faltando no JSON: {required}")
                return None
            result.setdefault('critical_objects', [])
            result.setdefault('collision_risk', 'No immediate risk')
            return result
        except json.JSONDecodeError:
            print(f"   [ERROR] JSON inválido na resposta final. Raw: {result_text[:150]}...")
            return None
    
    def _analyze_sync(self, frame: np.ndarray, detections: List[Dict], collisions: List[Dict]) -> Optional[Dict]:
        """Análise síncrona (bloqueante)"""
        context = [f"{d['label']} ID{d['id']}" for d in detections[:10]] # Aumentado para 10 objetos
        
        collision_info = "No collisions detected."
        # ================================================================= #
        # AQUI ESTÁ A CORREÇÃO
        # ================================================================= #
        if collisions:
            # Pega a colisão mais crítica (a primeira da lista)
            critical_collision = collisions[0]
            
            # Agora 'c' é um dicionário, e podemos acessar suas chaves
            c = critical_collision
            time_s = (c['time'] * 5) / 30.0 # O `prediction_step` é 5
            labels = c['labels']
            collision_info = f"\nCRITICAL COLLISION DETECTED: {labels[0]} vs {labels[1]} in {time_s:.1f}s.\n"
        # ================================================================= #

        img_b64 = self.encode_frame(frame)
        if not img_b64: return None

        prompt = f"""Analyze this driving scene. {len(detections)} objects tracked.
{collision_info}
Objects: {', '.join(context)}

Return ONLY valid JSON (no markdown, no thinking):
{{
  "scene_type": "urban|highway|residential",
  "risk_level": "low|medium|high|critical",
  "critical_objects": [ID1, ID2],
  "collision_risk": "brief description of the main risk"
}}"""

        payload = {"model": self.model, "prompt": prompt, "images": [img_b64], "stream": False}
        
        try:
            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.max_timeout)
            inference_time = time.time() - start_time
            
            if response.status_code != 200:
                print(f"   [ERROR] Ollama respondeu com status {response.status_code}")
                return None
            
            result = self._parse_ollama_response(response.json())
            if result:
                self.current_risk = result.get('risk_level', 'unknown')
                self.predictions = result
                self.analysis_count += 1
                print(f"🧠 Analysis #{self.analysis_count} ({inference_time:.2f}s) | Risk: {result['risk_level'].upper()}")
            return result
        except requests.exceptions.RequestException as e:
            print(f"   [ERROR] Falha na comunicação com Ollama: {e}")
            return None

    def _worker(self):
        """Thread worker para análise assíncrona"""
        while True:
            frame, detections, collisions = self.queue.get()
            if frame is None: # Sentinela para parar a thread
                break
            result = self._analyze_sync(frame, detections, collisions)
            if result: self.result_queue.put(result)
    
    def analyze_scene(self, frame: np.ndarray, detections: List[Dict], collisions: List[Dict]) -> Optional[Dict]:
        """Analisa cena (síncrono ou assíncrono)"""
        if not self.should_analyze(detections, collisions):
            return None
        
        self.last_analysis_time = time.time()
        
        if self.use_threading:
            if self.queue.full(): return None
            self.queue.put((frame.copy(), detections, collisions))
            if not self.result_queue.empty(): return self.result_queue.get()
            return None
        else:
            return self._analyze_sync(frame, detections, collisions)
    
    def get_latest_prediction(self) -> Dict:
        """Retorna última predição válida"""
        return self.predictions