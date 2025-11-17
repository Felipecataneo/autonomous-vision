# Conteúdo para: analysis/base_analyzer.py
from typing import Protocol, List, Dict, Optional
import numpy as np

class BaseVisionAnalyzer(Protocol):
    """
    Interface (Protocol) para analisadores de cena.
    Define os métodos que qualquer analisador de visão deve implementar.
    """
    
    analysis_count: int
    
    def analyze_scene(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        collisions: List[Dict]
    ) -> Optional[Dict]:
        """
        Analisa a cena de um frame com base nas detecções e colisões.
        Este método pode ser síncrono ou assíncrono (iniciar uma thread).
        
        Retorna:
            Um dicionário com a análise da cena ou None se a análise for pulada
            ou estiver em andamento (no modo assíncrono).
        """
        ...

    def should_analyze(
        self,
        detections: List[Dict],
        collisions: List[Dict]
    ) -> bool:
        """
        Decide se uma nova análise de cena deve ser executada.
        Isso é usado para economizar recursos, evitando chamadas de IA a cada frame.
        
        Retorna:
            True se a análise deve ser executada, False caso contrário.
        """
        ...

    def get_latest_prediction(self) -> Dict:
        """
        Retorna a última análise bem-sucedida.
        Útil para manter o estado da HUD enquanto uma nova análise
        está sendo processada em segundo plano.
        
        Retorna:
            Um dicionário com a última predição válida.
        """
        ...