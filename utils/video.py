"""
Gerenciador de captura e gravação de vídeo
"""
import cv2
import numpy as np
from datetime import datetime
from typing import Optional


class VideoManager:
    """Gerencia captura e gravação de vídeo"""
    
    def __init__(self, source: str = "0"):
        """
        Args:
            source: 0 para webcam, caminho para vídeo, ou URL
        """
        # Converte "0" para int
        if source.isdigit():
            source = int(source)
        
        self.source = source
        self.cap = None
        self.writer = None
        self.fps = 30
        self.width = 0
        self.height = 0
        self.frame_count = 0
    
    def open(self) -> bool:
        """Abre fonte de vídeo"""
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"❌ Cannot open: {self.source}")
            return False
        
        # Propriedades
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Opened: {self.source}")
        print(f"   Resolution: {self.width}x{self.height} @ {self.fps} FPS")
        
        return True
    
    def read(self) -> Optional[np.ndarray]:
        """Lê próximo frame"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            return frame
        return None
    
    def setup_writer(self, output_path: Optional[str] = None) -> bool:
        """
        Configura gravador de vídeo
        
        Args:
            output_path: Caminho de saída (auto-gera se None)
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"autonomous_vision_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if self.writer.isOpened():
            print(f"💾 Recording to: {output_path}")
            return True
        else:
            print(f"❌ Failed to open writer: {output_path}")
            return False
    
    def write(self, frame: np.ndarray):
        """Grava frame"""
        if self.writer is not None:
            self.writer.write(frame)
    
    def release(self):
        """Libera recursos"""
        if self.cap is not None:
            self.cap.release()
            print(f"\n📊 Total frames processed: {self.frame_count}")
        
        if self.writer is not None:
            self.writer.release()
            print("✅ Video saved")
    
    def is_opened(self) -> bool:
        """Verifica se está aberto"""
        return self.cap is not None and self.cap.isOpened()


class VideoDisplay:
    """Gerencia exibição de janelas"""
    
    def __init__(self, window_name: str = "Autonomous Vision"):
        self.window_name = window_name
        self.is_showing = False
    
    def create_window(self):
        """Cria janela OpenCV"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.is_showing = True
    
    def show(self, frame: np.ndarray) -> bool:
        """
        Mostra frame
        
        Returns:
            True se deve continuar, False se 'q' pressionado
        """
        if not self.is_showing:
            self.create_window()
        
        cv2.imshow(self.window_name, frame)
        
        # Verifica tecla 'q'
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
    
    def close(self):
        """Fecha janela"""
        if self.is_showing:
            cv2.destroyAllWindows()
            self.is_showing = False