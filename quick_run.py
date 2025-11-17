from core.config import SystemConfig
from main import AutonomousVisionSystem

config = SystemConfig()
config.video.source = "video.mp4"
config.video.save_output = True

system = AutonomousVisionSystem(config)
system.run()