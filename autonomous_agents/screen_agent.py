import time
from datetime import datetime
import os
from PIL import ImageGrab

print("🤖 [Screen Activity Agent Initialized]")
print("Target: Monitoring desktop activity and logging state...")

output_dir = os.path.expanduser("~/Desktop/agent_activity_logs")
os.makedirs(output_dir, exist_ok=True)

# Take a sample capture to demonstrate
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
screenshot_path = os.path.join(output_dir, f"activity_{timestamp}.png")

try:
    screenshot = ImageGrab.grab()
    screenshot.save(screenshot_path)
    print(f"📸 [Capture Success]: Saved screenshot to {screenshot_path}")
    print("🧠 [Analysis Engine]: Activity logged. Agent is ready for 24/7 background tracking.")
except Exception as e:
    print(f"⚠️ Capture note: {e}")
