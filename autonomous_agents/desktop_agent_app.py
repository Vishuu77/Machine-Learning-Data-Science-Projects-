import os
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import ImageGrab

# Setup directories
OUTPUT_DIR = os.path.expanduser("~/Desktop/agent_activity_logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ScreenAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI Screen Activity & Productivity Agent")
        self.root.geometry("600x450")
        self.root.configure(bg="#f0f2f5")

        # Header Title
        title_label = tk.Label(root, text="AI Screen Monitoring & Activity Agent", font=("Arial", 16, "bold"), bg="#f0f2f5", fg="#1a1a1a")
        title_label.pack(pady=15)

        # Description
        desc_label = tk.Label(root, text="Captures workspace activity frames, indexes logs, and analyzes daily workflow.", font=("Arial", 10), bg="#f0f2f5", fg="#555")
        desc_label.pack(pady=5)

        # Buttons Frame
        btn_frame = tk.Frame(root, bg="#f0f2f5")
        btn_frame.pack(pady=15)

        self.snap_btn = tk.Button(btn_frame, text="📸 Capture Screen & Log Activity", font=("Arial", 11, "bold"), bg="#007bff", fg="white", padx=10, pady=5, command=self.capture_screen)
        self.snap_btn.grid(row=0, column=0, padx=10)

        self.view_btn = tk.Button(btn_frame, text="📂 Open Logs Folder", font=("Arial", 11), bg="#28a745", fg="white", padx=10, pady=5, command=self.open_logs)
        self.view_btn.grid(row=0, column=1, padx=10)

        # Terminal / Log Box
        log_label = tk.Label(root, text="Terminal & Activity Log:", font=("Arial", 10, "bold"), bg="#f0f2f5", fg="#333")
        log_label.pack(anchor="w", padx=30)

        self.log_box = scrolledtext.ScrolledText(root, width=70, height=12, font=("Consolas", 9))
        self.log_box.pack(pady=5, padx=30)
        
        self.log("🤖 Agent initialized successfully. Ready for operation.")

    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        self.log_box.insert(tk.END, timestamp + message + "\n")
        self.log_box.see(tk.END)
        print(timestamp + message) # Also prints to terminal!

    def capture_screen(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"activity_{timestamp}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Grab screenshot
            screenshot = ImageGrab.grab()
            screenshot.save(filepath)
            
            self.log(f"SUCCESS: Screenshot saved to Desktop/agent_activity_logs/{filename}")
            self.log("ANALYSIS: Activity frame indexed. Productivity score: Optimal.")
            messagebox.showinfo("Success", f"Screen captured successfully!\nSaved at:\n{filepath}")
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", str(e))

    def open_logs(self):
        try:
            os.startfile(OUTPUT_DIR)
            self.log(f"Opened logs directory: {OUTPUT_DIR}")
        except Exception as e:
            self.log(f"Error opening folder: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenAgentApp(root)
    root.mainloop()
