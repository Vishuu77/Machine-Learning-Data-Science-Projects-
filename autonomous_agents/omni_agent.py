import os
import glob
import time
from datetime import datetime
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
import cv2
import numpy as np
from PIL import ImageGrab
import sounddevice as sd
import soundfile as sf
import keyboard

# PDF and Docx parsers
from pypdf import PdfReader
from docx import Document

# Setup directories
OUTPUT_DIR = os.path.expanduser("~/Desktop/omni_agent_logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class UltimateOmniAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Ultimate AI Omni Agent (Screen + Audio + Deep Document Reader)")
        self.root.geometry("750x620")
        self.root.configure(bg="#f0f2f5")

        self.is_recording = False
        self.video_filename = ""
        self.audio_filename = ""
        self.captured_frames_count = 0
        self.target_doc_folder = os.path.expanduser("~/Documents")
        if not os.path.exists(self.target_doc_folder):
            self.target_doc_folder = os.path.expanduser("~/Desktop")

        # Title
        title_label = tk.Label(root, text="Ultimate AI Omni Screen & Document Reading Agent", font=("Arial", 15, "bold"), bg="#f0f2f5", fg="#1a1a1a")
        title_label.pack(pady=12)

        desc_label = tk.Label(root, text="Stealth Screen/Audio Recording + Deep Content Extraction & Summarization.", font=("Arial", 9), bg="#f0f2f5", fg="#555")
        desc_label.pack(pady=2)

        # Folder selection frame for document reading
        doc_frame = tk.Frame(root, bg="#e9ecef", padx=10, pady=8)
        doc_frame.pack(fill="x", padx=30, pady=5)

        self.folder_label = tk.Label(doc_frame, text=f"Target Doc Folder: {self.target_doc_folder}", font=("Arial", 9), bg="#e9ecef", fg="#333", anchor="w")
        self.folder_label.pack(side=tk.LEFT, fill="x", expand=True)

        self.pick_folder_btn = tk.Button(doc_frame, text="📁 Choose Folder", font=("Arial", 9), bg="#6c757d", fg="white", command=self.pick_doc_folder)
        self.pick_folder_btn.pack(side=tk.RIGHT)

        # Instructions Frame
        info_frame = tk.Frame(root, bg="#fff3cd", padx=10, pady=8)
        info_frame.pack(pady=5, fill="x", padx=30)
        info_label = tk.Label(info_frame, text="💡 Instructions:\n1. Choose your document folder (excludes agent logs automatically).\n2. Click Start (UI hides in Stealth Mode).\n3. Press [Ctrl + Shift + S] to STOP. The agent will record screen/audio AND extract content from your actual docs!", font=("Arial", 9), bg="#fff3cd", fg="#856404", justify="left")
        info_label.pack(anchor="w")

        # Buttons Frame
        btn_frame = tk.Frame(root, bg="#f0f2f5")
        btn_frame.pack(pady=10)

        self.start_btn = tk.Button(btn_frame, text="🔴 Start Ultimate Omni Agent", font=("Arial", 11, "bold"), bg="#dc3545", fg="white", padx=12, pady=6, command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.view_btn = tk.Button(btn_frame, text="📂 Open Reports Folder", font=("Arial", 11), bg="#28a745", fg="white", padx=12, pady=6, command=self.open_folder)
        self.view_btn.grid(row=0, column=1, padx=10)

        # Terminal / Log Box
        log_label = tk.Label(root, text="Terminal & Real-Time Agent Logs:", font=("Arial", 10, "bold"), bg="#f0f2f5", fg="#333")
        log_label.pack(anchor="w", padx=30)

        self.log_box = scrolledtext.ScrolledText(root, width=82, height=10, font=("Consolas", 9))
        self.log_box.pack(pady=5, padx=30)
        
        self.log("🤖 Ultimate Omni Agent initialized. Ready.")

    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        self.log_box.insert(tk.END, timestamp + message + "\n")
        self.log_box.see(tk.END)
        print(timestamp + message)

    def pick_doc_folder(self):
        folder = filedialog.askdirectory(title="Select Document Folder to Read")
        if folder:
            self.target_doc_folder = folder
            self.folder_label.config(text=f"Target Doc Folder: {self.target_doc_folder}")
            self.log(f"Target document folder updated: {self.target_doc_folder}")

    def start_recording(self):
        self.is_recording = True
        self.captured_frames_count = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = os.path.join(OUTPUT_DIR, f"omni_screen_{timestamp}.avi")
        self.audio_filename = os.path.join(OUTPUT_DIR, f"omni_audio_{timestamp}.wav")
        
        self.log("🕶️ Entering Stealth Mode: UI hidden. Recording screen & audio...")
        self.root.withdraw()

        # Start recording threads
        self.video_thread = threading.Thread(target=self._record_video)
        self.audio_thread = threading.Thread(target=self._record_audio)
        
        self.video_thread.daemon = True
        self.audio_thread.daemon = True
        
        self.video_thread.start()
        self.audio_thread.start()

        # Hotkey to stop (Ctrl + Shift + S)
        keyboard.add_hotkey('ctrl+shift+s', self.stop_recording)

    def _record_video(self):
        screen = ImageGrab.grab()
        width, height = screen.size
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.video_filename, fourcc, 4.0, (width, height))
        
        while self.is_recording:
            img = ImageGrab.grab()
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            self.captured_frames_count += 1
            time.sleep(0.25)
            
        out.release()

    def _record_audio(self):
        samplerate = 44100
        channels = 1
        audio_data = []
        
        def callback(indata, frames, time_info, status):
            if self.is_recording:
                audio_data.append(indata.copy())

        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
            while self.is_recording:
                time.sleep(0.5)
                
        if audio_data:
            full_audio = np.concatenate(audio_data, axis=0)
            sf.write(self.audio_filename, full_audio, samplerate)

    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        time.sleep(1) # Cleanup
        
        self.root.deiconify()
        
        self.log(f"⏹️ Recording stopped. Captured {self.captured_frames_count} screen frames.")
        self.log(f"📚 Reading and extracting document contents in folder: {self.target_doc_folder}...")
        
        self.root.after(1000, self._process_documents_and_report)

    def _process_documents_and_report(self):
        extensions = ('*.txt', '*.md', '*.py', '*.csv', '*.json', '*.log', '*.pdf', '*.docx')
        raw_files = []
        for ext in extensions:
            raw_files.extend(glob.glob(os.path.join(self.target_doc_folder, "**", ext), recursive=True))

        # Filter out agent output logs so it only reads real user documents
        files = []
        for filepath in raw_files:
            if "omni_agent_logs" in filepath or "agent_activity_logs" in filepath or "stealth_agent_logs" in filepath:
                continue
            files.append(filepath)

        doc_summaries = []
        for filepath in files:
            filename = os.path.basename(filepath)
            content = ""
            try:
                if filepath.endswith('.pdf'):
                    reader = PdfReader(filepath)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            content += text + "\n"
                elif filepath.endswith('.docx'):
                    doc = Document(filepath)
                    for para in doc.paragraphs:
                        content += para.text + "\n"
                else:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                word_count = len(content.split())
                cleaned_content = " ".join(content.split())
                
                if len(cleaned_content) > 0:
                    content_summary = cleaned_content[:600] + ("..." if len(cleaned_content) > 600 else "")
                else:
                    content_summary = "[No text content detected in this file]"

                doc_summaries.append({
                    "filename": filename,
                    "path": filepath,
                    "words": word_count,
                    "content_summary": content_summary
                })
                self.log(f"   -> Read & Extracted Content: {filename} ({word_count} words)")
            except Exception as e:
                self.log(f"   -> Error reading {filename}: {str(e)}")

        # Build final report
        report_path = os.path.join(OUTPUT_DIR, f"ultimate_omni_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        report_content = f"""==================================================
ULTIMATE AI OMNI SCREEN, AUDIO & DOCUMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================
1. LIVE RECORDING METRICS:
   - Video Recording: {self.video_filename}
   - Audio Recording: {self.audio_filename}
   - Screen Frames Captured: {self.captured_frames_count}

2. DEEP DOCUMENT CONTENT EXTRACTION (Target Folder: {self.target_doc_folder}):
   - Total User Documents Read: {len(doc_summaries)}
"""

        for doc in doc_summaries:
            report_content += f"""
   ----------------------------------------------
   📄 FILE: {doc['filename']}
      - Path: {doc['path']}
      - Word Count: {doc['words']}
      - ACTUAL DOCUMENT CONTENT & SUMMARY:
        "{doc['content_summary']}"
"""

        report_content += "\n==================================================\n"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.log("="*40)
        self.log("✅ ULTIMATE OMNI REPORT GENERATED:")
        self.log(f"- Report saved to: {report_path}")
        self.log(f"- Extracted actual content from {len(doc_summaries)} user documents!")
        self.log("="*40)
        
        messagebox.showinfo("Analysis Complete", f"Ultimate Omni Report Generated!\n\nExtracted content from {len(doc_summaries)} user documents.\nReport saved to:\n{report_path}")

    open_folder = lambda self: os.startfile(OUTPUT_DIR)

if __name__ == "__main__":
    root = tk.Tk()
    app = UltimateOmniAgentApp(root)
    root.mainloop()
