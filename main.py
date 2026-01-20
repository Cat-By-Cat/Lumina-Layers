"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          LUMINA STUDIO v1.3                                   ║
║                    Multi-Material 3D Print Color System                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Author: [MIN]                                                                ║
║  License: CC BY-NC-SA 4.0                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Main Entry Point
"""

import os
import sys
import time
import threading
import webbrowser
import gradio as gr     # type:ignore
from ui.layout import create_app
from core.tray import LuminaTray
from ui.styles import CUSTOM_CSS

PORT = 7860

def start_browser():
    """Launch the default web browser after a short delay."""
    time.sleep(2)
    webbrowser.open(f"http://127.0.0.1:{PORT}")

if __name__ == "__main__":

    # 1. Start System Tray
    # macOS support: Uses run_detached() to avoid conflicts with Gradio
    print("🚀 Starting System Tray...")
    try:
        tray = LuminaTray(port=PORT)
        tray.run()
    except Exception as e:
        print(f"⚠️ Warning: System tray failed to start: {e}")
        print("   The application will continue without system tray support.")

    # 2. Start Browser Thread
    threading.Thread(target=start_browser, daemon=True).start()

    # 3. Launch Gradio App
    print(f"✨ Lumina Studio is running on http://127.0.0.1:{PORT}")
    app = create_app()

    try:
        app.launch(
            inbrowser=False,
            server_name="127.0.0.1",
            server_port=PORT,
            show_error=True,
            prevent_thread_lock=True,
            favicon_path="icon.ico" if os.path.exists("icon.ico") else None,
            css=CUSTOM_CSS, 
            theme=gr.themes.Soft()
        )
    except Exception as e:
        raise
    except BaseException as e:
        raise

    # 4. Keep Main Thread Alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        os._exit(0)
