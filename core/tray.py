"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          LUMINA STUDIO v1.3                                   ║
║                    Multi-Material 3D Print Color System                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Author: [MIN]                                                                ║
║  License: CC BY-NC-SA 4.0                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

System Tray Icon Module
"""

import os
import sys
import threading
import webbrowser
import pystray
from PIL import Image


class LuminaTray:
    def __init__(self, port=7860):
        self.port = port
        self.icon = None
        self.running = False

    def open_browser(self, icon=None, item=None):
        """Open web interface in default browser."""
        url = f"http://127.0.0.1:{self.port}"
        webbrowser.open(url)

    def exit_app(self, icon=None, item=None):
        """Shutdown the application completely."""
        print("Exiting application...")
        if self.icon:
            self.icon.stop()
        self.running = False
        os._exit(0)  # Force kill all threads (including Gradio)

    def setup_tray(self):
        """Configure tray icon and menu."""
        # Try to load icon, fallback to red square if missing
        icon_path = "icon.ico" if os.path.exists("icon.ico") else "gradio.png"

        try:
            image = Image.open(icon_path)
            # On macOS, menu bar icons should be small (16x16 to 22x22)
            # Resize if needed for better display
            if sys.platform == "darwin":
                # macOS menu bar icons work best at 22x22 or smaller
                if image.size[0] > 22 or image.size[1] > 22:
                    image = image.resize((22, 22), Image.Resampling.LANCZOS)
            else:
                # Other platforms can use larger icons
                if image.size[0] > 64 or image.size[1] > 64:
                    image = image.resize((64, 64), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load icon from {icon_path}: {e}")
            # Create a simple fallback icon
            if sys.platform == "darwin":
                image = Image.new('RGB', (22, 22), color='red')
            else:
                image = Image.new('RGB', (64, 64), color='red')

        menu = pystray.Menu(
            pystray.MenuItem("Open Web UI", self.open_browser, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self.exit_app)
        )

        self.icon = pystray.Icon(
            "LuminaStudio",
            image,
            "Lumina Studio v1.3",
            menu
        )

    def run(self):
        """Start the tray icon in a daemon thread."""
        self.setup_tray()
        self.running = True
        
        # Run pystray in a separate thread to avoid blocking main execution
        if sys.platform == "darwin":
            # macOS: Use run_detached() to avoid blocking and conflicts with Gradio
            # This allows the icon to run in the background without taking over the main thread
            try:
                def setup_icon(icon):
                    """Setup callback for detached mode."""
                    icon.visible = True
                    print("✅ System tray icon started (macOS)")
                
                self.icon.run_detached(setup=setup_icon)
            except Exception as e:
                print(f"⚠️ Warning: Failed to start system tray on macOS: {e}")
                print("   The application will continue without system tray support.")
                self.running = False
                
        else:
            try:
                threading.Thread(target=self.icon.run, daemon=True).start()
                print("✅ System tray icon started")
            except Exception as e:
                print(f"⚠️ Warning: Failed to start system tray: {e}")
                self.running = False
