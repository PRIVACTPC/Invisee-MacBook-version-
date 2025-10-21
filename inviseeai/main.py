import sys
import os
import webbrowser
from waitress import serve
from dotenv import load_dotenv

# The core 'app' object is created in app.py, so we import it here.
# The modifications below will configure it for the bundled .exe environment.
from app import app

# =============================================================================
#  HELPER FUNCTION FOR PYINSTALLER
# =============================================================================
def resource_path(relative_path):
    """
    Get the absolute path to a resource, which works for both development
    and for a PyInstaller bundled executable.
    """
    try:
        # PyInstaller creates a temporary folder and stores its path in _MEIPASS.
        # This is a special attribute added by the PyInstaller bootloader.
        base_path = sys._MEIPASS
    except Exception:
        # If not running as a bundled exe, the base path is just the current directory.
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# =============================================================================
#  CRITICAL CONFIGURATION FOR PYINSTALLER
# =============================================================================
# When running as a bundled executable ('frozen'), we must tell Flask where to find
# the 'static' and 'templates' folders. The 'app' object was already created
# in app.py, so we modify its attributes here before running it.

if getattr(sys, 'frozen', False):
    print("Application is running in bundled mode (frozen).")
    # Update the paths for the static and template folders
    app.static_folder = resource_path('static')
    app.template_folder = resource_path('templates')
else:
    print("Application is running in development mode.")
# =============================================================================


# Load environment variables from a .env file (primarily for development)
# In the bundled app, this will do nothing if a .env file isn't present,
# which is fine because we provide default values below.
load_dotenv()

def run_app():
    """
    Configures and runs the Flask application using the Waitress server,
    and opens the user's web browser.
    """
    # Get host and port, with sensible defaults for a desktop app.
    # The host should always be '127.0.0.1' (localhost) for security.
    host = '127.0.0.1'
    port = int(os.getenv('FLASK_PORT', 5000)) # Using 5000 as a common default

    # This message will appear in the console window when the user runs the .exe
    print("===================================================")
    print("      Privacy Preserving Application Server      ")
    print("===================================================")
    print(f" * Starting server at: http://{host}:{port}")
    print(" * Your web browser will open automatically.")
    print(" * To stop the server, simply close this window.")
    print("---------------------------------------------------")

    # Automatically open the application in the user's default web browser
    webbrowser.open(f"http://{host}:{port}")

    # Start the production-ready Waitress server. This is more robust
    # than Flask's built-in development server.
    serve(app, host=host, port=port)

# This is the main entry point for the script
if __name__ == '__main__':
    run_app()