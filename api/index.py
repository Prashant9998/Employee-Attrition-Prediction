import os
import sys

# Ensure local api folder and parent are on sys.path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from flask_app import app
except ImportError:
    from api.flask_app import app

application = app
