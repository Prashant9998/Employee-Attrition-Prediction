import sys
import os

# Add parent directory to path so flask_app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask_app import app

# Vercel entrypoint
# app is exported at the top-level
