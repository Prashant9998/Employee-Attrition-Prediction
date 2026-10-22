import os
from flask_app import app

# Top-level export for Vercel, Render, Heroku, or standard WSGI runners
application = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
