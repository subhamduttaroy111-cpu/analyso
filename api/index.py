"""
api/index.py — Vercel Python Serverless Entry Point
────────────────────────────────────────────────────
Vercel routes /scan, /get_stock_details, /get_news here.
The Flask app from backend/ handles all API logic.
NLTK data is stored in /tmp (Vercel's ephemeral storage).
"""

import sys
import os

# ── Add backend/ to Python path ──────────────────────
_ROOT = os.path.dirname(os.path.dirname(__file__))  # project root
_BACKEND = os.path.join(_ROOT, "backend")
sys.path.insert(0, _BACKEND)
sys.path.insert(0, _ROOT)

# ── Use /tmp for NLTK data (Vercel serverless /tmp is writable) ──
os.environ["NLTK_DATA"] = "/tmp/nltk_data"
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ── Import the Flask app (server.py creates it) ───────
#    Static folder not needed in production — Vercel serves frontend directly
from flask import Flask
from flask_cors import CORS
from routes import register_routes
import nltk

# Download vader_lexicon to /tmp on cold start if missing
_NLTK_DIR = "/tmp/nltk_data"
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    SentimentIntensityAnalyzer()  # trigger lookup error if not downloaded
except LookupError:
    nltk.download("vader_lexicon", download_dir=_NLTK_DIR, quiet=True)

# ── Build Flask App ────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")  # Vercel handles HTTPS, all origins fine here

register_routes(app)

# ── Vercel handler ─────────────────────────────────────
# Vercel's Python runtime looks for a variable named 'handler'
# OR it will use 'app' if it's a WSGI application.
handler = app
