# 🚀 Analyso — Vercel Deployment Guide

## Architecture (Everything on Vercel)
```
Vercel Project
├── frontend/   → Static CDN (HTML / CSS / JS)
└── api/        → Python Serverless Functions
    └── index.py  ← Flask app handles /scan, /get_stock_details, /get_news
                    NLTK data lives in Vercel's /tmp ephemeral storage
```

> API routes and the frontend sit on the **same Vercel domain** — no CORS issues, no second service.

---

## Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — Vercel ready"
git remote add origin https://github.com/YOUR_USERNAME/analyso.git
git push -u origin main
```

> **.gitignore** already excludes `data/`, `models/`, `ml traiing for indian stocks/`, `.venv/`
> (too large for GitHub and not needed at runtime — yfinance fetches live data).

---

## Step 2 — Deploy on Vercel

1. Go to **[vercel.com](https://vercel.com)** → sign up with GitHub (free)
2. **"Add New Project"** → import your GitHub repo
3. Configure:

   | Setting | Value |
   |---------|-------|
   | Framework Preset | **Other** |
   | Root Directory | `.` (repo root) |
   | Build Command | *(leave empty)* |
   | Output Directory | *(leave empty)* |

4. Click **Deploy** — Vercel auto-detects `api/index.py` and `vercel.json`

---

## Step 3 — Add Firebase Authorized Domain

After deployment you'll get a URL like `https://analyso.vercel.app`.

1. Go to [Firebase Console](https://console.firebase.google.com/) → your project
2. **Authentication** → **Settings** → **Authorized Domains**
3. Click **Add Domain** → paste your Vercel URL (e.g. `analyso.vercel.app`)

That's it — Google Sign-In will now work on the deployed site.

---

## How It Works

| Request | Handled by |
|---------|-----------|
| `GET /` | `frontend/index.html` via Vercel CDN |
| `GET /login` | `frontend/login.html` via Vercel CDN |
| `GET /css/*`, `/js/*` | `frontend/css/` & `frontend/js/` via Vercel CDN |
| `POST /scan` | `api/index.py` (Python serverless) |
| `POST /get_stock_details` | `api/index.py` (Python serverless) |
| `POST /get_news` | `api/index.py` (Python serverless) |

---

## Local Development (unchanged)

```bash
.\run_app.bat          # starts Flask on http://127.0.0.1:5001
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `413 Payload Too Large` on deploy | Add `vercel-build` to slim dependencies |
| `FUNCTION_INVOCATION_TIMEOUT` | First cold start for ML deps can take ~10s — normal |
| Login doesn't work | Add Vercel URL to Firebase Authorized Domains (Step 3) |
| `ModuleNotFoundError` | Check `requirements.txt` includes all backend deps |

> ⚠️ **Package size note:** Vercel serverless has a ~250MB compressed limit. If deploy fails with a size error, open an issue and we can slim down the dependencies (e.g. drop matplotlib/seaborn which aren't used at runtime).
