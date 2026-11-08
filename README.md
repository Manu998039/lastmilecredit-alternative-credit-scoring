# LastMileCredit - setup and run

## 1. Install
```
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## 2. Generate data and train the model
```
python train_model.py
```
Creates `msme_loans.csv` (15,000 rows, 12 columns) and `model.joblib`; prints AUC-ROC and accuracy.

## 3. Start the backend
```
uvicorn main:app --reload
```
- API docs: http://127.0.0.1:8000/docs
- Optional live SHAP explanations: set `USE_SHAP=1` (needs more memory)

## 4. Open the frontend
Open `frontend/index.html` in a browser (or `python -m http.server 5500` inside `frontend/`).

## 5. Deploy (free tier)
- Backend -> Render.com: push this folder to GitHub (include `model.joblib`),
  build `pip install -r requirements.txt`, start `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Frontend -> Netlify: drag-and-drop the `frontend/` folder; set the Render URL in `API` in index.html
- Keep-alive: set the URL in `keep_alive.html` and leave it open (pings /health every 5 min)

## Project layout
```
train_model.py   data generation + XGBoost training
main.py          FastAPI backend (scoring, SQLite, lenders, stats)
frontend/index.html   mobile-first 3-step web app
keep_alive.html  health-check pinger
requirements.txt
```
