# Automate-data-preprocessing

An automated data preprocessing app built with Streamlit and LangGraph. It loads a CSV, detects the target column, splits and classifies features, generates preprocessing pipelines, and exports a reusable bundle.

## Requirements

- Python 3.10+
- A valid `GOOGLE_API_KEY`
- A `MODEL_NAME` supported by your GenAI provider

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the values you need.

## Run

```bash
streamlit run app.py
```

## Streamlit Cloud

1. Push the repo to GitHub.
2. Create a new Streamlit app from that repository.
3. Set the main file path to `app.py`.
4. Add these secrets in the Streamlit Cloud Secrets editor:

```toml
GOOGLE_API_KEY = "your-key"
MODEL_NAME = "your-model"
```

5. If you want persistent LangGraph checkpoints, also add `SUPABASE_DB_URL` or `DATABASE_URL`.

The app can run without a database URL; it falls back to in-memory checkpoints.

## Deployment notes

- If `SUPABASE_DB_URL` or `DATABASE_URL` is set, the app uses Postgres-backed LangGraph checkpoints.
- If no database URL is provided, the app falls back to in-memory checkpoints and still runs.
- Dataset profiling is best-effort; if the profiling dependency is unavailable or fails, preprocessing still continues.
- On Streamlit Cloud, use Secrets for `GOOGLE_API_KEY` and `MODEL_NAME` instead of a local `.env` file.

## Output

The app generates a downloadable ZIP bundle with the fitted preprocessor, split data, processed data, and any generated feature engineering assets.