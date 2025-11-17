# 🥗 ChatBot: AI-Driven Health & Nutrition Assistant (Streamlit)

This is a clean, production-ready Streamlit app that uses your **saved model artifacts** (Sentence-Transformers + FAISS) to do:
- semantic recipe search,
- **constraint-safe 3‑meal planning** (hard caps on sodium/sugar + macro targets),
- **LLM explanations and chat** with OpenAI (optional).

## What to copy from your previous project
Copy these files/folders into `data/processed/`:

- `pipeline_config.json`
- `faiss_index_small.faiss`
- `recipes_meta_small.json`
- `model_all-MiniLM-L6-v2/` (entire folder)
- (optional) `recipes_nutrition_small.jsonl`, `foods.csv`

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...   # optional but recommended
streamlit run app.py
```

## Deploy (Streamlit Cloud)
- Push this folder to GitHub.
- In Streamlit Cloud, set `OPENAI_API_KEY` in the app's Secrets/Env.
- Ensure `data/processed/` contains the artifacts listed above.
