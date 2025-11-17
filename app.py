import os, json, math, itertools, faiss, numpy as np, streamlit as st
from pathlib import Path

st.set_page_config(page_title="AI-Driven Health & Nutrition Assistant", page_icon="🥗", layout="wide")

DATA_PROC = Path("data/processed")
CONFIG_PATH = DATA_PROC / "pipeline_config.json"

@st.cache_resource(show_spinner=False)
def load_pipeline():
    cfg = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["model_dir"])
    index = faiss.read_index(cfg["faiss_index"])
    meta  = json.loads(Path(cfg["meta_json"]).read_text(encoding="utf-8"))
    return model, index, meta, cfg

def search_recipes(query, k, model, index, meta):
    from sentence_transformers import SentenceTransformer
    emb = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(emb, k)
    out = []
    for idx, score in zip(I[0], D[0]):
        m = meta[int(idx)]
        out.append({
            "idx": int(idx),
            "title": m.get("title"),
            "score": float(score),
            "nutrients_total": m.get("nutrients_total", {})
        })
    return out

def macro_targets(calories, style):
    if style == "high_protein": pct={"p":0.30,"f":0.25,"c":0.45}
    elif style == "low_carb":   pct={"p":0.30,"f":0.40,"c":0.30}
    else:                       pct={"p":0.25,"f":0.30,"c":0.45}
    return {
        "calories": int(calories),
        "protein_g": round(calories*pct["p"]/4,1),
        "fat_g":     round(calories*pct["f"]/9,1),
        "carb_g":    round(calories*pct["c"]/4,1),
    }

def _num(x):
    try: return float(x or 0.0)
    except: return 0.0

def _score_set(totals, targets, w=None):
    w = w or {"kcal":1.0, "protein_g":1.2, "fat_g":0.8, "carb_g":0.9}
    return (
        w["kcal"]      * (totals["kcal"]      - targets["calories"])**2 +
        w["protein_g"] * (totals["protein_g"] - targets["protein_g"])**2 +
        w["fat_g"]     * (totals["fat_g"]     - targets["fat_g"])**2 +
        w["carb_g"]    * (totals["carb_g"]    - targets["carb_g"])**2
    )

def strict_plan_from_hits(hits, targets, max_sodium_mg=2300, max_sugar_g=50, max_meal_kcal=1000):
    pool=[]
    for h in hits:
        n = h.get("nutrients_total",{}) or {}
        row = {
            "title": h["title"],
            "kcal": _num(n.get("kcal")), "protein_g": _num(n.get("protein_g")),
            "fat_g": _num(n.get("fat_g")), "carb_g": _num(n.get("carb_g")),
            "sugar_g": _num(n.get("sugar_g")), "sodium_mg": _num(n.get("sodium_mg")),
            "score": float(h.get("score",0))
        }
        if row["kcal"] <= 0 or row["kcal"] > 2000: continue
        if row["sodium_mg"] < 0 or row["sodium_mg"] > 4000: continue
        if row["kcal"] > max_meal_kcal: continue
        pool.append(row)

    best=None; best_score=math.inf
    for a,b,c in itertools.combinations(pool, 3):
        totals = {
            "kcal": a["kcal"]+b["kcal"]+c["kcal"],
            "protein_g": a["protein_g"]+b["protein_g"]+c["protein_g"],
            "fat_g": a["fat_g"]+b["fat_g"]+c["fat_g"],
            "carb_g": a["carb_g"]+b["carb_g"]+c["carb_g"],
            "sugar_g": a["sugar_g"]+b["sugar_g"]+c["sugar_g"],
            "sodium_mg": a["sodium_mg"]+b["sodium_mg"]+c["sodium_mg"],
        }
        if totals["sodium_mg"] > max_sodium_mg: continue
        if totals["sugar_g"]  > max_sugar_g:  continue
        s = _score_set(totals, targets)
        if s < best_score:
            best_score = s
            best = {"meals": [a,b,c], "totals": {k: round(v,1) for k,v in totals.items()}, "score": round(s,2)}
    return best

def llm_available():
    try:
        from openai import OpenAI
        _ = os.environ.get("OPENAI_API_KEY")
        return _ is not None and len(_) > 0
    except Exception:
        return False

def llm_explain_plan(plan_dict, model_name="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI()
    system = "You are a nutrition coach. Explain in 3-5 bullet points why this 3-meal plan fits the user's targets and constraints. Be specific about macros and sodium. Keep it concise."
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":json.dumps(plan_dict)}
    ]
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()

model, index, meta, cfg = load_pipeline()
st.success(f"Loaded model & index • recipes: {len(meta):,}")

with st.sidebar:
    st.header("User & Constraints")
    query = st.text_input("Search intent", value="balanced vegetarian high-protein")
    k      = st.slider("Candidates (k)", 10, 100, 60, 10)
    calories = st.number_input("Calories target", 1200, 5000, 2200, 50)
    macro   = st.selectbox("Macro style", ["balanced","high_protein","low_carb"])
    max_sodium = st.number_input("Max sodium (mg)", 0, 6000, 2300, 50)
    max_sugar  = st.number_input("Max sugar (g)", 0, 200, 50, 1)
    max_meal_kcal = st.number_input("Max calories per meal", 400, 2000, 1000, 50)
    want_llm = st.checkbox("Use LLM to explain the final plan (requires OPENAI_API_KEY)", value=True)
    run = st.button("Generate Plan")

tabs = st.tabs(["🔎 Search", "🗓️ Planner", "💬 Chat"])

if run:
    hits = search_recipes(query, k, model, index, meta)

    with tabs[0]:
        st.subheader("Top results (first 10)")
        for i, h in enumerate(hits[:10], 1):
            n = h["nutrients_total"]
            st.write(f"{i}. **{h['title']}**  | score `{h['score']:.3f}` | kcal `{n.get('kcal')}` | protein `{n.get('protein_g')}` g")

    with tabs[1]:
        st.subheader("Planner")
        tgt = macro_targets(int(calories), macro)
        st.markdown("**🎯 Targets**")
        st.json(tgt)

        best = strict_plan_from_hits(hits, tgt, max_sodium_mg=max_sodium, max_sugar_g=max_sugar, max_meal_kcal=max_meal_kcal)
        if best:
            st.markdown("**🏆 Strict 3-meal plan (constraint-safe)**")
            for i, m in enumerate(best["meals"], 1):
                st.markdown(f"**{i}. {m['title']}** — kcal `{m['kcal']}` | protein `{m['protein_g']}`g | fat `{m['fat_g']}`g | carb `{m['carb_g']}`g | Na `{m['sodium_mg']}`mg")
            st.markdown(f"**Day totals:** {best['totals']}")

            if want_llm and llm_available():
                st.markdown("---")
                st.subheader("🤖 LLM Rationale")
                plan_payload = {
                    "query": query,
                    "targets": tgt,
                    "meals": best["meals"],
                    "day_totals": best["totals"],
                    "rules": {"max_sodium_mg": int(max_sodium), "max_sugar_g": int(max_sugar)}
                }
                try:
                    rationale = llm_explain_plan(plan_payload)
                    st.write(rationale)
                except Exception as e:
                    st.warning(f"LLM explanation failed: {e}")
            elif want_llm:
                st.info("Set the OPENAI_API_KEY environment variable to enable LLM explanations.")
        else:
            st.warning("No 3-meal combo satisfied your caps. Try increasing k or relaxing caps.")

    with tabs[2]:
        st.subheader("Chat (LLM)")
        st.caption("Ask: *Why sodium cap?*, *Swap dinner under 700 kcal*, etc.")
        if "chat" not in st.session_state: st.session_state.chat = []
        user_msg = st.text_input("You:", key="chat_input")
        if st.button("Send", key="send_btn"):
            if user_msg.strip():
                st.session_state.chat.append(("user", user_msg.strip()))
                if llm_available():
                    try:
                        from openai import OpenAI
                        client = OpenAI()
                        context = {"last_query": query, "targets": macro_targets(int(calories), macro)}
                        sys = "You are a helpful nutrition assistant. Keep answers concise and grounded."
                        messages = [{"role":"system","content":sys},
                                    {"role":"user","content":json.dumps(context)},
                                    {"role":"user","content":user_msg.strip()}]
                        r = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.3)
                        st.session_state.chat.append(("assistant", r.choices[0].message.content.strip()))
                    except Exception as e:
                        st.session_state.chat.append(("assistant", f"(LLM error: {e})"))
                else:
                    st.session_state.chat.append(("assistant", "LLM not configured. Set OPENAI_API_KEY."))

        for role, msg in st.session_state.chat[-12:]:
            if role == "user": st.markdown(f"**You:** {msg}")
            else: st.markdown(f"**Assistant:** {msg}")
else:
    with tabs[0]:
        st.info("Enter your search intent on the left and click **Generate Plan**.")
    with tabs[1]:
        st.info("Planner will appear here after you click **Generate Plan**.")
    with tabs[2]:
        st.info("Enable OPENAI_API_KEY to use chat.")
