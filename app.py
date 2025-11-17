# app.py
# =============================================================================
# ChatBot: AI-Driven Health & Nutrition Assistant
# -----------------------------------------------------------------------------
# - Loads compact FAISS + slim metadata from data/processed/metadata/
# - Fast recipe retrieval + deterministic 3-meal planning with hard constraints
# - Home (main-pane wizard), Analysis (visuals), Chat (GPT-style, nutrition-only)
# - LLM assists with plan explanations & parsing preferences if OPENAI_API_KEY set
# =============================================================================


from __future__ import annotations

import os
import re
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import faiss
import altair as alt
from sentence_transformers import SentenceTransformer

# --- Streamlit rerun compat ---
try:
    _RERUN = st.rerun  # Streamlit >= 1.27
except AttributeError:          # older versions
    _RERUN = st.experimental_rerun
# ---------------------------
# Page config & Navigation
# ---------------------------
st.set_page_config(
    page_title="🥗 AI Health & Nutrition Assistant",
    page_icon="🥗",
    layout="wide",
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.header("Navigation")
st.session_state.page = st.sidebar.radio(
    "Go to",
    ["Home", "Analysis", "Chat"],
    index=["Home", "Analysis", "Chat"].index(st.session_state.page),
)

DISCLAIMER = (
    "This tool is for educational purposes only and does **not** provide medical advice. "
    "Consult a qualified professional for personalized guidance."
)

# ---------------------------
# Utilities (paths, helpers)
# ---------------------------
def _first_existing(paths: List[str | Path | None]) -> Path | None:
    for p in paths:
        if p and Path(p).is_file():
            return Path(p)
    return None

def _human_mb(path: Path) -> str:
    try:
        return f"{os.path.getsize(path) / (1024*1024):.1f} MB"
    except Exception:
        return "?"

def _resolve_relative_to(base: Path, maybe_path: str) -> Path:
    p = Path(maybe_path)
    return p if p.is_absolute() else (base / p)

def find_config_path() -> Path:
    env_p = os.environ.get("PIPELINE_CONFIG")
    candidates: list[str | Path | None] = [
        env_p,
        "data/processed/metadata/pipeline_config.json",  # ✅ expected
        "data/metadata/pipeline_config.json",            # fallback (old layout)
        "data/processed/pipeline_config.json",           # fallback
        "pipeline_config.json",                          # last resort
    ]
    cfgp = _first_existing(candidates)
    if not cfgp:
        st.error(
            "Could not find `pipeline_config.json`.\n\n"
            "I looked in:\n- " + "\n- ".join(str(c) for c in candidates) +
            "\n\nFix by placing the config at `data/processed/metadata/pipeline_config.json` "
            "or set `PIPELINE_CONFIG` to the exact file path."
        )
        try:
            st.write("Top-level files:", [str(p) for p in Path('.').iterdir()])
        except Exception:
            pass
        st.stop()
    return cfgp

# ---------------------------
# Cached pipeline loader
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline() -> Tuple[SentenceTransformer, faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    cfg_path = find_config_path()
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    model_name = cfg_raw.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

    base = cfg_path.parent
    faiss_path = _resolve_relative_to(base, cfg_raw["faiss_index"])
    meta_path  = _resolve_relative_to(base, cfg_raw["meta_json"])

    # Validate artifacts
    missing = [p for p in [faiss_path, meta_path] if not p.is_file()]
    if missing:
        st.error(
            "Required artifact(s) not found:\n- " + "\n- ".join(str(m) for m in missing) +
            "\n\nEnsure these files exist in your repo (e.g., under `data/processed/metadata/`)."
        )
        try:
            st.write("Nearby files:", [str(p) for p in base.iterdir()])
        except Exception:
            pass
        st.stop()

    model = SentenceTransformer(model_name)
    index = faiss.read_index(str(faiss_path))
    meta  = json.loads(meta_path.read_text(encoding="utf-8"))

    if len(meta) != index.ntotal:
        st.error(
            f"Meta rows ({len(meta)}) != index.ntotal ({index.ntotal}).\n"
            "Ensure your slim meta aligns 1:1 with index vectors."
        )
        st.stop()

    cfg = {
        **cfg_raw,
        "_resolved_cfg_path": str(cfg_path),
        "_resolved_faiss": str(faiss_path),
        "_resolved_meta": str(meta_path),
        "_faiss_size": _human_mb(faiss_path),
        "_meta_size": _human_mb(meta_path),
        "_model_name": model_name,
    }
    return model, index, meta, cfg

model, index, meta, cfg = load_pipeline()

with st.expander("Artifacts (resolved)", expanded=False):
    st.write("Config:", cfg.get("_resolved_cfg_path"))
    st.write("FAISS :", f"{cfg.get('_resolved_faiss')} ({cfg.get('_faiss_size')})")
    st.write("Meta  :", f"{cfg.get('_resolved_meta')} ({cfg.get('_meta_size')})")
    st.write("Model :", cfg.get("_model_name"))

# ---------------------------
# Retrieval & planning utils
# ---------------------------
def encode_query(text: str) -> np.ndarray:
    emb = model.encode([text], normalize_embeddings=True).astype("float32")
    return emb

def search_recipes(query: str, k: int) -> list[dict]:
    emb = encode_query(query)
    D, I = index.search(emb, int(k))
    out = []
    for idx_i, score in zip(I[0], D[0]):
        m = meta[int(idx_i)]
        out.append({
            "idx": int(idx_i),
            "title": m.get("title"),
            "score": float(score),
            "nutrients_total": m.get("nutrients_total", {}) or {},
        })
    return out

def tdee_msj(age: int, sex: str, height_cm: float, weight_kg: float, activity: str) -> float:
    s = 5 if (sex or "").lower() == "male" else -161
    bmr = 10 * float(weight_kg) + 6.25 * float(height_cm) - 5 * int(age) + s
    factors = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "athlete": 1.9}
    return float(bmr) * factors.get(activity, 1.55)

def macro_targets(calories: int, style: str) -> dict:
    if style == "high_protein":
        pct = {"p": 0.30, "f": 0.25, "c": 0.45}
    elif style == "low_carb":
        pct = {"p": 0.30, "f": 0.40, "c": 0.30}
    else:
        pct = {"p": 0.25, "f": 0.30, "c": 0.45}
    cal = int(calories)
    return {
        "calories": cal,
        "protein_g": round(cal * pct["p"] / 4, 1),
        "fat_g":     round(cal * pct["f"] / 9, 1),
        "carb_g":    round(cal * pct["c"] / 4, 1),
    }

def _num(x: Any) -> float:
    try:
        if x is None: return 0.0
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip()
        if not s: return 0.0
        return float(s)
    except Exception:
        return 0.0

def _within(x: float, lo: float, hi: float) -> bool:
    return (x >= lo) and (x <= hi)

def _score_totals(totals: dict, targets: dict, weights: dict | None = None) -> float:
    w = weights or {"kcal": 1.0, "protein_g": 1.2, "fat_g": 0.8, "carb_g": 0.9}
    return (
        w["kcal"]      * (totals["kcal"]      - targets["calories"])**2 +
        w["protein_g"] * (totals["protein_g"] - targets["protein_g"])**2 +
        w["fat_g"]     * (totals["fat_g"]     - targets["fat_g"])**2 +
        w["carb_g"]    * (totals["carb_g"]    - targets["carb_g"])**2
    )

def strict_plan_from_hits(
    hits: list[dict],
    targets: dict,
    *,
    max_sodium_mg: float = 2300.0,
    max_sugar_g: float = 50.0,
    max_meal_kcal: int = 1000,
    kcal_bounds_per_meal: tuple[float, float] = (200.0, 1200.0),
    sodium_bounds_per_meal: tuple[float, float] = (0.0, 4000.0),
    weights: dict | None = None,
) -> dict | None:
    lo_kcal, hi_kcal = kcal_bounds_per_meal
    lo_na, hi_na = sodium_bounds_per_meal

    pool: list[dict] = []
    for h in hits:
        n = h.get("nutrients_total", {}) or {}
        row = {
            "title":       h.get("title"),
            "kcal":        _num(n.get("kcal")),
            "protein_g":   _num(n.get("protein_g")),
            "fat_g":       _num(n.get("fat_g")),
            "carb_g":      _num(n.get("carb_g")),
            "sugar_g":     _num(n.get("sugar_g")),
            "sodium_mg":   _num(n.get("sodium_mg")),
            "retr_score":  _num(h.get("score")),
        }
        if not _within(row["kcal"], lo_kcal, min(hi_kcal, float(max_meal_kcal))):
            continue
        if not _within(row["sodium_mg"], lo_na, hi_na):
            continue
        pool.append(row)

    if len(pool) < 3:
        return None

    N = min(120, len(pool))
    def protein_density(r):  # g per 100 kcal
        kcal = max(r["kcal"], 1.0)
        return r["protein_g"] / (kcal / 100.0)
    ranked = sorted(pool, key=lambda r: (protein_density(r), r["retr_score"]), reverse=True)[:N]

    best = None
    best_score = math.inf
    for a_i in range(len(ranked)):
        A = ranked[a_i]
        for b_i in range(a_i + 1, len(ranked)):
            B = ranked[b_i]
            for c_i in range(b_i + 1, len(ranked)):
                C = ranked[c_i]
                totals = {
                    "kcal":      A["kcal"]      + B["kcal"]      + C["kcal"],
                    "protein_g": A["protein_g"] + B["protein_g"] + C["protein_g"],
                    "fat_g":     A["fat_g"]     + B["fat_g"]     + C["fat_g"],
                    "carb_g":    A["carb_g"]    + B["carb_g"]    + C["carb_g"],
                    "sugar_g":   A["sugar_g"]   + B["sugar_g"]   + C["sugar_g"],
                    "sodium_mg": A["sodium_mg"] + B["sodium_mg"] + C["sodium_mg"],
                }
                if totals["sodium_mg"] > max_sodium_mg: 
                    continue
                if totals["sugar_g"] > max_sugar_g: 
                    continue

                s = _score_totals(totals, targets, weights)
                if s < best_score:
                    best_score = s
                    best = {
                        "meals": [A, B, C],
                        "totals": {k: round(v, 1) for k, v in totals.items()},
                        "score": round(float(s), 2),
                    }
    return best

# ---------------------------
# LLM helpers
# ---------------------------
def llm_available() -> bool:
    try:
        from openai import OpenAI  # type: ignore
        return bool(os.environ.get("OPENAI_API_KEY"))
    except Exception:
        return False

def _openai_client_safe():
    try:
        from openai import OpenAI  # type: ignore
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAI()
    except Exception:
        pass
    return None

def llm_parse_preferences(diet_choice: str, allergies_text: str) -> dict:
    """
    Returns: {"diet_tags":[...], "blocked_terms":[...]}
    """
    client = _openai_client_safe()
    if client:
        try:
            sys = ("Extract diet tags and blocked terms for recipe retrieval and filtering. "
                   "Allowed diet tags: vegetarian, vegan, non-veg, pescatarian, gluten-free, dairy-free, "
                   "low-sodium, low-sugar, high-protein, low-carb, balanced. "
                   "Return JSON: {diet_tags: [str], blocked_terms: [str]}. Keep to 3-8 diet tags max.")
            usr = {"diet_choice": diet_choice, "allergies": allergies_text}
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},
                          {"role":"user","content":json.dumps(usr)}],
                temperature=0.1
            )
            data = json.loads(r.choices[0].message.content)
            return {
                "diet_tags": [t.strip() for t in data.get("diet_tags", []) if t.strip()],
                "blocked_terms": [t.strip() for t in data.get("blocked_terms", []) if t.strip()],
            }
        except Exception:
            pass
    # Heuristic fallback
    diet_tags = []
    dc = (diet_choice or "").lower()
    if "vegetarian" in dc: diet_tags += ["vegetarian"]
    if "vegan" in dc:      diet_tags += ["vegan"]
    if "pesc" in dc:       diet_tags += ["pescatarian"]
    if "non" in dc:        diet_tags += ["non-veg"]
    if "gluten" in dc:     diet_tags += ["gluten-free"]

    a = (allergies_text or "").lower()
    if "low sodium" in a: diet_tags += ["low-sodium"]
    if "low sugar" in a:  diet_tags += ["low-sugar"]

    blocked = [w.strip() for w in re.split(r"[,\n;]", a) if w.strip()]
    return {"diet_tags": sorted(set(diet_tags)) or ["balanced"], "blocked_terms": blocked}

def explain_plan_with_llm(plan: dict, targets: dict, profile: dict) -> str:
    if not plan or not plan.get("meals") or not llm_available():
        return ""
    from openai import OpenAI  # type: ignore
    client = OpenAI()
    sys = (
        "You are a nutrition coach. Explain a 3-meal plan in a friendly, concise way. "
        "For each meal, infer a short 'typical ingredients' list and 3–5 simple steps. "
        "Keep recipes beginner-friendly, minimal steps, and common pantry items. "
        "Respect that the nutrient numbers are already chosen—do not change them. "
        "Use short bullets. Keep sodium/sugar awareness in mind."
    )
    usr = {"profile": profile, "targets": targets, "plan": plan, "notes": "No medical advice."}
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},{"role":"user","content":json.dumps(usr)}],
        temperature=0.2
    )
    return resp.choices[0].message.content or ""

# ---------------------------
# Home (main-pane wizard)
# ---------------------------
if st.session_state.page == "Home":
    st.title("🥗 AI-Driven Health & Nutrition Assistant")
    st.caption(DISCLAIMER)

    # Defaults
    prof = st.session_state.get("profile", {
        "age": 24, "sex": "male", "height_cm": 176, "weight_kg": 72,
        "activity": "moderate", "goal": "maintain", "macro_style": "balanced"
    })
    caps  = st.session_state.get("caps", {"max_sodium_mg": 2300, "max_sugar_g": 50, "max_meal_kcal": 1000})
    search_cfg = st.session_state.get("search", {"intent": "balanced high-protein", "k": 60})

    st.subheader("Your details")
    c1, c2, c3, c4 = st.columns(4)
    with c1: age = st.number_input("Age", 14, 100, prof["age"])
    with c2: sex = st.selectbox("Sex", ["male","female"], index=0 if prof["sex"]=="male" else 1)
    with c3: height_cm = st.number_input("Height (cm)", 120, 220, prof["height_cm"])
    with c4: weight_kg = st.number_input("Weight (kg)", 35, 200, prof["weight_kg"])

    c5, c6, c7 = st.columns(3)
    with c5:
        activity = st.selectbox("Activity", ["sedentary","light","moderate","active","athlete"],
                                index=["sedentary","light","moderate","active","athlete"].index(prof["activity"]))
    with c6:
        goal = st.selectbox("Goal", ["maintain","loss","gain"],
                            index=["maintain","loss","gain"].index(prof["goal"]))
    with c7:
        macro_style = st.selectbox("Macro style", ["balanced","high_protein","low_carb"],
                                   index=["balanced","high_protein","low_carb"].index(prof["macro_style"]))

    st.subheader("Diet & constraints")
    c8, c9, c10 = st.columns(3)
    with c8:
        diet_choice = st.selectbox("Diet plan", ["vegetarian","vegan","non-veg","pescatarian","gluten-free","balanced"], index=0)
    with c9:
        max_sodium_mg = st.number_input("Max sodium (mg/day)", 0, 6000, caps["max_sodium_mg"], 50)
    with c10:
        max_sugar_g = st.number_input("Max sugar (g/day)", 0, 200, caps["max_sugar_g"], 1)

    c11, c12 = st.columns([2,1])
    with c11:
        allergies_text = st.text_input("Allergies / foods to avoid (comma-separated)", value="")
    with c12:
        max_meal_kcal = st.number_input("Max kcal per meal", 400, 2000, caps["max_meal_kcal"], 50)

    intent_seed = st.text_input("(Optional) Extra tags", value=search_cfg.get("intent",""))
    k = st.slider("Candidates (k)", 10, 150, search_cfg.get("k", 60), 10)

    base_cal = round(tdee_msj(age, sex, height_cm, weight_kg, activity))
    if goal == "loss": base_cal = round(base_cal * 0.85)
    if goal == "gain": base_cal = round(base_cal * 1.10)
    targets = macro_targets(base_cal, macro_style)
    st.markdown(
        f"**Daily targets** → kcal: `{targets['calories']}`, protein: `{targets['protein_g']} g`, "
        f"fat: `{targets['fat_g']} g`, carbs: `{targets['carb_g']} g`"
    )

    # Persist
    st.session_state.profile = {
        "age": age, "sex": sex, "height_cm": height_cm, "weight_kg": weight_kg,
        "activity": activity, "goal": goal, "macro_style": macro_style
    }
    st.session_state.caps = {"max_sodium_mg": int(max_sodium_mg), "max_sugar_g": int(max_sugar_g), "max_meal_kcal": int(max_meal_kcal)}
    st.session_state.search = {"intent": intent_seed, "k": int(k)}
    st.session_state.targets = targets

    build = st.button("🗓️ Build basic meal plan", type="primary")

    if build:
        parsed = llm_parse_preferences(diet_choice, allergies_text)
        diet_tags = parsed["diet_tags"]
        blocked_terms = [b.lower() for b in parsed["blocked_terms"]]

        goal_bias = []
        if macro_style == "high_protein": goal_bias.append("high-protein")
        if macro_style == "low_carb":     goal_bias.append("low-carb")
        if max_sodium_mg <= 2300:         goal_bias.append("low-sodium")
        if max_sugar_g <= 50:             goal_bias.append("low-sugar")

        final_intent = " ".join(sorted(set(diet_tags + goal_bias + (intent_seed.split() if intent_seed else [])))) or "balanced"
        st.session_state.search["intent"] = final_intent

        hits = search_recipes(final_intent, k)
        if blocked_terms:
            filtered = []
            for h in hits:
                title = (h.get("title") or "").lower()
                if any(b in title for b in blocked_terms):
                    continue
                filtered.append(h)
            hits = filtered
        st.session_state.last_hits = hits

        plan = strict_plan_from_hits(
            hits, targets,
            max_sodium_mg=max_sodium_mg,
            max_sugar_g=max_sugar_g,
            max_meal_kcal=max_meal_kcal
        )
        st.session_state.last_plan = plan

    plan = st.session_state.get("last_plan")
    if plan:
        st.markdown("### 🏆 Your 3-meal day")
        df_plan = pd.DataFrame(plan["meals"])
        st.dataframe(df_plan, use_container_width=True, hide_index=True)
        st.markdown(f"**Day totals:** {plan['totals']}")

        with st.expander("🤖 Coach explanation & simple recipes", expanded=True):
            if llm_available():
                txt = explain_plan_with_llm(plan, targets, st.session_state.profile)
                st.markdown(txt or "_No explanation generated._")
            else:
                st.info("Set `OPENAI_API_KEY` to enable LLM explanations of the plan and simple recipes.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 Depth Analysis", use_container_width=True):
                st.session_state.page = "Analysis"
                st.experimental_rerun()
        with c2:
            if st.button("🧑‍⚕️ Chat with AI Coach", use_container_width=True):
                st.session_state.page = "Chat"
                st.experimental_rerun()
    else:
        st.info("Click **Build basic meal plan** to generate a plan under your constraints.")

# ---------------------------
# Analysis Page
# ---------------------------
def _hits_df(hits: list[dict], limit: int = 200) -> pd.DataFrame:
    rows = []
    for h in hits[:limit]:
        n = h.get("nutrients_total", {}) or {}
        rows.append({
            "title": h.get("title"),
            "score": _num(h.get("score")),
            "kcal": _num(n.get("kcal")),
            "protein_g": _num(n.get("protein_g")),
            "fat_g": _num(n.get("fat_g")),
            "carb_g": _num(n.get("carb_g")),
            "sugar_g": _num(n.get("sugar_g")),
            "sodium_mg": _num(n.get("sodium_mg")),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all", subset=["kcal","protein_g","fat_g","carb_g"])
    return df

if st.session_state.page == "Analysis":
    st.markdown("---")
    st.header("📈 Analytics & Insights")

    df_hits = _hits_df(st.session_state.get("last_hits", []), limit=200)

    st.subheader("🔎 Retrieved Set (top candidates)")
    if not df_hits.empty:
        melt = df_hits.melt(value_vars=["kcal","protein_g","fat_g","carb_g"], var_name="nutrient", value_name="value")
        box = (
            alt.Chart(melt.dropna())
            .mark_boxplot()
            .encode(
                x=alt.X("nutrient:N", title="Nutrient"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("nutrient:N", legend=None),
            ).properties(height=280)
        )
        hist = (
            alt.Chart(df_hits.dropna(subset=["kcal"]))
            .mark_bar()
            .encode(
                x=alt.X("kcal:Q", bin=alt.Bin(maxbins=30), title="Energy (kcal)"),
                y=alt.Y("count():Q", title="Count"),
            ).properties(height=220)
        )
        scatter = (
            alt.Chart(df_hits.dropna(subset=["kcal","protein_g"]))
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("kcal:Q", title="Energy (kcal)"),
                y=alt.Y("protein_g:Q", title="Protein (g)"),
                color=alt.Color("score:Q", title="Retrieval score"),
                tooltip=["title","kcal","protein_g","fat_g","carb_g","sodium_mg","score"],
            ).interactive().properties(height=280)
        )
        c1, c2 = st.columns(2)
        with c1: st.altair_chart(box, use_container_width=True)
        with c2: st.altair_chart(hist, use_container_width=True)
        st.altair_chart(scatter, use_container_width=True)
    else:
        st.info("Retrieve some candidates first to see dataset visuals.")

    st.subheader("🗓️ Plan vs Targets")
    plan = st.session_state.get("last_plan")
    targets = st.session_state.get("targets", {"calories":2000,"protein_g":120,"fat_g":67,"carb_g":225})
    if plan:
        totals = plan["totals"]
        df_macros = pd.DataFrame([
            {"type": "Target", "kcal": targets["calories"], "protein_g": targets["protein_g"], "fat_g": targets["fat_g"], "carb_g": targets["carb_g"]},
            {"type": "Plan",   "kcal": totals["kcal"],       "protein_g": totals["protein_g"], "fat_g": totals["fat_g"], "carb_g": totals["carb_g"]},
        ])
        macros_melt = df_macros.melt(id_vars=["type"], value_vars=["kcal","protein_g","fat_g","carb_g"],
                                     var_name="metric", value_name="value")
        bars = (
            alt.Chart(macros_melt)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title=""),
                y=alt.Y("value:Q", title="Amount"),
                color=alt.Color("type:N", title="", scale=alt.Scale(scheme="set2")),
                column=alt.Column("type:N", header=alt.Header(labelOrient="bottom")),
            ).properties(height=260)
        )
        st.altair_chart(bars, use_container_width=True)

        caps = st.session_state.caps
        df_caps = pd.DataFrame([
            {"metric": "Sodium (mg)", "value": totals.get("sodium_mg", 0.0), "cap": caps["max_sodium_mg"]},
            {"metric": "Sugar (g)",   "value": totals.get("sugar_g", 0.0),   "cap": caps["max_sugar_g"]},
        ])
        for _, row in df_caps.iterrows():
            metric = row["metric"]
            value = float(row["value"] or 0)
            cap   = float(row["cap"] or 0)
            pct = (value / cap) if cap > 0 else 0.0
            st.markdown(f"**{metric}** — {int(value)} / {int(cap)} ( {pct*100:.0f}% of cap )")
            df_bar = pd.DataFrame({"label": [metric, "Cap"], "amount": [value, cap]})
            chart = (
                alt.Chart(df_bar)
                .mark_bar()
                .encode(
                    x=alt.X("label:N", title=""),
                    y=alt.Y("amount:Q", title="Amount"),
                    color=alt.Color("label:N", legend=None,
                                    scale=alt.Scale(domain=[metric,"Cap"], range=["#e66101","#5e3c99"])),
                    tooltip=["label","amount"]
                ).properties(height=180)
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Build a plan to compare against your targets and caps.")

# ---------------------------
# Chat Page (GPT-style)
# ---------------------------
def run_agent(user_query: str, ctx: dict) -> str:
    """
    Very lightweight agent:
    - if message mentions swap → regenerate plan
    - if mentions grocery → list meal titles (LLM can elaborate)
    - else: if question, LLM answers about nutrition; if request, rebuild plan
    """
    q = user_query.lower()
    reply_lines: List[str] = []

    # ensure hits/targets exist for actions
    hits = ctx.get("last_hits") or []
    targets = ctx.get("targets") or st.session_state.get("targets")
    caps    = ctx.get("caps") or st.session_state.get("caps", {})
    k       = ctx.get("settings", {}).get("k", 60)
    intent  = ctx.get("settings", {}).get("intent", st.session_state.get("search", {}).get("intent", "balanced"))
    max_meal_kcal = ctx.get("settings", {}).get("max_meal_kcal", st.session_state.get("caps", {}).get("max_meal_kcal", 1000))

    # retrieve on demand
    if ("retrieve" in q or (not hits and ("plan" in q or "build" in q))):
        st.session_state.last_hits = search_recipes(intent, k)
        hits = st.session_state.last_hits
        reply_lines.append(f"🔎 Retrieved {len(hits)} candidates for: **{intent}**.")

    # swap/regenerate plan
    if "swap" in q or "rebuild" in q or "new plan" in q or "change plan" in q:
        plan = strict_plan_from_hits(
            hits or search_recipes(intent, k),
            targets,
            max_sodium_mg=caps.get("max_sodium_mg", 2300),
            max_sugar_g=caps.get("max_sugar_g", 50),
            max_meal_kcal=max_meal_kcal
        )
        st.session_state.last_plan = plan
        if plan:
            reply_lines.append("🗓️ I rebuilt your 3-meal day under your constraints.")
        else:
            reply_lines.append("I couldn't find a feasible 3-meal combo under current caps. Try increasing `Candidates (k)` or loosening caps.")

    # grocery list
    if "grocery" in q or "shopping" in q:
        plan = st.session_state.get("last_plan")
        if not plan:
            reply_lines.append("You don't have a plan yet. Ask me to *build a plan* first.")
        else:
            # We don't have structured ingredients; give meal titles (LLM can expand)
            titles = [m["title"] for m in plan["meals"]]
            reply_lines.append("🛒 Grocery list (starter):")
            reply_lines += [f"- {t}" for t in titles]

    # If purely a question or request for explanation, defer to LLM if available
    if llm_available() and not reply_lines:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        sys = (
            "You are a nutrition coach. Answer clearly and concisely, grounded in general nutrition guidance. "
            "If asked for a plan change, outline steps and specify any constraint changes. "
            "Keep scope to diet & meal planning. No medical advice."
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user_query}],
            temperature=0.3
        )
        return r.choices[0].message.content

    if not reply_lines:
        reply_lines.append("Tell me to *build a plan*, *swap dinner*, or *make a grocery list*.")

    return "\n".join(reply_lines)

if st.session_state.page == "Chat":
    st.header("💬 Coach Chat")

    st.markdown(
        """
        <style>
          .bubble-user   { background: #1f6feb22; border:1px solid #1f6feb44; padding:12px 14px; border-radius:12px; }
          .bubble-assist { background: #2ea04322; border:1px solid #2ea04344; padding:12px 14px; border-radius:12px; }
          .sysnote       { color:#8b949e; font-size:0.9rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []

    st.markdown('<div class="sysnote">Ask about meals, swaps, allergens, macros, or a grocery list.</div>',
                unsafe_allow_html=True)

    # render history
    for m in st.session_state.chat_msgs[-40:]:
        with st.chat_message(m["role"]):
            klass = "bubble-user" if m["role"] == "user" else "bubble-assist"
            st.markdown(f'<div class="{klass}">{m["content"]}</div>', unsafe_allow_html=True)

    user_msg = st.chat_input("Type your message…")
    if user_msg:
        # keep to nutrition domain
        diet_terms = ["diet","meal","recipe","protein","carb","fat","kcal","calorie","macro","sodium","sugar",
                      "fiber","vegetarian","vegan","pescatarian","gluten","lactose","allergen","plan",
                      "breakfast","lunch","dinner","snack","grocery","swap","rebuild"]
        if not any(t in user_msg.lower() for t in diet_terms):
            user_msg = ("I’m your nutrition coach, so I stay focused on **diet & meal planning**. "
                        "Try: *“Build a high-protein vegetarian plan under 2300 mg sodium”* "
                        "or *“Swap dinner and keep the rest”*.")
            st.session_state.chat_msgs.append({"role":"assistant","content":user_msg})
            st.experimental_rerun()

        st.session_state.chat_msgs.append({"role":"user","content":user_msg})
        ctx = {
            "profile": st.session_state.get("profile", {}),
            "targets": st.session_state.get("targets", {}),
            "caps":     st.session_state.get("caps", {}),
            "settings": {
                "k": st.session_state.get("search", {}).get("k", 60),
                "intent": st.session_state.get("search", {}).get("intent", "balanced"),
                "max_meal_kcal": st.session_state.get("caps", {}).get("max_meal_kcal", 1000),
            },
            "last_hits": st.session_state.get("last_hits", []),
            "last_plan": st.session_state.get("last_plan"),
        }
        reply = run_agent(user_msg, ctx)
        st.session_state.chat_msgs.append({"role":"assistant","content":reply})
        st.experimental_rerun()

# ---------------------------
# Small polish & footer
# ---------------------------
APP_VERSION = "1.0.0"
st.markdown("---")
col_f1, col_f2 = st.columns([3,1])
with col_f1:
    st.caption("© 2025 AI Health & Nutrition Assistant — Educational use only. Not medical advice.")
with col_f2:
    st.caption(f"v{APP_VERSION}")
