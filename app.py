# =========================
# Section 0: Conversational Onboarding
# =========================
# - On first run, ask for user info & goal in main pane (not sidebar)
# - Offer goal presets + example prompts
# - LLM-based interpreter (with regex fallback) extracts profile, macro style, intent
# - Auto-builds plan, then reveals sidebar for advanced tuning and analysis

from __future__ import annotations

import os
import json
import math
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
def _openai_client_safe():
    try:
        from openai import OpenAI  # type: ignore
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAI()
    except Exception:
        pass
    return None

# --- FALLBACKS so Section 0 can run even if Section 2 is below ---
try:
    _ = tdee_msj  # type: ignore[name-defined]
except NameError:
    def tdee_msj(age, sex, height_cm, weight_kg, activity):
        s = 5 if (str(sex).lower() == "male") else -161
        bmr = 10*float(weight_kg) + 6.25*float(height_cm) - 5*int(age) + s
        factors = {"sedentary":1.2, "light":1.375, "moderate":1.55, "active":1.725, "athlete":1.9}
        return bmr * factors.get(str(activity).lower(), 1.55)

try:
    _ = macro_targets  # type: ignore[name-defined]
except NameError:
    def macro_targets(calories: int, style: str):
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
def _coerce_sex(s: str) -> str:
    s = (s or "").lower()
    if "f" in s and "male" not in s: return "female"
    return "male" if "m" in s else ("female" if "f" in s else "male")

def _infer_activity(text: str) -> str:
    t = (text or "").lower()
    if "athlete" in t or "very active" in t: return "athlete"
    if "active" in t: return "active"
    if "moderate" in t: return "moderate"
    if "light" in t or "lightly" in t: return "light"
    return "moderate"

def _infer_macro_style(text: str) -> str:
    t = (text or "").lower()
    if "high-protein" in t or "high protein" in t: return "high_protein"
    if "low-carb" in t or "low carb" in t: return "low_carb"
    return "balanced"

def _parse_numbers(text: str):
    import re
    # crude, resilient extraction
    age = next((int(x) for x in re.findall(r"\b(\d{2})\b", text) if 14 <= int(x) <= 99), 24)
    cm  = next((int(x) for x in re.findall(r"(\d{3})\s*cm", text.lower())), 176)
    kg  = next((int(x) for x in re.findall(r"(\d{2,3})\s*kg", text.lower())), 72)
    return age, cm, kg

def interpret_onboarding(free_text: str, preset: str | None) -> dict:
    """
    Returns a dict with:
      profile: {age, sex, height_cm, weight_kg, activity, goal, macro_style}
      caps:    {max_sodium_mg, max_sugar_g, max_meal_kcal}
      search:  {intent, k}
    Uses LLM if available; otherwise, uses resilient heuristics.
    """
    # Defaults
    profile = {
        "age": 24, "sex": "male", "height_cm": 176, "weight_kg": 72,
        "activity": "moderate", "goal": "maintain", "macro_style": "balanced"
    }
    caps = {"max_sodium_mg": 2300, "max_sugar_g": 50, "max_meal_kcal": 1000}
    search = {"intent": "balanced high-protein vegetarian", "k": 60}

    client = _openai_client_safe()
    if client:
        try:
            system = (
                "Extract nutrition planning parameters from the user text. "
                "Return strict JSON with keys: profile{age:int,sex:str,height_cm:int,weight_kg:int,"
                "activity in [sedentary,light,moderate,active,athlete], goal in [maintain,loss,gain],"
                "macro_style in [balanced,high_protein,low_carb]}, "
                "caps{max_sodium_mg:int,max_sugar_g:int,max_meal_kcal:int}, "
                "search{intent:str,k:int}. Keep intent short (3-8 tags)."
            )
            prompt = free_text + ("\n\nPreset: " + preset if preset else "")
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
                temperature=0.1
            )
            import json as _json
            parsed = _json.loads(r.choices[0].message.content)
            profile.update(parsed.get("profile", {}))
            caps.update(parsed.get("caps", {}))
            search.update(parsed.get("search", {}))
            # sanitize
            profile["sex"] = _coerce_sex(profile.get("sex","male"))
            profile["activity"] = profile.get("activity","moderate")
            profile["goal"] = profile.get("goal","maintain")
            profile["macro_style"] = profile.get("macro_style","balanced")
            search["k"] = int(search.get("k",60))
            return {"profile":profile,"caps":caps,"search":search}
        except Exception:
            pass  # fall back

    # ---- Heuristic fallback (no API key / LLM error) ----
    t = (free_text or "") + (" " + (preset or ""))
    age, cm, kg = _parse_numbers(t)
    profile.update({
        "age": age,
        "sex": _coerce_sex(t),
        "height_cm": cm,
        "weight_kg": kg,
        "activity": _infer_activity(t),
        "goal": ("loss" if "loss" in t or "cut" in t else ("gain" if "gain" in t or "bulk" in t else "maintain")),
        "macro_style": _infer_macro_style(t),
    })
    if "low sodium" in t or "heart" in t: caps["max_sodium_mg"] = 2000
    if "low sugar" in t: caps["max_sugar_g"] = 40
    if "small meals" in t: caps["max_meal_kcal"] = 800
    # intent seeds
    intent_bits = []
    if "vegetarian" in t or "veggie" in t: intent_bits.append("vegetarian")
    if "vegan" in t: intent_bits.append("vegan")
    if "pescatarian" in t: intent_bits.append("pescatarian")
    if profile["macro_style"] == "high_protein": intent_bits.append("high-protein")
    if profile["macro_style"] == "low_carb": intent_bits.append("low-carb")
    if "gluten" in t: intent_bits.append("gluten-free")
    if not intent_bits: intent_bits = ["balanced"]
    search.update({"intent": " ".join(sorted(set(intent_bits))), "k": 60})
    return {"profile":profile,"caps":caps,"search":search}

# ---------- Onboarding UI ----------
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False

if not st.session_state.onboarded:
    st.title("🥗 AI-Driven Health & Nutrition Assistant")
    st.subheader("Tell me about you and your goal")
    c1, c2 = st.columns([1,3])
    with c1:
        go = st.button("Create my plan", type="primary")
    with c2:
        st.caption("I’ll compute calories/macros, retrieve recipes, and assemble a 3-meal plan under sensible caps.")

    if go:
        parsed = interpret_onboarding(free.strip() or EXAMPLE_PROMPTS[0], preset)
        # write into session state used by later sections
        st.session_state.profile = parsed["profile"]
        st.session_state.caps    = parsed["caps"]
        st.session_state.search  = parsed["search"]
        # compute targets
        p = st.session_state.profile
        base_cal = round(tdee_msj(p["age"], p["sex"], p["height_cm"], p["weight_kg"], p["activity"]))
        if p["goal"] == "loss": base_cal = round(base_cal * 0.85)
        if p["goal"] == "gain": base_cal = round(base_cal * 1.10)
        st.session_state.targets = macro_targets(base_cal, p["macro_style"])
        # retrieve + plan immediately
        st.session_state.last_hits = search_recipes(st.session_state.search["intent"], st.session_state.search["k"])
        st.session_state.last_plan = strict_plan_from_hits(
            st.session_state.last_hits,
            st.session_state.targets,
            max_sodium_mg=st.session_state.caps["max_sodium_mg"],
            max_sugar_g=st.session_state.caps["max_sugar_g"],
            max_meal_kcal=st.session_state.caps["max_meal_kcal"],
        )
        st.session_state.onboarded = True
        st.experimental_rerun()


# =========================
# Section 1: Bootstrapping
# =========================
# - Imports & Streamlit page config
# - Robust discovery of pipeline_config.json
# - Cached pipeline loader (model by name + FAISS + slim meta)
# - Hard validation of artifacts so failures are obvious


# === Global navigation (sidebar just for nav) ===
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.header("Navigation")
st.session_state.page = st.sidebar.radio(
    "Go to",
    ["Home", "Analysis", "Chat"],
    index=["Home","Analysis","Chat"].index(st.session_state.page),
)


# ---- Streamlit page config (set this before anything renders)
st.set_page_config(
    page_title="🥗 AI Health & Nutrition Assistant",
    page_icon="🥗",
    layout="wide",
)

# ---- Constants
DISCLAIMER = (
    "This tool is for educational purposes only and does **not** provide medical advice. "
    "Consult a qualified professional for personalized guidance."
)

# ---- Small utility
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

# -----------------------------------------
# Robust config discovery & artifact resolve
# -----------------------------------------
def find_config_path() -> Path:
    """
    Returns the path to pipeline_config.json by checking common locations and an env override.
    This makes the app tolerant to small repo layout differences.
    """
    env_p = os.environ.get("PIPELINE_CONFIG")
    candidates: list[str | Path | None] = [
    env_p,                                           # if set via env var
    "data/processed/metadata/pipeline_config.json",  # ✅ your real path
    "data/metadata/pipeline_config.json",            # fallback (old structure)
    "data/processed/pipeline_config.json",           # fallback
    "pipeline_config.json",                          # last resort
    ]
    cfgp = _first_existing(candidates)
    if not cfgp:
        st.error(
            "Could not find `pipeline_config.json`.\n\n"
            "I looked in:\n- " + "\n- ".join(str(c) for c in candidates) +
            "\n\nFix by placing the config at `data/processed/metadata/pipeline_config.json` "
            "or set an env var `PIPELINE_CONFIG` to the exact file path."
        )
        # Show top-level tree to help debug quickly
        try:
            st.write("Top-level files:", [str(p) for p in Path('.').iterdir()])
        except Exception:
            pass
        st.stop()
    return cfgp

def _resolve_relative_to(base: Path, maybe_path: str) -> Path:
    """
    If `maybe_path` is relative, interpret it relative to `base` (the folder of the config).
    """
    p = Path(maybe_path)
    return p if p.is_absolute() else (base / p)

# -----------------------------------------
# Cached pipeline: model (by name), FAISS, meta
# -----------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline() -> Tuple[SentenceTransformer, faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads:
      - SentenceTransformer by *name* (auto-download to HF cache, keeps repo small)
      - FAISS IVF-PQ index (or any FAISS index path pointed by config)
      - Slim metadata (titles + nutrients), aligned 1:1 with index rows

    Returns:
      (model, index, meta, cfg) where cfg also includes resolved file paths for debugging.
    """
    cfg_path = find_config_path()
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Model by name keeps your repo tiny and reproducible
    model_name = cfg_raw.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

    # Resolve artifact paths relative to the config file’s parent
    base = cfg_path.parent
    faiss_path = _resolve_relative_to(base, cfg_raw["faiss_index"])
    meta_path  = _resolve_relative_to(base, cfg_raw["meta_json"])

    # Validate artifacts exist
    missing = [p for p in [faiss_path, meta_path] if not p.is_file()]
    if missing:
        st.error(
            "Required artifact(s) not found:\n- " + "\n- ".join(str(m) for m in missing) +
            "\n\nEnsure these files exist in your repo (e.g., under `data/processed/metadata/`)."
        )
        # Show neighbors to quickly diagnose misplaced files
        try:
            st.write("Nearby files:", [str(p) for p in base.iterdir()])
        except Exception:
            pass
        st.stop()

    # Load assets
    model = SentenceTransformer(model_name)
    index = faiss.read_index(str(faiss_path))
    meta  = json.loads(meta_path.read_text(encoding="utf-8"))

    # Strong sanity check: meta size must match index rows
    if len(meta) != index.ntotal:
        st.error(
            f"Meta rows ({len(meta)}) != index.ntotal ({index.ntotal}).\n"
            "Make sure your meta is the slimmed file aligned to the same vectors as the FAISS index."
        )
        st.stop()

    # Enrich cfg for debugging and UI visibility
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

# ---- Bootstrap the pipeline early so later sections can rely on it
model, index, meta, cfg = load_pipeline()

# Optional: a tiny expander to confirm what the app is actually using
with st.expander("Artifacts (resolved)", expanded=False):
    st.write("Config:", cfg.get("_resolved_cfg_path"))
    st.write("FAISS :", f"{cfg.get('_resolved_faiss')} ({cfg.get('_faiss_size')})")
    st.write("Meta  :", f"{cfg.get('_resolved_meta')} ({cfg.get('_meta_size')})")
    st.write("Model :", cfg.get("_model_name"))


# =========================
# Section 2: Core utilities
# =========================
# - encode_query / search_recipes: semantic retrieval via your FAISS index
# - tdee_msj / macro_targets: daily energy and macro targets
# - strict_plan_from_hits: deterministic 3-meal selector with hard constraints
# - Helper scoring & number parsing utilities

from typing import Iterable

# -------- Retrieval --------
def encode_query(model: SentenceTransformer, text: str) -> np.ndarray:
    """
    Encodes a single query into a normalized embedding compatible with the FAISS index.
    """
    emb = model.encode([text], normalize_embeddings=True).astype("float32")
    return emb

def search_recipes(query: str, k: int) -> list[dict]:
    """
    Semantic search on the prebuilt FAISS index.
    Returns list of dicts: {idx, title, score, nutrients_total}.
    """
    emb = encode_query(model, query)
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

# -------- Nutrition calculators --------
def tdee_msj(age: int, sex: str, height_cm: float, weight_kg: float, activity: str) -> float:
    """
    Mifflin–St Jeor TDEE with standard activity multipliers.
    sex: 'male' or 'female'
    activity in {'sedentary','light','moderate','active','athlete'}
    """
    s = 5 if (sex or "").lower() == "male" else -161
    bmr = 10 * float(weight_kg) + 6.25 * float(height_cm) - 5 * int(age) + s
    factors = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "athlete": 1.9}
    return float(bmr) * factors.get(activity, 1.55)

def macro_targets(calories: int, style: str) -> dict:
    """
    Macro split presets -> grams based on kcal.
    - balanced:     P 25% / F 30% / C 45%
    - high_protein: P 30% / F 25% / C 45%
    - low_carb:     P 30% / F 40% / C 30%
    """
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

# -------- Helpers: numeric parsing & scoring --------
def _num(x: Any) -> float:
    """
    Best-effort safe numeric conversion.
    Accepts None/str/float/int; returns float (0.0 on failure).
    """
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
    """
    Weighted squared error to bias fit toward desired macros.
    Lower is better. Defaults mildly favor protein accuracy.
    """
    w = weights or {"kcal": 1.0, "protein_g": 1.2, "fat_g": 0.8, "carb_g": 0.9}
    return (
        w["kcal"]      * (totals["kcal"]      - targets["calories"])**2 +
        w["protein_g"] * (totals["protein_g"] - targets["protein_g"])**2 +
        w["fat_g"]     * (totals["fat_g"]     - targets["fat_g"])**2 +
        w["carb_g"]    * (totals["carb_g"]    - targets["carb_g"])**2
    )

# -------- Deterministic, constraint-safe 3-meal selector --------
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
    """
    Deterministically pick EXACTLY 3 meals from `hits` that:
      • respect per-meal kcal and sodium bounds,
      • keep daily sodium ≤ max_sodium_mg and sugar ≤ max_sugar_g,
      • minimize macro error vs. `targets`.

    Returns:
      {
        "meals": [ {title,kcal,protein_g,fat_g,carb_g,sugar_g,sodium_mg}, x3 ],
        "totals": {...},
        "score": float
      }
      or None if no feasible triple was found.
    """
    lo_kcal, hi_kcal = kcal_bounds_per_meal
    lo_na, hi_na = sodium_bounds_per_meal

    # Pre-filter pool for feasibility & quality
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
            "retr_score":  _num(h.get("score")),  # retrieval similarity, FYI
        }
        # Hard per-meal sanity filters
        if not _within(row["kcal"], lo_kcal, min(hi_kcal, float(max_meal_kcal))): 
            continue
        if not _within(row["sodium_mg"], lo_na, hi_na): 
            continue
        pool.append(row)

    # Early exit if pool too small
    if len(pool) < 3:
        return None

    # Efficient heuristic: try combinations from the top-N by retrieval + protein density
    # This balances quality with speed for large k.
    # You can tune N; 80–120 usually finds a good feasible set quickly.
    N = min(120, len(pool))
    def protein_density(r):  # g protein per 100 kcal (simple quality signal)
        kcal = max(r["kcal"], 1.0)
        return r["protein_g"] / (kcal / 100.0)
    ranked = sorted(pool, key=lambda r: (protein_density(r), r["retr_score"]), reverse=True)[:N]

    best = None
    best_score = math.inf

    # Try all triples from the ranked shortlist
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
                # Hard day caps
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

# === LLM explanation / simple ingredients & steps for the chosen plan ===
def llm_available() -> bool:
    try:
        from openai import OpenAI  # type: ignore
        return bool(os.environ.get("OPENAI_API_KEY"))
    except Exception:
        return False

def explain_plan_with_llm(plan: dict, targets: dict, profile: dict) -> str:
    """
    Returns a readable explanation + simple ingredients and steps for each meal.
    Works only if OPENAI_API_KEY is set; otherwise returns an empty string.
    """
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

    usr = {
        "profile": profile,
        "targets": targets,
        "plan": plan,
        "notes": "Keep it short, actionable. No medical advice."
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":sys},
            {"role":"user","content":json.dumps(usr)}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content or ""




# =========================
# Section 3R: Home (main-pane inputs + plan)
# =========================
if st.session_state.page == "Home":

    st.title("🥗 AI-Driven Health & Nutrition Assistant")
    st.caption(DISCLAIMER)

    # Defaults / state hookup
    prof = st.session_state.get("profile", {
        "age": 24, "sex": "male", "height_cm": 176, "weight_kg": 72,
        "activity": "moderate", "goal": "maintain", "macro_style": "balanced"
    })
    caps = st.session_state.get("caps", {"max_sodium_mg": 2300, "max_sugar_g": 50, "max_meal_kcal": 1000})
    search_cfg = st.session_state.get("search", {"intent": "balanced vegetarian high-protein", "k": 60})

    # ---- Main-pane inputs (no sidebar) ----
    st.subheader("Tell me about you")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", 14, 100, prof["age"])
    with c2:
        sex = st.selectbox("Sex", ["male","female"], index=0 if prof["sex"]=="male" else 1)
    with c3:
        height_cm = st.number_input("Height (cm)", 120, 220, prof["height_cm"])
    with c4:
        weight_kg = st.number_input("Weight (kg)", 35, 200, prof["weight_kg"])

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

    st.subheader("Constraints")
    c8, c9, c10 = st.columns(3)
    with c8:
        max_sodium_mg = st.number_input("Max sodium (mg/day)", 0, 6000, caps["max_sodium_mg"], 50)
    with c9:
        max_sugar_g = st.number_input("Max sugar (g/day)", 0, 200, caps["max_sugar_g"], 1)
    with c10:
        max_meal_kcal = st.number_input("Max kcal per meal", 400, 2000, caps["max_meal_kcal"], 50)

    st.subheader("What do you feel like eating?")
    intent = st.text_input("Intent / tags (e.g., 'vegetarian high-protein low sodium')", search_cfg["intent"])
    k = st.slider("Candidates (k)", 10, 150, search_cfg["k"], 10)

    # compute targets from profile
    base_cal = round(tdee_msj(age, sex, height_cm, weight_kg, activity))
    if goal == "loss": base_cal = round(base_cal * 0.85)
    if goal == "gain": base_cal = round(base_cal * 1.10)
    targets = macro_targets(base_cal, macro_style)

    st.markdown(
        f"**Daily targets** → kcal: `{targets['calories']}`, protein: `{targets['protein_g']} g`, "
        f"fat: `{targets['fat_g']} g`, carbs: `{targets['carb_g']} g`"
    )

    cact1, cact2, cact3 = st.columns([1,1,2])
    with cact1:
        run_search = st.button("🔎 Retrieve")
    with cact2:
        run_plan = st.button("🗓️ Build 3-meal plan", type="primary")

    # persist state
    st.session_state.profile = {
        "age": age, "sex": sex, "height_cm": height_cm, "weight_kg": weight_kg,
        "activity": activity, "goal": goal, "macro_style": macro_style
    }
    st.session_state.caps = {"max_sodium_mg": int(max_sodium_mg), "max_sugar_g": int(max_sugar_g), "max_meal_kcal": int(max_meal_kcal)}
    st.session_state.search = {"intent": intent, "k": int(k)}
    st.session_state.targets = targets

    # actions
    if run_search:
        st.session_state.last_hits = search_recipes(intent, k)

    if run_plan:
        if not st.session_state.get("last_hits"):
            st.session_state.last_hits = search_recipes(intent, k)
        st.session_state.last_plan = strict_plan_from_hits(
            st.session_state.last_hits, targets,
            max_sodium_mg=max_sodium_mg, max_sugar_g=max_sugar_g, max_meal_kcal=max_meal_kcal
        )

    # ---- Results area ----
    st.markdown("### 🔎 Top results")
    hits = st.session_state.get("last_hits", [])
    if hits:
        df_hits = pd.DataFrame([
            {
                "rank": i+1, "title": h["title"], "score": round(h["score"], 3),
                "kcal": h["nutrients_total"].get("kcal"),
                "protein_g": h["nutrients_total"].get("protein_g"),
                "fat_g": h["nutrients_total"].get("fat_g"),
                "carb_g": h["nutrients_total"].get("carb_g"),
                "sugar_g": h["nutrients_total"].get("sugar_g"),
                "sodium_mg": h["nutrients_total"].get("sodium_mg"),
            } for i, h in enumerate(hits[:30])
        ])
        st.dataframe(df_hits, use_container_width=True, hide_index=True)
    else:
        st.info("Click **Retrieve** to fetch candidates.")

    st.markdown("### 🏆 Your 3-meal day")
    plan = st.session_state.get("last_plan")
    if plan:
        df_plan = pd.DataFrame(plan["meals"])
        st.dataframe(df_plan, use_container_width=True, hide_index=True)
        st.markdown(f"**Day totals:** {plan['totals']}")

        # LLM explanation (ingredients + simple steps)
        with st.expander("🤖 Coach explanation & simple recipes", expanded=True):
            if llm_available():
                txt = explain_plan_with_llm(plan, targets, st.session_state.profile)
                st.markdown(txt or "_No explanation generated._")
            else:
                st.info("Set `OPENAI_API_KEY` to enable LLM explanations of the plan and simple recipes.")

        cdl1, cdl2 = st.columns(2)
        with cdl1:
            st.download_button("⬇️ Meals (CSV)", df_plan.to_csv(index=False).encode("utf-8"), "plan_meals.csv", "text/csv")
        with cdl2:
            st.download_button("⬇️ Totals (JSON)", json.dumps(plan["totals"], indent=2).encode("utf-8"), "plan_totals.json", "application/json")
    else:
        st.info("Click **Build 3-meal plan** to generate a plan under your constraints.")

# =========================
# Section 4: Analytics & Insights
# =========================
# Visuals on the retrieved set and the current plan:
# - Boxplots for kcal / macros on top hits
# - Histogram of kcal distribution
# - Protein vs kcal scatter (quality view)
# - Macro fit: plan vs targets
# - Sodium & sugar vs caps
if st.session_state.page != "Analysis":
    st.stop()

import altair as alt

st.markdown("---")
st.header("📈 Analytics & Insights")

# ------- Build a compact hits DataFrame (first 200 for speed/clarity) -------
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
    # Filter out totally empty rows for cleaner charts
    if not df.empty:
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all", subset=["kcal","protein_g","fat_g","carb_g"])
    return df

df_hits = _hits_df(st.session_state.last_hits, limit=200)

# ------- Subsection: Retrieved set visuals -------
st.subheader("🔎 Retrieved Set (top candidates)")

if not df_hits.empty:
    # Boxplots for kcal/protein/fat/carb
    melt = df_hits.melt(value_vars=["kcal","protein_g","fat_g","carb_g"], var_name="nutrient", value_name="value")
    box = (
        alt.Chart(melt.dropna())
        .mark_boxplot()
        .encode(
            x=alt.X("nutrient:N", title="Nutrient"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color("nutrient:N", legend=None),
        )
        .properties(height=280)
    )

    # kcal histogram
    hist = (
        alt.Chart(df_hits.dropna(subset=["kcal"]))
        .mark_bar()
        .encode(
            x=alt.X("kcal:Q", bin=alt.Bin(maxbins=30), title="Energy (kcal)"),
            y=alt.Y("count():Q", title="Count"),
        )
        .properties(height=220)
    )

    # protein vs kcal scatter colored by similarity score
    scatter = (
        alt.Chart(df_hits.dropna(subset=["kcal","protein_g"]))
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("kcal:Q", title="Energy (kcal)"),
            y=alt.Y("protein_g:Q", title="Protein (g)"),
            color=alt.Color("score:Q", title="Retrieval score"),
            tooltip=["title","kcal","protein_g","fat_g","carb_g","sodium_mg","score"],
        )
        .interactive()
        .properties(height=280)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(box, use_container_width=True)
    with c2:
        st.altair_chart(hist, use_container_width=True)

    st.altair_chart(scatter, use_container_width=True)
else:
    st.info("Retrieve some candidates first to see dataset visuals.")

# ------- Subsection: Plan vs Targets -------
st.subheader("🗓️ Plan vs Targets")

plan = st.session_state.last_plan
targets = current_targets

if plan:
    totals = plan["totals"]
    # Macro comparison
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
        )
        .properties(height=260)
    )
    st.altair_chart(bars, use_container_width=True)

    # Sodium & sugar vs caps
    caps = st.session_state.caps
    df_caps = pd.DataFrame([
        {"metric": "Sodium (mg)", "value": totals.get("sodium_mg", 0.0), "cap": caps["max_sodium_mg"]},
        {"metric": "Sugar (g)",   "value": totals.get("sugar_g", 0.0),   "cap": caps["max_sugar_g"]},
    ])
    # Two side-by-side bars with conditional color if exceeding cap
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
            )
            .properties(height=180)
        )
        st.altair_chart(chart, use_container_width=True)

else:
    st.info("Build a plan to compare against your targets and caps.")

# =========================
# Section 5R: Chat (two-way, ChatGPT-style)
# =========================
if st.session_state.page == "Chat":
    st.header("💬 Coach Chat")

    # --- Minimal CSS for nicer bubbles ---
    st.markdown(
        """
        <style>
          .bubble-user    { background: #1f6feb22; border:1px solid #1f6feb44; padding:12px 14px; border-radius:12px; }
          .bubble-assist  { background: #2ea04322; border:1px solid #2ea04344; padding:12px 14px; border-radius:12px; }
          .bubble-tool    { background: #6e768166; border:1px dashed #6e7681; padding:10px 12px; border-radius:10px; font-size:0.9rem; }
          .sysnote        { color:#8b949e; font-size:0.9rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Create chat state ---
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []  # list[dict]: {"role":"user|assistant","content":str}

    # --- Quick intents (optional chips) ---
    cols = st.columns(4)
    quicks = [
        "Build a high-protein vegetarian day under 2300 mg sodium.",
        "Swap dinner, keep breakfast & lunch.",
        "Avoid peanuts and shellfish and rebuild.",
        "Make a grocery list for the current plan.",
    ]
    for i, c in enumerate(cols):
        with c:
            if st.button(quicks[i], use_container_width=True):
                st.session_state.chat_msgs.append({"role": "user", "content": quicks[i]})

                # Prepare agent context
                ctx = {
                    "profile": st.session_state.get("profile", {}),
                    "targets": st.session_state.get("targets", {}),
                    "caps":     st.session_state.get("caps", {}),
                    "settings": {
                        "k": st.session_state.get("search", {}).get("k", 60),
                        "intent": st.session_state.get("search", {}).get("intent", "balanced high-protein"),
                        "max_meal_kcal": st.session_state.get("caps", {}).get("max_meal_kcal", 1000),
                    },
                    "last_hits": st.session_state.get("last_hits", []),
                    "last_plan": st.session_state.get("last_plan"),
                }
                reply = run_agent(quicks[i], ctx) if llm_available() else (
                    "LLM not configured. Set `OPENAI_API_KEY` to enable the coach chat."
                )
                st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
                st.experimental_rerun()

    st.markdown('<div class="sysnote">Ask me anything about <b>diet & nutrition</b>. '
                'I can retrieve recipes, build/swaps plans, honor sodium/sugar caps, and make grocery lists.</div>',
                unsafe_allow_html=True)

    # --- Render history ---
    for m in st.session_state.chat_msgs[-40:]:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="bubble-user">{m["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<div class="bubble-assist">{m["content"]}</div>', unsafe_allow_html=True)

    # --- Input at the bottom ---
    user_msg = st.chat_input("Ask about meals, swaps, allergens, macros, or grocery list…")

    if user_msg:
        # Domain guard: keep chat strictly about diet/nutrition
        _umsg = user_msg.strip()
        diet_terms = ["diet", "meal", "recipe", "protein", "carb", "fat", "kcal",
                      "calorie", "macro", "sodium", "sugar", "fiber", "vegetarian",
                      "vegan", "pescatarian", "gluten", "lactose", "allergen",
                      "plan", "breakfast", "lunch", "dinner", "snack", "grocery"]
        if not any(t in _umsg.lower() for t in diet_terms):
            polite_refusal = (
                "I’m your nutrition coach, so I stay focused on **diet & meal planning**. "
                "Try something like: *“High-protein vegetarian plan under 2300 mg sodium”*, "
                "*“Swap dinner and keep the rest”*, or *“Avoid peanuts and shellfish”*."
            )
            st.session_state.chat_msgs.append({"role": "user", "content": _umsg})
            st.session_state.chat_msgs.append({"role": "assistant", "content": polite_refusal})
            st.experimental_rerun()

        # Normal diet/nutrition query → call agent
        st.session_state.chat_msgs.append({"role": "user", "content": _umsg})

        ctx = {
            "profile": st.session_state.get("profile", {}),
            "targets": st.session_state.get("targets", {}),
            "caps":     st.session_state.get("caps", {}),
            "settings": {
                "k": st.session_state.get("search", {}).get("k", 60),
                "intent": st.session_state.get("search", {}).get("intent", "balanced high-protein"),
                "max_meal_kcal": st.session_state.get("caps", {}).get("max_meal_kcal", 1000),
            },
            "last_hits": st.session_state.get("last_hits", []),
            "last_plan": st.session_state.get("last_plan"),
        }

        reply = run_agent(_umsg, ctx) if llm_available() else (
            "LLM not configured. Set `OPENAI_API_KEY` to enable the coach chat."
        )
        st.session_state.chat_msgs.append({"role": "assistant", "content": reply})

        # Keep Home/Analysis in sync if tools updated hits/plan
        # (run_agent already writes to st.session_state.last_hits/last_plan)

        st.experimental_rerun()
