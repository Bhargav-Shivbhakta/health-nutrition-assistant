# =========================
# Section 1: Bootstrapping
# =========================
# - Imports & Streamlit page config
# - Robust discovery of pipeline_config.json
# - Cached pipeline loader (model by name + FAISS + slim meta)
# - Hard validation of artifacts so failures are obvious

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
    candidates: List[str | Path | None] = [
        env_p,
        "data/processed/metadata/pipeline_config.json",  # <— recommended
        "data/metadata/pipeline_config.json",
        "data/processed/pipeline_config.json",
        "pipeline_config.json",
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
