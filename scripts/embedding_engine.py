
import os
import warnings
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



def _resolve_device() -> str:
    """
    Return "cuda" only if we can actually run a kernel on the GPU.
    Otherwise fall back to "cpu" and print a helpful message.
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        # Attempt a tiny matmul -- this triggers the actual kernel dispatch
        # and will throw RuntimeError if no kernel image exists for the arch.
        a = torch.randn(2, 2, device="cuda")
        _ = a @ a
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[embedding_engine] Using GPU: {gpu_name}")
        return "cuda"
    except RuntimeError as exc:
        warnings.warn(
            f"\n{'='*70}\n"
            f"CUDA detected but kernel launch failed:\n  {exc}\n\n"
            f"This usually means your GPU architecture (e.g. Blackwell / sm_120)\n"
            f"is newer than the CUDA kernels shipped with your PyTorch build.\n\n"
            f"FIX: Install a nightly or updated PyTorch that supports your GPU:\n"
            f"  pip install --pre torch torchvision torchaudio "
            f"--index-url https://download.pytorch.org/whl/nightly/cu128\n\n"
            f"Falling back to CPU for now (still fast for this model).\n"
            f"{'='*70}\n"
        )
        # Make sure we don't accidentally hit CUDA again
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"


_device: str | None = None


def get_device() -> str:
    global _device
    if _device is None:
        _device = _resolve_device()
    return _device


# ---------------------------------------------------------------------------
# Model singleton so it loads once per session
# ---------------------------------------------------------------------------
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = get_device()
        print(f"[embedding_engine] Loading all-MiniLM-L6-v2 on '{device}' ...")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _model


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_embedding(text: str) -> np.ndarray:
    """Return a 384-dim embedding for the given text."""
    model = get_model()
    return model.encode(text, convert_to_numpy=True, show_progress_bar=False)


def compute_cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D embeddings (returns 0-100 scale)."""
    a = embedding_a.reshape(1, -1)
    b = embedding_b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0]) * 100


# ---------------------------------------------------------------------------
# Section-level analysis
# ---------------------------------------------------------------------------

SECTION_WEIGHTS = {
    "Skills":                  0.30,
    "Technical Skills":        0.30,
    "Experience":              0.25,
    "Work Experience":         0.25,
    "Professional Experience": 0.25,
    "Education":               0.15,
    "Projects":                0.15,
    "Summary":                 0.10,
    "Objective":               0.10,
    "Profile":                 0.10,
    "Certifications":          0.10,
    "Header":                  0.05,
    "full_text":               1.00,   # fallback when no sections detected
}


def analyze_resume(resume_sections: dict, jd_text: str) -> dict:
    """
    Compare each resume section against the full job description.
    Returns per-section scores, a weighted overall score, and
    the raw overall (full-text) cosine score.
    """
    jd_emb = compute_embedding(jd_text)

    section_scores = {}
    weighted_sum = 0.0
    weight_sum = 0.0

    for section_name, section_text in resume_sections.items():
        if not section_text.strip():
            continue
        sec_emb = compute_embedding(section_text)
        score = compute_cosine_similarity(sec_emb, jd_emb)
        section_scores[section_name] = round(score, 2)

        weight = SECTION_WEIGHTS.get(section_name, 0.05)
        weighted_sum += score * weight
        weight_sum += weight

    weighted_overall = round(weighted_sum / weight_sum, 2) if weight_sum else 0.0

    # Also compute a single full-text similarity
    full_text = " ".join(resume_sections.values())
    full_emb = compute_embedding(full_text)
    raw_overall = round(compute_cosine_similarity(full_emb, jd_emb), 2)

    return {
        "section_scores": section_scores,
        "weighted_score": weighted_overall,
        "raw_cosine_score": raw_overall,
    }


def batch_analyze(resumes: list[dict], jd_text: str) -> list[dict]:
    """
    Analyze a list of resume dicts (each containing 'sections' and 'filename').
    Returns a list sorted by weighted_score descending.
    """
    results = []
    for resume in resumes:
        analysis = analyze_resume(resume["sections"], jd_text)
        results.append({
            "filename": resume["filename"],
            "word_count": resume.get("word_count", 0),
            **analysis,
        })
    results.sort(key=lambda r: r["weighted_score"], reverse=True)
    return results
