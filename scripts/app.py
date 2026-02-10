"""
Resume Ranker & Feedback System
================================
Streamlit UI that combines BERT-based cosine similarity with
Llama 3.1 reasoning to rank and evaluate resumes against a job description.

Run:  streamlit run scripts/app.py
"""

import streamlit as st
import time
from io import BytesIO

from pdf_processor import process_pdf
from embedding_engine import batch_analyze, get_device
from llama_ranker import rank_resumes, get_detailed_feedback


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Ranker",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { max-width: 1100px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .rank-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4361ee;
    }
    .rank-card h3 { margin-top: 0; }
    .strong-match { border-left-color: #2ec4b6; }
    .good-match   { border-left-color: #4361ee; }
    .partial-match{ border-left-color: #ff9f1c; }
    .weak-match   { border-left-color: #e71d36; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")

    llama_model = st.selectbox(
        "Llama Model (Ollama)",
        ["llama3.1", "llama3.1:8b", "llama3.1:70b", "llama3.2", "llama3.2:3b"],
        index=0,
        help="Choose the Ollama model. Make sure it is pulled locally.",
    )

    st.divider()
    st.markdown("### How it works")
    st.markdown("""
1. **Upload** a job description PDF and one or more resume PDFs.
2. **BERT** (all-MiniLM-L6-v2) computes semantic similarity scores between each resume and the JD.
3. **Llama 3.1** reads every resume, considers the BERT scores, and produces a reasoned ranking with per-resume feedback.
4. **Results** show both quantitative scores and qualitative analysis.
    """)

    st.divider()

    # Show device status so user knows if GPU is active
    device = get_device()
    if device == "cuda":
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"BERT device: GPU ({gpu_name})")
    else:
        st.warning(
            "BERT device: CPU (GPU unavailable)\n\n"
            "Your PyTorch build may lack kernels for your GPU.\n"
            "To fix, run:\n"
            "```\n"
            "pip install --pre torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/nightly/cu128\n"
            "```"
        )

    st.caption("Runs 100% locally on your machine. No data leaves your laptop.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("Resume Ranker")
st.markdown("Upload a job description and resumes to get AI-powered ranking with detailed feedback.")

col_jd, col_resumes = st.columns([1, 1], gap="large")

with col_jd:
    st.subheader("Job Description")
    jd_file = st.file_uploader(
        "Upload JD (PDF)",
        type=["pdf"],
        key="jd_upload",
        help="Upload the job description as a PDF file.",
    )
    if jd_file:
        st.success(f"Loaded: {jd_file.name}")

with col_resumes:
    st.subheader("Resumes")
    resume_files = st.file_uploader(
        "Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_upload",
        help="Upload one or more resumes as PDF files.",
    )
    if resume_files:
        st.success(f"Loaded {len(resume_files)} resume(s)")

st.divider()

# ---------------------------------------------------------------------------
# Analysis trigger
# ---------------------------------------------------------------------------
can_run = jd_file is not None and len(resume_files or []) > 0

if st.button("Analyze & Rank", type="primary", disabled=not can_run, use_container_width=True):

    # ------------------------------------------------------------------
    # Step 1: Extract PDF text
    # ------------------------------------------------------------------
    with st.status("Extracting text from PDFs...", expanded=True) as status:
        t0 = time.time()
        jd_data = process_pdf(BytesIO(jd_file.read()))
        jd_file.seek(0)
        st.write(f"JD: {jd_data['word_count']} words extracted")

        resumes_data = []
        for rf in resume_files:
            rdata = process_pdf(BytesIO(rf.read()))
            rf.seek(0)
            rdata["filename"] = rf.name
            resumes_data.append(rdata)
            st.write(f"{rf.name}: {rdata['word_count']} words extracted")

        status.update(label=f"Text extraction done ({time.time()-t0:.1f}s)", state="complete")

    # ------------------------------------------------------------------
    # Step 2: BERT similarity
    # ------------------------------------------------------------------
    with st.status("Computing BERT similarity scores...", expanded=True) as status:
        t0 = time.time()
        sim_results = batch_analyze(
            [{"filename": r["filename"], "sections": r["sections"], "word_count": r["word_count"]}
             for r in resumes_data],
            jd_data["cleaned_text"],
        )
        status.update(label=f"BERT analysis done ({time.time()-t0:.1f}s)", state="complete")

    # ------------------------------------------------------------------
    # Step 3: Llama ranking
    # ------------------------------------------------------------------
    with st.status("Llama is analyzing and ranking resumes...", expanded=True) as status:
        st.write("This may take a minute depending on model size and GPU speed.")
        t0 = time.time()
        llm_result = rank_resumes(
            jd_text=jd_data["cleaned_text"],
            resumes=[{"filename": r["filename"], "cleaned_text": r["cleaned_text"]}
                     for r in resumes_data],
            similarity_results=sim_results,
            model=llama_model,
        )
        status.update(label=f"Llama ranking done ({time.time()-t0:.1f}s)", state="complete")

    # ------------------------------------------------------------------
    # Store results in session state
    # ------------------------------------------------------------------
    st.session_state["sim_results"] = sim_results
    st.session_state["llm_result"] = llm_result
    st.session_state["resumes_data"] = resumes_data
    st.session_state["jd_data"] = jd_data
    st.session_state["llama_model"] = llama_model


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "llm_result" in st.session_state:
    sim_results = st.session_state["sim_results"]
    llm_result = st.session_state["llm_result"]
    resumes_data = st.session_state["resumes_data"]
    jd_data = st.session_state["jd_data"]

    # Handle errors
    if "error" in llm_result:
        st.error(f"LLM Error: {llm_result['error']}")
        if llm_result.get("raw_response"):
            with st.expander("Raw LLM response"):
                st.code(llm_result["raw_response"])
    else:
        # ==============================================================
        # BERT Scores Overview
        # ==============================================================
        st.header("BERT Similarity Scores")
        cols = st.columns(min(len(sim_results), 4))
        for i, sr in enumerate(sim_results):
            with cols[i % len(cols)]:
                st.metric(
                    label=sr["filename"][:30],
                    value=f"{sr['weighted_score']}%",
                    delta=f"Raw: {sr['raw_cosine_score']}%",
                )

        # Section-level breakdown
        with st.expander("Section-level BERT scores"):
            for sr in sim_results:
                st.markdown(f"**{sr['filename']}**")
                if sr.get("section_scores"):
                    for sec, score in sorted(sr["section_scores"].items(), key=lambda x: -x[1]):
                        bar_width = max(int(score), 1)
                        st.markdown(
                            f"`{sec:.<30s}` {score:.1f}%"
                        )
                        st.progress(min(score / 100, 1.0))
                st.divider()

        # ==============================================================
        # Llama Ranking
        # ==============================================================
        st.header("Llama-Powered Ranking")

        ranking = llm_result.get("final_ranking", [])
        for entry in ranking:
            rec = entry.get("recommendation", "").upper().replace(" ", "-").lower()
            css_class = "rank-card"
            if "strong" in rec:
                css_class += " strong-match"
            elif "good" in rec:
                css_class += " good-match"
            elif "partial" in rec:
                css_class += " partial-match"
            else:
                css_class += " weak-match"

            st.markdown(f"""
<div class="{css_class}">
    <h3>#{entry.get('rank', '?')} &mdash; {entry.get('filename', 'Unknown')}</h3>
    <p><strong>Match:</strong> {entry.get('overall_match_pct', 'N/A')}% &bull; 
       <strong>Recommendation:</strong> {entry.get('recommendation', 'N/A')}</p>
</div>
""", unsafe_allow_html=True)

            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown("**Strengths**")
                for s in entry.get("strengths", []):
                    st.markdown(f"- {s}")
            with col_w:
                st.markdown("**Weaknesses**")
                for w in entry.get("weaknesses", []):
                    st.markdown(f"- {w}")

            missing = entry.get("missing_requirements", [])
            if missing:
                st.markdown("**Missing Requirements**")
                for m in missing:
                    st.markdown(f"- {m}")

            feedback = entry.get("feedback", "")
            if feedback:
                st.info(f"**Feedback:** {feedback}")

            st.divider()

        # ==============================================================
        # Pairwise Comparisons
        # ==============================================================
        comparisons = llm_result.get("pairwise_comparisons", [])
        if comparisons:
            st.header("Head-to-Head Comparisons")
            for comp in comparisons:
                with st.expander(
                    f"{comp.get('higher_ranked', '?')}  vs  {comp.get('lower_ranked', '?')}"
                ):
                    st.markdown(comp.get("reasoning", "No reasoning provided."))

        # ==============================================================
        # Summary
        # ==============================================================
        summary = llm_result.get("summary", "")
        if summary:
            st.header("Overall Summary")
            st.markdown(summary)

        # ==============================================================
        # Deep-dive feedback per resume
        # ==============================================================
        st.header("Detailed Individual Feedback")
        st.caption("Click a tab to request an in-depth analysis for a single resume.")

        tabs = st.tabs([r["filename"][:30] for r in resumes_data])
        for i, tab in enumerate(tabs):
            with tab:
                cache_key = f"detailed_feedback_{resumes_data[i]['filename']}"
                if cache_key in st.session_state:
                    st.markdown(st.session_state[cache_key])
                else:
                    if st.button(f"Generate detailed feedback", key=f"fb_{i}"):
                        with st.spinner("Llama is writing detailed feedback..."):
                            fb = get_detailed_feedback(
                                jd_text=jd_data["cleaned_text"],
                                resume_text=resumes_data[i]["cleaned_text"],
                                resume_filename=resumes_data[i]["filename"],
                                model=st.session_state.get("llama_model", "llama3.1"),
                            )
                            st.session_state[cache_key] = fb
                            st.markdown(fb)

    # ==================================================================
    # Raw data explorer
    # ==================================================================
    with st.expander("Raw data (debug)"):
        st.json({"bert_results": sim_results, "llm_result": llm_result})
