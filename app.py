"""Streamlit UI for the RAG QA system with hallucination detection.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import tempfile
import traceback
from pathlib import Path

import streamlit as st

from src.config import (
    ENABLE_JUDGE_DEFAULT,
    NUM_SAMPLES_UI,
    UI_DEFAULT_TOP_K,
    UI_MAX_TOP_K,
    UI_TITLE,
)
from src.ui_helpers import (
    confidence_label,
    decision_style,
    format_detector_table,
    format_sources,
    highlight_citations,
    validate_upload,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title=UI_TITLE, layout="wide")


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load the RAGPipeline exactly once per Streamlit server process."""
    from src.pipeline import RAGPipeline

    return RAGPipeline()


def _load_pipeline():
    """Load the pipeline and return it; show spinner while loading."""
    with st.spinner("Loading models… (first run may take a few minutes)"):
        return get_pipeline()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(pipeline):
    """Render the sidebar controls and return settings."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Index stats
        stats = pipeline.index_stats()
        st.metric("Indexed files", stats["files"])
        st.metric("Total chunks", stats["chunks"])

        st.divider()

        # Retrieval controls
        top_k = st.slider(
            "Top-K chunks",
            min_value=1,
            max_value=UI_MAX_TOP_K,
            value=UI_DEFAULT_TOP_K,
        )
        use_bm25 = st.toggle("Hybrid BM25", value=False)
        use_judge = st.toggle("Factuality judge", value=ENABLE_JUDGE_DEFAULT)
        num_samples = st.slider(
            "SelfCheck samples",
            min_value=0,
            max_value=5,
            value=NUM_SAMPLES_UI,
        )

        st.divider()

        # Document upload
        st.subheader("📄 Add Documents")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )

        col1, col2 = st.columns(2)
        add_clicked = col1.button("Add to index")
        rebuild_confirm = col2.checkbox("Rebuild?")
        rebuild_clicked = col2.button("Rebuild index") if rebuild_confirm else False

        if add_clicked and uploaded_files:
            _handle_upload(pipeline, uploaded_files, rebuild=False)
        elif rebuild_clicked:
            _handle_upload(pipeline, uploaded_files or [], rebuild=True)

        st.divider()

        # History
        st.subheader("📜 Recent Questions")
        history = st.session_state.get("history", [])
        if history:
            for i, item in enumerate(reversed(history[-5:])):
                dec = item.get("decision", "")
                style = decision_style(dec)
                st.markdown(f"{style['emoji']} {item['query'][:60]}")
        else:
            st.caption("No questions yet.")

    return {
        "top_k": top_k,
        "use_bm25": use_bm25,
        "use_judge": use_judge,
        "num_samples": num_samples,
    }


def _handle_upload(pipeline, uploaded_files, rebuild: bool):
    """Save uploaded files to temp, call pipeline.ingest, report results."""
    if not uploaded_files and not rebuild:
        st.sidebar.warning("No files selected.")
        return

    valid_paths: list[Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for uf in uploaded_files:
            ok, msg = validate_upload(uf.name, uf.size)
            if not ok:
                st.sidebar.error(f"{uf.name}: {msg}")
                continue
            tmp_path = Path(tmpdir) / uf.name
            tmp_path.write_bytes(uf.getvalue())
            valid_paths.append(tmp_path)

        if valid_paths or rebuild:
            with st.spinner("Ingesting documents…"):
                try:
                    result = pipeline.ingest(valid_paths, rebuild=rebuild)
                    st.sidebar.success(
                        f"Added {result['added']} file(s). "
                        f"Index now has {result['chunks']} chunks."
                    )
                except Exception as exc:
                    st.sidebar.error(f"Ingestion failed: {exc}")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
def main():
    """Render the main UI."""
    # Try loading the pipeline
    try:
        pipeline = _load_pipeline()
    except FileNotFoundError as exc:
        st.error("**Model file not found.**")
        st.code(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to initialise the pipeline: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

    settings = sidebar(pipeline)

    # Title and description
    st.title(UI_TITLE)
    st.markdown(
        "Ask a question about your indexed documents. "
        "The system retrieves evidence, generates an answer, and "
        "verifies it with four hallucination detectors."
    )

    # Check for empty index
    stats = pipeline.index_stats()
    if stats["chunks"] == 0:
        st.info(
            "📂 **No documents indexed.** Add documents using the sidebar uploader, "
            "or place files in `data/raw/` and run: `python -m src.ingestion.pipeline`"
        )

    # Question input
    query = st.chat_input("Ask a question about your documents…")

    if query:
        try:
            with st.spinner("Thinking…"):
                output = pipeline.ask(
                    query,
                    top_k=settings["top_k"],
                    use_bm25=settings["use_bm25"],
                    use_judge=settings["use_judge"],
                    num_samples=settings["num_samples"],
                )
        except ValueError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:
            st.error(f"An error occurred: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            return

        # Save to session history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append(
            {
                "query": query,
                "decision": output.final.decision.value,
            }
        )

        # Store result for display
        st.session_state["last_output"] = output
        st.session_state["last_query"] = query

    # Display the last result
    _display_result()


def _display_result():
    """Render the answer, confidence, verification details, and sources."""
    if "last_output" not in st.session_state:
        return

    output = st.session_state["last_output"]
    query = st.session_state.get("last_query", "")
    final = output.final
    decision = final.decision.value
    style = decision_style(decision)

    # User question
    st.markdown(f"**Q:** {query}")

    # Decision banner
    if decision == "SHOW":
        st.success(f"{style['emoji']} **{style['label']}** — {highlight_citations(final.answer)}")
    elif decision == "WARN":
        st.warning(f"{style['emoji']} **{style['label']}** — {highlight_citations(final.answer)}")
        if final.warning:
            st.caption(final.warning)
    else:  # REFUSE
        st.error(f"{style['emoji']} **{style['label']}** — {final.answer}")
        with st.expander("Show the unverified draft"):
            st.markdown(highlight_citations(final.generated.answer))

    # Confidence meter
    conf = final.confidence
    label = confidence_label(conf)
    st.progress(conf, text=f"Confidence: {conf:.0%} ({label})")

    # Verification details
    with st.expander("🔍 Verification Details"):
        table = format_detector_table(final.report)
        if table:
            st.table(table)

            # Per-sentence scores
            if final.report and final.report.sentences:
                st.subheader("Per-sentence scores")
                for i, sentence in enumerate(final.report.sentences):
                    st.markdown(f"**Sentence {i+1}:** {sentence}")
                    for result in final.report.results:
                        if not result.skipped and i < len(result.sentence_scores):
                            score = result.sentence_scores[i]
                            st.caption(f"  {result.name}: {score:.2f}")
        else:
            st.info("No verification data (detection was skipped).")

    # Sources
    cited_ids = final.generated.cited_ids if final.generated else []
    sources = format_sources(output.chunks, cited_ids)
    if sources:
        with st.expander("📚 Sources"):
            for src_item in sources:
                icon = "📌" if src_item["is_cited"] else "📄"
                score_str = (
                    f" (score: {src_item['score']:.4f})"
                    if src_item["score"] is not None
                    else ""
                )
                st.markdown(
                    f"{icon} **[{src_item['id']}] {src_item['filename']}**{score_str}"
                )
                st.caption(src_item["text"])

    # Timing
    t = output.timings
    timing_parts = [f"{stage}: {secs:.2f}s" for stage, secs in t.items()]
    st.caption("⏱ " + " | ".join(timing_parts))


if __name__ == "__main__":
    main()

