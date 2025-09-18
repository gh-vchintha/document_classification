# ui.py
# Streamlit UI for the local demo of the Combined OCR + Classification pipeline
# Run with:  streamlit run ui.py

import os
import sys
import json
import time
import uuid
import queue
import shutil
import types
import threading
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# ======================================================================================
# Import HandwritingExtractor directly from local file
# ======================================================================================
from ocr_only_v2 import HandwritingExtractor

# ======================================================================================
# Constants & dirs
# ======================================================================================
APP_TITLE = "OCR + Classification (Local Demo)"
CACHE_DIR = os.path.abspath(".app_cache")
UPLOAD_DIR = os.path.join(CACHE_DIR, "uploads")
ARTIFACTS_DIR = os.path.join(CACHE_DIR, "artifacts")
LOGS_DIR = os.path.join(CACHE_DIR, "logs")
for d in (CACHE_DIR, UPLOAD_DIR, ARTIFACTS_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

# Chat/QA defaults
CHAT_OPENAI_MODEL_DEFAULT = os.environ.get("CHAT_OPENAI_MODEL", "gpt-4o-mini")

# ======================================================================================
# Logging to Streamlit (via queue) — safe for threads
# ======================================================================================
class StreamHandlerToBuffer(logging.Handler):
    """Logging handler that writes records into a Queue for the UI to read."""
    def __init__(self, buffer: "queue.Queue[str]", level: int = logging.INFO):
        super().__init__(level)
        self.buffer = buffer
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.buffer.put_nowait(msg)
        except Exception:
            pass

# ======================================================================================
# Session state model
# ======================================================================================
@dataclass
class PDFItem:
    id: str
    name: str
    path: str
    size: int
    status: str = "Queued"  # Queued | Running | Done | Error
    pages_total: Optional[int] = None
    pages_done: int = 0
    eta_seconds: Optional[float] = None
    logs: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None

def _init_state():
    if "job" not in st.session_state:
        st.session_state.job = {
            "id": None,
            "status": "idle",  # idle|running|cancelled|done|error
            "start_ts": None,
            "end_ts": None,
        }
    if "pdfs" not in st.session_state:
        st.session_state.pdfs: List[PDFItem] = []
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any]] = []
    if "log_queue" not in st.session_state:
        st.session_state.log_queue = queue.Queue()
    if "log_lines" not in st.session_state:
        st.session_state.log_lines: List[str] = []
    if "runner_thread" not in st.session_state:
        st.session_state.runner_thread: Optional[threading.Thread] = None
    if "cancel_flag" not in st.session_state:
        st.session_state.cancel_flag = threading.Event()
    if "seen_hashes" not in st.session_state:
        st.session_state.seen_hashes = set()
    # cross-thread signaling (worker never touches st.session_state)
    if "worker_result" not in st.session_state:
        st.session_state.worker_result = None  # dict set at start
    if "worker_done_event" not in st.session_state:
        st.session_state.worker_done_event = None  # threading.Event set at start
    if "log_handler" not in st.session_state:
        st.session_state.log_handler = None

# ======================================================================================
# Helpers
# ======================================================================================
def _human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def _safe_json(value: Any) -> Any:
    if isinstance(value, str):
        v = value.strip()
        try:
            return json.loads(v)
        except Exception:
            if v.startswith("```"):
                v = v.strip().strip("`")
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return value
    return value

def _attach_logging(buffer: "queue.Queue[str]") -> logging.Handler:
    handler = StreamHandlerToBuffer(buffer)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return handler

def _detach_logging(handler: logging.Handler):
    try:
        logging.getLogger().removeHandler(handler)
    except Exception:
        pass

# JSON default for downloads (handles pandas/numpy/time)
def _json_default(o: Any):
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        np = None  # type: ignore
        pd = None  # type: ignore
    if pd is not None and isinstance(o, getattr(pd, "Timestamp", ())):
        return o.isoformat()
    if np is not None and isinstance(o, getattr(np, "integer", ())):
        return int(o)
    if np is not None and isinstance(o, getattr(np, "floating", ())):
        return float(o)
    if np is not None and isinstance(o, getattr(np, "bool_", ())):
        return bool(o)
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    return str(o)

def _norm_patient_name(name: Optional[str]) -> Optional[str]:
    if not name or not isinstance(name, str):
        return None
    s = name.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s or None

def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

def dataframe_full_width(df, **kwargs):
    """
    Prefer the new API (width='stretch'); fall back to use_container_width=True
    on older Streamlit builds that require an integer width.
    """
    try:
        return st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(df, use_container_width=True, **kwargs)

# ---------- Stable artifacts for download buttons ----------
import hashlib
def _sanitize(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)
    return keep.strip("_") or "file"

def _write_artifacts(name: str, rows: List[Dict[str, Any]]) -> tuple[str, str]:
    """
    Write CSV & JSONL once into ARTIFACTS_DIR and return their file paths.
    Rewrites only if content changed (hash).
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    base = _sanitize(name)
    csv_path = os.path.join(ARTIFACTS_DIR, f"{base}.csv")
    jsonl_path = os.path.join(ARTIFACTS_DIR, f"{base}.jsonl")

    # Build strings once
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    jsonl_str = "\n".join(json.dumps(r, ensure_ascii=False, default=_json_default) for r in rows)
    jsonl_bytes = jsonl_str.encode("utf-8")

    # Idempotent write using content hash
    def _safe_write(path: str, data: bytes):
        h_new = hashlib.sha1(data).hexdigest()
        h_old = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                h_old = hashlib.sha1(f.read()).hexdigest()
        if h_new != h_old:
            with open(path, "wb") as f:
                f.write(data)

    _safe_write(csv_path, csv_bytes)
    _safe_write(jsonl_path, jsonl_bytes)
    return csv_path, jsonl_path

# Flatten results to a simple list[dict] regardless of pipeline shape
def _flatten_results(results: List[Any]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for r in results or []:
        if isinstance(r, dict):
            flat.append(r)
        elif isinstance(r, list):
            for x in r:
                if isinstance(x, dict):
                    flat.append(x)
    return flat

# Save uploads
def _save_uploaded_files(files: List[types.SimpleNamespace]) -> List[PDFItem]:
    import hashlib
    items: List[PDFItem] = []
    for f in files:
        data = f.read()
        digest = hashlib.sha1(data).hexdigest()
        if digest in st.session_state.seen_hashes:
            continue  # skip exact duplicate content
        st.session_state.seen_hashes.add(digest)
        fid = uuid.uuid4().hex[:8]
        safe_name = f.name or "upload.pdf"
        out_path = os.path.join(UPLOAD_DIR, f"{fid}_{safe_name}")
        with open(out_path, "wb") as w:
            w.write(data)
        items.append(PDFItem(id=fid, name=safe_name, path=out_path, size=len(data)))
    return items

# ======================================================================================
# Worker — NEVER touches st.session_state
# ======================================================================================
def _run_pipeline_in_thread(
        pdf_paths: List[str],
        is_classify: bool,
        max_pages: Optional[int],
        settings_env: Dict[str, str],
        extractor_cls,
        result_container: Dict[str, Any],
        done_event: threading.Event,
):
    """Worker thread: writes to result_container and sets done_event; no Streamlit calls."""
    orig_env: Dict[str, Optional[str]] = {}
    try:
        for k, v in settings_env.items():
            orig_env[k] = os.environ.get(k)
            os.environ[k] = str(v)

        extractor = extractor_cls(aws_region=os.environ.get("AWS_REGION", "us-east-1"))
        # Your extractor signature: batch_process_pdfs(pdf_paths, is_classify=True, max_pages_to_process=int)
        rows = extractor.batch_process_pdfs(
            pdf_paths,
            is_classify=is_classify,
            max_pages_to_process=max_pages,
        )
        result_container["rows"] = rows or []
        result_container["status"] = "done"
        result_container["error"] = None
    except Exception as e:
        logging.getLogger(__name__).exception("Runner error: %s", e)
        result_container["rows"] = []
        result_container["status"] = "error"
        result_container["error"] = str(e)
    finally:
        for k, v in orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        done_event.set()

# ======================================================================================
# Sidebar
# ======================================================================================
def sidebar_controls():
    with st.sidebar:
        st.title("⚙️ Controls")
        run_mode = st.radio("Run Mode", ["Real Run", "Dry Run"], index=0,
                            help="Dry Run generates fake results for demo without calling models.")

        st.subheader("Project Root")
        default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        proj_root = st.text_input(
            "Add this folder to sys.path (so imports like 'poc.ocr_only_v2' work)",
            value=default_root,
            help="Point this to the parent folder that contains the 'poc' directory.",
        )
        st.session_state["project_root"] = proj_root

        st.subheader("Upload PDFs")
        uploads = st.file_uploader("Select one or more PDFs", type=["pdf"], accept_multiple_files=True)
        add_to_queue = st.button("➕ Add to queue", help="Click to add selected files to the list below.")

        st.subheader("Knobs")
        col1, col2 = st.columns(2)
        with col1:
            dpi = st.number_input("PDF_RENDER_DPI", value=350, min_value=72, max_value=600, step=10)
            fmt = st.selectbox("PDF_RENDER_FMT", ["jpeg", "png"], index=1)
            max_pages = st.number_input("Max pages (0 = 100)", value=15, min_value=0, step=1)
            log_level = st.selectbox("LOG_LEVEL", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
        with col2:
            max_workers = st.number_input("PDF_OCR_MAX_WORKERS", value=6, min_value=1, max_value=64)
            max_ocr = st.number_input("PDF_OCR_MAX_OCR", value=4, min_value=1, max_value=128)
            max_pdfs = st.number_input("PDF_OCR_MAX_PDFS", value=3, min_value=1, max_value=64)
            ocr_retries = st.number_input("PDF_OCR_MAX_RETRIES", value=4, min_value=0, max_value=10)

        col3, _ = st.columns(2)
        with col3:
            cls_retries = st.number_input("CLS_MAX_RETRIES", value=3, min_value=0, max_value=10)

        start = st.button("▶️ Start Job", type="primary")
        cancel = st.button("⏹️ Cancel", disabled=(st.session_state.job["status"] != "running"))
        reset = st.button("♻️ Reset State")

        # Paths & chat settings
        st.subheader("Paths & Chat")
        artifacts_dir = st.text_input("ARTIFACTS_DIR", value=ARTIFACTS_DIR)
        chat_model = st.text_input("CHAT_OPENAI_MODEL", value=CHAT_OPENAI_MODEL_DEFAULT)

        settings_env = {
            "PDF_RENDER_DPI": str(dpi),
            "PDF_RENDER_FMT": str(fmt),
            "PDF_OCR_MAX_WORKERS": str(max_workers),
            "PDF_OCR_MAX_OCR": str(max_ocr),
            "PDF_OCR_MAX_PDFS": str(max_pdfs),
            "PDF_OCR_MAX_RETRIES": str(ocr_retries),
            "CLS_MAX_RETRIES": str(cls_retries),
            "LOG_LEVEL": str(log_level),
            "ARTIFACTS_DIR": str(artifacts_dir),
            "CHAT_OPENAI_MODEL": str(chat_model),
        }

        return {
            "mode": run_mode,
            "uploads": uploads,
            "add_to_queue": add_to_queue,
            "start": start,
            "cancel": cancel,
            "reset": reset,
            "settings_env": settings_env,
            "max_pages": int(max_pages) if max_pages > 0 else 100,
            "project_root": proj_root,
            "artifacts_dir": artifacts_dir,
            "chat_model": chat_model,
        }

# ======================================================================================
# Tabs
# ======================================================================================
def tab_dashboard():
    st.subheader("Selected PDFs")
    if not st.session_state.pdfs:
        st.info("Upload PDFs in the sidebar to get started.")
        return
    data = {
        "filename": [p.name for p in st.session_state.pdfs],
        "size": [_human_size(p.size) for p in st.session_state.pdfs],
        "status": [p.status for p in st.session_state.pdfs],
    }
    dataframe_full_width(pd.DataFrame(data), hide_index=True)

    total = len(st.session_state.pdfs)
    done = sum(1 for p in st.session_state.pdfs if p.status == "Done")
    error = sum(1 for p in st.session_state.pdfs if p.status == "Error")
    running = sum(1 for p in st.session_state.pdfs if p.status == "Running")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queued", total - (done + error + running))
    c2.metric("Running", running)
    c3.metric("Done", done)
    c4.metric("Error", error)

    if total:
        frac = (done + error) / total
        st.progress(frac)

    # ---- Last run artifacts (no 'classification' assumptions) ----
    results_flat = _flatten_results(st.session_state.results)
    if results_flat:
        with st.expander("Last Run Artifacts"):
            df = pd.DataFrame(results_flat)
            dataframe_full_width(df, hide_index=True)

            csv_path, jsonl_path = _write_artifacts("all_results", results_flat)
            with open(csv_path, "rb") as f:
                csv_bytes = f.read()
            with open(jsonl_path, "rb") as f:
                jsonl_bytes = f.read()

            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                key="dl_csv_dashboard_all",
            )
            st.download_button(
                "Download JSONL",
                data=jsonl_bytes,
                file_name="results.jsonl",
                mime="application/json",
                key="dl_jsonl_dashboard_all",
            )

def tab_logs():
    st.subheader("Live Logs")
    drained = 0
    try:
        while True:
            line = st.session_state.log_queue.get_nowait()
            st.session_state.log_lines.append(line)
            drained += 1
    except queue.Empty:
        pass
    if drained:
        st.session_state.log_lines = st.session_state.log_lines[-2000:]
    if not st.session_state.log_lines:
        st.info("No logs yet. Start a job to see streaming logs.")
    else:
        st.code("\n".join(st.session_state.log_lines), language="text")

def tab_per_pdf():
    st.subheader("Per-PDF Details")
    if not st.session_state.pdfs:
        st.info("Upload PDFs first.")
        return

    left, right = st.columns([1, 2])
    with left:
        selected_name = st.radio("Choose a PDF", [p.name for p in st.session_state.pdfs], key="pdf_select")
        sel = next((p for p in st.session_state.pdfs if p.name == selected_name), None)
        if sel:
            st.write(f"**Status:** {sel.status}")
            st.write(f"**File:** `{sel.path}`")
            st.write(f"**Size:** {_human_size(sel.size)}")
            if sel.pages_total:
                st.write(f"**Pages:** {sel.pages_done}/{sel.pages_total}")
            if sel.eta_seconds:
                st.write(f"**ETA:** ~{int(sel.eta_seconds)}s")

    with right:
        rows = _flatten_results(st.session_state.results)
        if not rows:
            st.info("Run first to see per-PDF outputs.")
            return

        # Show all rows belonging to this filename
        subset = [r for r in rows if (r.get("filename") == selected_name)]
        if not subset:
            st.info("No rows found for this file in the results.")
            return

        st.write("**Rows for this file**")
        dataframe_full_width(pd.DataFrame(subset), hide_index=True)

        # Per-file downloads (stable keys & files)
        base = f"results_{_sanitize(selected_name)}"
        csv_path, jsonl_path = _write_artifacts(base, subset)
        with open(csv_path, "rb") as f:
            csv_bytes = f.read()
        with open(jsonl_path, "rb") as f:
            jsonl_bytes = f.read()

        c1, c2 = st.columns(2)
        c1.download_button(
            "Download this file's JSONL",
            data=jsonl_bytes,
            file_name=f"{selected_name}_results.jsonl",
            mime="application/json",
            key=f"dl_jsonl_file_{base}",
        )
        c2.download_button(
            "Download this file's CSV",
            data=csv_bytes,
            file_name=f"{selected_name}_results.csv",
            mime="text/csv",
            key=f"dl_csv_file_{base}",
        )

def tab_results():
    st.subheader("Results Table")
    rows = _flatten_results(st.session_state.results)
    if not rows:
        st.info("No results yet.")
        return

    df = pd.DataFrame(rows)
    dataframe_full_width(df, hide_index=True)

    # Downloads (all results) — stable keys & files
    csv_path, jsonl_path = _write_artifacts("all_results", rows)
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    with open(jsonl_path, "rb") as f:
        jsonl_bytes = f.read()

    c1, c2 = st.columns(2)
    c1.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="results.csv",
        mime="text/csv",
        key="dl_csv_results_all",
    )
    c2.download_button(
        "Download JSONL",
        data=jsonl_bytes,
        file_name="results.jsonl",
        mime="application/json",
        key="dl_jsonl_results_all",
    )

def _list_available_artifacts(artifacts_dir: str) -> List[str]:
    try:
        return [f[:-5] for f in os.listdir(artifacts_dir) if f.endswith('.json')]
    except Exception:
        return []

def _load_artifact(artifacts_dir: str, filename: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(artifacts_dir, f"{filename}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _patient_pages_from_artifact(artifact: Dict[str, Any], patient_name: str) -> List[int]:
    pages = []
    pnorm = _norm_patient_name(patient_name) or "unknown"
    for entry in artifact.get("patient_index", []) or []:
        if entry.get("patient_name_norm") == pnorm:
            spans = entry.get("page_spans") or []
            for a, b in spans:
                pages.extend(list(range(int(a), int(b) + 1)))
            break
    return sorted(set(pages))

def _score_chunks_by_overlap(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    qtok = set(re.findall(r"\w+", question.lower()))
    for ch in chunks:
        text = (ch.get("text") or "").lower()
        ttok = set(re.findall(r"\w+", text))
        overlap = len(qtok & ttok)
        ch["score"] = overlap
    return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

def _chat_call(model: str, system_prompt: str, user_prompt: str) -> str:
    if OpenAI is None:
        return "I don't know based on the provided documents."
    try:
        client = OpenAI()
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return getattr(resp, "output_text", "").strip() or "I don't know based on the provided documents."
    except Exception:
        return "I don't know based on the provided documents."

def tab_chat(actions: Dict[str, Any]):
    st.subheader("Patient Chat (Grounded)")
    artifacts_dir = actions.get("artifacts_dir") or ARTIFACTS_DIR
    os.makedirs(artifacts_dir, exist_ok=True)

    rows = _flatten_results(st.session_state.results)
    if not rows:
        st.info("No results yet. Run the pipeline first.")
        return

    processed = sorted({r.get("filename") for r in rows if r.get("filename")})
    available = [fn for fn in processed if os.path.exists(os.path.join(artifacts_dir, f"{fn}.json"))]
    if not available:
        st.warning("No artifacts found yet. After a run completes, artifacts will be saved to ARTIFACTS_DIR for chat.")
        st.write(f"ARTIFACTS_DIR: `{artifacts_dir}`")
        return

    sel_file = st.selectbox("Choose a processed PDF", available, key="chat_pdf_select")
    artifact = _load_artifact(artifacts_dir, sel_file)
    if not artifact:
        st.error("Failed to load artifact for this file.")
        return

    patients = []
    for p in (artifact.get("patient_index") or []):
        name = p.get("patient_name", "Unknown")
        if not isinstance(name, list):
            patients.append(name)
    patients = sorted(set(patients)) or ["Unknown"]
    sel_patient = st.selectbox("Choose a patient", patients, key="chat_patient_select")

    # Summary
    page_nums = _patient_pages_from_artifact(artifact, sel_patient)
    st.caption(f"Pages attributed to patient: {page_nums if page_nums else 'Unknown'}")

    # Prepare chat state
    pkey = _norm_patient_name(sel_patient) or "unknown"
    chat_key = f"chat::{sel_file}::{pkey}"
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if chat_key not in st.session_state.chat_histories:
        # try load persisted
        slug = _slugify(pkey)
        log_path = os.path.join(artifacts_dir, f"{sel_file}__{slug}__chat.jsonl")
        hist: List[Dict[str, Any]] = []
        try:
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            hist.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass
        st.session_state.chat_histories[chat_key] = hist

    history = st.session_state.chat_histories[chat_key]
    for m in history:
        with st.chat_message(m.get("role", "assistant")):
            st.write(m.get("content", ""))
            if m.get("sources"):
                st.caption(f"Sources: {m.get('sources')}")

    question = st.chat_input("Ask a question about this patient's document…")
    if question:
        # Append user message
        user_msg = {"role": "user", "content": question}
        history.append(user_msg)
        with st.chat_message("user"):
            st.write(question)

        # Retrieval: build chunks from patient pages only
        page_map = {int(p.get("page_no")): (p.get("raw_text") or "") for p in (artifact.get("pages") or []) if isinstance(p.get("page_no"), int)}
        candidate_pages = page_nums or sorted(page_map.keys())
        chunks = [{"page": p, "text": page_map.get(p, "")} for p in candidate_pages]
        ranked = _score_chunks_by_overlap(question, chunks)
        topk = ranked[: min(6, len(ranked))]
        used_pages = [c.get("page") for c in topk if c.get("text")]
        context = "\n\n".join([f"Page {c['page']}\n{c['text']}" for c in topk if c.get("text")])
        if not context.strip():
            context = "No usable context found. If you cannot answer from context, reply with: I don't know based on the provided documents."

        system_prompt = (
            "You are a careful, grounded assistant. Answer only using the provided context. "
            "If the answer is not clearly contained in the context, respond exactly: I don't know based on the provided documents. "
            "Be concise and cite page numbers from the context when possible."
        )
        user_prompt = f"Question: {question}\n\nContext:\n{context}"

        model = actions.get("chat_model") or CHAT_OPENAI_MODEL_DEFAULT
        with st.chat_message("assistant"):
            answer = _chat_call(model, system_prompt, user_prompt)
            st.write(answer)
            if used_pages:
                st.caption(f"Sources: pages {used_pages}")

        asst_msg = {"role": "assistant", "content": answer, "sources": used_pages}
        history.append(asst_msg)

        # Persist chat log
        try:
            slug = _slugify(pkey)
            log_path = os.path.join(artifacts_dir, f"{sel_file}__{slug}__chat.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(user_msg, ensure_ascii=False) + "\n")
                f.write(json.dumps(asst_msg, ensure_ascii=False) + "\n")
        except Exception:
            pass

def tab_settings():
    st.subheader("Paths & Presets")
    st.write(f"**Cache Dir:** `{CACHE_DIR}`")
    st.write(f"**Uploads:** `{UPLOAD_DIR}`")
    st.write(f"**Artifacts:** `{ARTIFACTS_DIR}`")
    st.write(f"**Logs:** `{LOGS_DIR}`")
    if st.button("🧹 Clean cache (uploads, logs, artifacts)"):
        try:
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            for d in (CACHE_DIR, UPLOAD_DIR, ARTIFACTS_DIR, LOGS_DIR):
                os.makedirs(d, exist_ok=True)
            st.success("Cache cleaned.")
        except Exception as e:
            st.error(f"Failed to clean cache: {e}")

def tab_debug():
    st.subheader("Debug Info")
    st.write("Thread running:", bool(st.session_state.runner_thread and st.session_state.runner_thread.is_alive()))
    st.write("Job:", st.session_state.job)
    st.write("# PDFs:", len(st.session_state.pdfs))
    st.write("# Log lines:", len(st.session_state.log_lines))

# ======================================================================================
# Finalization helper — addresses “backend done but UI not updating” race
# ======================================================================================
def _finalize_if_done() -> bool:
    """
    Finalize the job if the worker is finished.
    We consider the job done if either:
      - worker_done_event.is_set(), or
      - the runner_thread is no longer alive (fallback).
    Returns True if we finalized and triggered a rerun.
    """
    res = st.session_state.worker_result or {}
    is_done = False

    if st.session_state.worker_done_event and st.session_state.worker_done_event.is_set():
        is_done = True
    elif st.session_state.runner_thread and not st.session_state.runner_thread.is_alive():
        # Fallback in case the event was missed or the object changed across reruns
        is_done = True

    if is_done:
        rows = res.get("rows")
        if rows is None:
            rows = []
        st.session_state.results = rows
        st.session_state.job["status"] = "error" if res.get("status") == "error" else "done"
        st.session_state.job["end_ts"] = time.time()
        if st.session_state.log_handler:
            _detach_logging(st.session_state.log_handler)
            st.session_state.log_handler = None
        st.rerun()
        return True

    return False

# ======================================================================================
# Main
# ======================================================================================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title("Correspondence Classification")
    _init_state()

    actions = sidebar_controls()

    # Reset
    if actions["reset"]:
        st.session_state.clear()
        st.rerun()

    # Uploads — only when user explicitly clicks "Add to queue"
    if actions["add_to_queue"] and actions["uploads"]:
        new_items = _save_uploaded_files(actions["uploads"])  # hashes prevent dupes
        if new_items:
            st.session_state.pdfs.extend(new_items)
            st.success(f"Added {len(new_items)} file(s) to queue.")
        else:
            st.info("No new files (duplicates were ignored).")

    tabs = st.tabs(["Dashboard", "Live Logs", "Per-PDF", "Results", "Chat", "Settings", "Debug"])
    with tabs[0]:
        tab_dashboard()
    with tabs[1]:
        tab_logs()
    with tabs[2]:
        tab_per_pdf()
    with tabs[3]:
        tab_results()
    with tabs[4]:
        tab_chat(actions)
    with tabs[5]:
        tab_settings()
    with tabs[6]:
        tab_debug()

    # Start / Cancel
    if actions["start"] and st.session_state.job["status"] != "running":
        if not st.session_state.pdfs:
            st.warning("Please upload at least one PDF.")
        else:
            st.session_state.job = {
                "id": uuid.uuid4().hex[:8],
                "status": "running",
                "start_ts": time.time(),
                "end_ts": None,
            }
            st.session_state.cancel_flag.clear()

            handler = _attach_logging(st.session_state.log_queue)
            st.session_state.log_handler = handler

            if actions["mode"] == "Dry Run":
                # Minimal rows (no classification)
                rows = []
                for p in st.session_state.pdfs:
                    p.status = "Done"
                    rows.append({
                        "filename": p.name,
                        "job_run_time": pd.Timestamp.utcnow().isoformat(),
                        "mock": True,
                    })
                st.session_state.results = rows
                st.session_state.job["status"] = "done"
                st.session_state.job["end_ts"] = time.time()
                if st.session_state.log_handler:
                    _detach_logging(st.session_state.log_handler)
                    st.session_state.log_handler = None
                st.rerun()
            else:
                pdf_paths = [p.path for p in st.session_state.pdfs]

                # Prepare thread-safe result signaling
                res = {"status": "running", "rows": None, "error": None}
                done_event = threading.Event()
                st.session_state.worker_result = res
                st.session_state.worker_done_event = done_event

                t = threading.Thread(
                    target=_run_pipeline_in_thread,
                    args=(
                        pdf_paths, True, actions["max_pages"],
                        actions["settings_env"], HandwritingExtractor,
                        res, done_event
                    ),
                    daemon=True,
                    name="ocr-classify-worker",
                )
                st.session_state.runner_thread = t
                t.start()

    if actions["cancel"] and st.session_state.runner_thread and st.session_state.runner_thread.is_alive():
        st.session_state.cancel_flag.set()
        st.session_state.job["status"] = "cancelled"
        if st.session_state.worker_result is not None:
            st.session_state.worker_result["status"] = "cancelled"

    # While running, poll worker completion and finalize in the main thread
    if st.session_state.job["status"] == "running":
        # Try to finalize if the worker has finished (event or thread dead)
        if not _finalize_if_done():
            # Small pause to avoid rapid-fire reruns that can starve the worker
            st.toast("Job running… logs will update live.")
            time.sleep(0.35)
            st.rerun()

if __name__ == "__main__":
    main()
