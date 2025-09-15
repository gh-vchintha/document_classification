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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

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

def dataframe_full_width(df, **kwargs):
    """
    Prefer the new API (width='stretch'); fall back to use_container_width=True
    on older Streamlit builds that require an integer width.
    """
    try:
        # New API (may raise TypeError on older versions)
        return st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        # Old API
        return st.dataframe(df, use_container_width=True, **kwargs)


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

        settings_env = {
            "PDF_RENDER_DPI": str(dpi),
            "PDF_RENDER_FMT": str(fmt),
            "PDF_OCR_MAX_WORKERS": str(max_workers),
            "PDF_OCR_MAX_OCR": str(max_ocr),
            "PDF_OCR_MAX_PDFS": str(max_pdfs),
            "PDF_OCR_MAX_RETRIES": str(ocr_retries),
            "CLS_MAX_RETRIES": str(cls_retries),
            "LOG_LEVEL": str(log_level),
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

    # st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")

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
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            jsonl_bytes = "\n".join(
                json.dumps(row, ensure_ascii=False, default=_json_default)
                for row in results_flat
            ).encode("utf-8")

            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                key="dl_csv_dashboard",
            )
            st.download_button(
                "Download JSONL",
                data=jsonl_bytes,
                file_name="results.jsonl",
                mime="application/json",
                key="dl_jsonl_dashboard",
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
        # st.dataframe(pd.DataFrame(subset), width="stretch", hide_index=True)
        dataframe_full_width(pd.DataFrame(subset), hide_index=True)


        # Per-file downloads (unique keys per file)
        jsonl_bytes = "\n".join(
            json.dumps(r, ensure_ascii=False, default=_json_default) for r in subset
        ).encode("utf-8")
        csv_bytes = pd.DataFrame(subset).to_csv(index=False).encode("utf-8")

        c1, c2 = st.columns(2)
        c1.download_button(
            "Download this file's JSONL",
            data=jsonl_bytes,
            file_name=f"{selected_name}_results.jsonl",
            mime="application/json",
            key=f"dl_jsonl_{uuid.uuid4().hex[:8]}",
        )
        c2.download_button(
            "Download this file's CSV",
            data=csv_bytes,
            file_name=f"{selected_name}_results.csv",
            mime="text/csv",
            key=f"dl_csv_{uuid.uuid4().hex[:8]}",
        )

def tab_results():
    st.subheader("Results Table")
    rows = _flatten_results(st.session_state.results)
    if not rows:
        st.info("No results yet.")
        return

    df = pd.DataFrame(rows)
    dataframe_full_width(df, hide_index=True)

    # Downloads (all results)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    jsonl_bytes = "\n".join(
        json.dumps(r, ensure_ascii=False, default=_json_default)
        for r in rows
    ).encode("utf-8")

    c1, c2 = st.columns(2)
    c1.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="results.csv",
        mime="text/csv",
        key="dl_csv_results",
    )
    c2.download_button(
        "Download JSONL",
        data=jsonl_bytes,
        file_name="results.jsonl",
        mime="application/json",
        key="dl_jsonl_results",
    )

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

    tabs = st.tabs(["Dashboard", "Live Logs", "Per-PDF", "Results", "Settings", "Debug"])
    with tabs[0]:
        tab_dashboard()
    with tabs[1]:
        tab_logs()
    with tabs[2]:
        tab_per_pdf()
    with tabs[3]:
        tab_results()
    with tabs[4]:
        tab_settings()
    with tabs[5]:
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
        if st.session_state.worker_done_event and st.session_state.worker_done_event.is_set():
            res = st.session_state.worker_result or {}
            if res.get("rows") is not None:
                st.session_state.results = res.get("rows") or []
            st.session_state.job["status"] = "error" if res.get("status") == "error" else "done"
            st.session_state.job["end_ts"] = time.time()
            if st.session_state.log_handler:
                _detach_logging(st.session_state.log_handler)
                st.session_state.log_handler = None
            st.rerun()
        else:
            st.toast("Job running… logs will update live.")
            st.rerun()

if __name__ == "__main__":
    main()
