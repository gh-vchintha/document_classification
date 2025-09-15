#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined OCR + Classification pipeline (single file) — local-friendly & thread-safe

- Per-page OCR using OpenAI Vision (gpt-4o) by default (Nova path kept as optional)
- Per-PDF classification (NO truncation) inside batch_process_pdfs
- Returns rows: **filename**, classification, job_run_time
- Robust logging with progress heartbeats
- Timeout hardening and adaptive image downscale for OCR
- No Streamlit imports (safe to call from any thread)

Environment knobs (defaults mirror your previous script):

  AWS_REGION (default us-east-1)
  NOVA_MODEL_ID (default us.amazon.nova-pro-v1:0)
  CLASSIFICATION_MODEL_ID (default us.amazon.nova-premier-v1:0)
  CLASSIFICATION_OPENAI_MODEL (default gpt-5)

  # Rendering & concurrency
  PDF_RENDER_DPI=220
  PDF_RENDER_FMT=jpeg            (jpeg|png)
  PDF_OCR_MAX_WORKERS=6          (per-PDF thread pool)
  PDF_OCR_MAX_OCR=4              (concurrent OCR calls per PDF)
  PDF_OCR_MAX_PDFS=3             (PDFs in parallel at job level)
  PDF_OCR_MAX_RETRIES=4          (OCR retries)
  CLS_MAX_RETRIES=3              (classification retries)

  # Progress logging
  OCR_PROGRESS_PAGES=5
  OCR_PROGRESS_SECS=20
  LOG_LEVEL=INFO

  # Bedrock timeouts
  BEDROCK_READ_TIMEOUT=180
  BEDROCK_CONNECT_TIMEOUT=10

  # OCR image sizing
  OCR_MAX_SIDE=2000
  OCR_MAX_SIDE_FALLBACK=1400
  OCR_JPEG_QUALITY=85
  OCR_JPEG_QUALITY_FALLBACK=75

  # Debug output directory
  DEBUG_DIR=./.app_cache/debug
"""

import os
import io
import gc
import json
import base64
import time
import random
import shutil
import logging
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.config import Config
import botocore.exceptions

from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from PIL import Image

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
import cv2
import numpy as np

# ---------------- OpenAI client (for Vision + classification) -------------------------
try:
    from openai import OpenAI  # SDK >= 1.x
    # Guardant PoC credential shim
    try:
        # Prefer your existing api_key wiring if available
        openai_client = OpenAI(api_key=os.environ["api_key"])
    except Exception:
        openai_client = OpenAI()  # falls back to env OPENAI_API_KEY
except Exception as _e:
    openai_client = None  # type: ignore

# ---------------- Guardant PoC prompt & helpers --------------------------------------
# Keep absolute import for your layout, but allow fallback to relative if needed
try:
    from promts import EXTRACTION_PROMPT, classify_v3_prompt  # type: ignore
except Exception:
    # fallback if running from inside poc/
    from promts import EXTRACTION_PROMPT, classify_v3_prompt  # type: ignore

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("ocr_only_v2")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --------------------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------------------
CLASSIFICATION_OPENAI_MODEL = os.environ.get("CLASSIFICATION_OPENAI_MODEL", "gpt-5")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
NOVA_MODEL_ID = os.environ.get("NOVA_MODEL_ID", "us.amazon.nova-pro-v1:0")
CLASSIFICATION_MODEL_ID = os.environ.get("CLASSIFICATION_MODEL_ID", "us.amazon.nova-premier-v1:0")

PDF_RENDER_DPI = int(os.environ.get("PDF_RENDER_DPI", "220"))
PDF_RENDER_FMT = os.environ.get("PDF_RENDER_FMT", "jpeg")  # jpeg|png

BEDROCK_READ_TIMEOUT = int(os.environ.get("BEDROCK_READ_TIMEOUT", "300"))
BEDROCK_CONNECT_TIMEOUT = int(os.environ.get("BEDROCK_CONNECT_TIMEOUT", "10"))

OCR_MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", "2000"))
OCR_MAX_SIDE_FALLBACK = int(os.environ.get("OCR_MAX_SIDE_FALLBACK", "1400"))
OCR_JPEG_QUALITY = int(os.environ.get("OCR_JPEG_QUALITY", "85"))
OCR_JPEG_QUALITY_FALLBACK = int(os.environ.get("OCR_JPEG_QUALITY_FALLBACK", "75"))

CLS_MAX_RETRIES = int(os.environ.get("CLS_MAX_RETRIES", "3"))

DEBUG_DIR = os.environ.get(
    "DEBUG_DIR",
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".app_cache", "debug"),
)
os.makedirs(DEBUG_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# Parquet schema (kept for compatibility; not required by UI)
# --------------------------------------------------------------------------------------
DOC_SCHEMA = pa.schema([
    pa.field("pdf_file", pa.string()),
    pa.field("results", pa.string()),
    pa.field("classification", pa.string()),
    pa.field("job_run_time", pa.timestamp("s")),
])


def write_results_parquet_s3(rows: List[Dict[str, Any]], bucket: str, key: str):
    """Optional S3 writer; not used by the Streamlit UI but kept for parity."""
    df = pd.DataFrame(rows, columns=["pdf_file", "results", "classification", "job_run_time"])
    df = df.astype({
        "pdf_file": "string",
        "results": "string",
        "classification": "string",
    })
    if "job_run_time" in df.columns:
        df["job_run_time"] = pd.to_datetime(df["job_run_time"], errors="coerce", utc=True)

    with open("output.json", "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({k: (row[k].isoformat() if k == "job_run_time" and pd.notna(row[k]) else row[k])
                                for k in df.columns}, ensure_ascii=False) + "\n")
    logger.info("Saved JSON lines to output.json (rows=%d)", len(rows))

    table = pa.Table.from_pandas(df, schema=DOC_SCHEMA, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)

    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.info("Saved Parquet to s3://%s/%s (rows=%d)", bucket, key, len(rows))


# --------------------------------------------------------------------------------------
# S3 helpers
# --------------------------------------------------------------------------------------

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    u = urlparse(uri)
    if u.scheme != "s3" or not u.netloc or not u.path:
        raise ValueError(f"Invalid S3 URI: {uri!r}. Expected: s3://<bucket>/<key>")
    return u.netloc, u.path.lstrip("/")


def get_bytes_from_s3(s3_uri: str) -> bytes:
    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


# --------------------------------------------------------------------------------------
# Bedrock client (optional)
# --------------------------------------------------------------------------------------

def _build_bedrock_client(region: str):
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(connect_timeout=BEDROCK_CONNECT_TIMEOUT, read_timeout=BEDROCK_READ_TIMEOUT),
    )


# --------------------------------------------------------------------------------------
# Extractor + Classifier
# --------------------------------------------------------------------------------------

class HandwritingExtractor:
    def __init__(self, aws_region: Optional[str] = None, model_id: Optional[str] = None):
        self.aws_region = aws_region or AWS_REGION
        self.model_id = model_id or NOVA_MODEL_ID
        self.classification_model_id = CLASSIFICATION_MODEL_ID
        self._local = threading.local()  # thread-local Bedrock client

        # Concurrency knobs
        self.max_workers = int(os.environ.get("PDF_OCR_MAX_WORKERS", "6"))
        self.max_ocr_workers = int(os.environ.get("PDF_OCR_MAX_OCR", "4"))
        self.max_retries = int(os.environ.get("PDF_OCR_MAX_RETRIES", "4"))
        self._ocr_sem = threading.Semaphore(self.max_ocr_workers)

    # ---------------- OCR image prep (adaptive) ----------------
    @staticmethod
    def _prep_for_ocr(image: Image.Image, max_side: int = OCR_MAX_SIDE, jpeg_quality: int = OCR_JPEG_QUALITY) -> Tuple[str, str]:
        im = image.convert("L")
        w, h = im.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / float(m)
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64, "jpeg"

    # ---------------- Optional: OpenCV preproc -----------------
    def preprocess_image(self, img_bytes: bytes) -> bytes:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        thresh = cv2.adaptiveThreshold(
            inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        _, buffer = cv2.imencode('.png', thresh)
        return buffer.tobytes()

    def prepare_image_for_vision(self, img_bytes: bytes, max_side: int = 1400, jpeg_quality: int = 85) -> bytes:
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        w, h = img.size
        scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
        return out.getvalue()

    # ---------------- OCR via OpenAI Vision --------------------
    def transcribe_handwritten(self, img_bytes: bytes, prompt: str) -> str:
        if openai_client is None:
            raise RuntimeError("OpenAI client not available; set OPENAI_API_KEY or wire poc.client.api_key")
        processed_img = self.prepare_image_for_vision(img_bytes)
        b64 = base64.b64encode(processed_img).decode("utf-8")

        system_msg = {
            "role": "system",
            "content": (
                "You are a vision-enabled assistant. "
                "Look carefully at the image(s). If a page is not human-readable, do not guess; return an empty string. "
                "Always return a valid JSON object only."
            ),
        }

        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                system_msg,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}:: Provide output in JSON only. No other text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                    ],
                },
            ],
        )
        all_res = resp.choices
        all_texts = []
        for choice in all_res:
            content = getattr(choice.message, "content", "") or ""
            all_texts.append(self._strip_code_fences(content))
        return "\n\n".join(all_texts).strip()

    # ---------------- Nova OCR path (kept for parity) ---------
    def extract_text_from_image(self, image: Image.Image, custom_prompt: Optional[str] = None) -> str:
        with self._ocr_sem:
            delay = 0.5
            max_side = OCR_MAX_SIDE
            quality = OCR_JPEG_QUALITY
            for attempt in range(1, self.max_retries + 1):
                if attempt > 1:
                    max_side = min(max_side, OCR_MAX_SIDE_FALLBACK)
                    quality = min(quality, OCR_JPEG_QUALITY_FALLBACK)
                b64, fmt = self._prep_for_ocr(image, max_side=max_side, jpeg_quality=quality)
                request_body = {
                    "schemaVersion": "messages-v1",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"image": {"format": fmt, "source": {"bytes": b64}}},
                            {"text": custom_prompt or EXTRACTION_PROMPT},
                        ],
                    }],
                    "inferenceConfig": {"temperature": 0.1, "topP": 0.0},
                }
                try:
                    br = getattr(self._local, "bedrock", None)
                    if br is None:
                        br = _build_bedrock_client(self.aws_region)
                        self._local.bedrock = br
                    resp = br.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(request_body),
                        accept="application/json",
                        contentType="application/json",
                    )
                    data = json.loads(resp["body"].read())
                    out_msg = data.get("output", {}).get("message", {})
                    for item in out_msg.get("content", []) or []:
                        if isinstance(item, dict) and "text" in item:
                            return (item["text"] or "").strip()
                    raise ValueError("Nova returned no text")
                except (botocore.exceptions.ReadTimeoutError,
                        botocore.exceptions.ConnectTimeoutError,
                        botocore.exceptions.EndpointConnectionError) as e:
                    if attempt < self.max_retries:
                        sleep_for = delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                        logger.warning(
                            "OCR timeout/conn on attempt %d (max_side=%s q=%s): %s; retrying in %.2fs…",
                            attempt, max_side, quality, e, sleep_for,
                        )
                        time.sleep(sleep_for)
                        continue
                    logger.error("OCR failed after %d attempts due to timeouts.", attempt)
                    raise
                except Exception as e:
                    if attempt < self.max_retries:
                        sleep_for = delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                        logger.warning("OCR attempt %d failed (%s); retrying in %.2fs…", attempt, e, sleep_for)
                        time.sleep(sleep_for)
                    else:
                        raise

    # ---------------- Render one page -------------------------
    def _render_single_page(self, pdf_bytes: bytes, page: int,
                            dpi: int = PDF_RENDER_DPI, fmt: str = PDF_RENDER_FMT,
                            grayscale: bool = True) -> Image.Image:
        tmpdir = tempfile.mkdtemp(dir="/tmp")
        try:
            paths = convert_from_bytes(
                pdf_bytes, dpi=dpi, fmt=fmt, grayscale=grayscale,
                output_folder=tmpdir, paths_only=True, thread_count=1,
                first_page=page, last_page=page,
            )
            if not paths:
                raise RuntimeError(f"Failed to render page {page}")
            p = paths[0]
            try:
                im = Image.open(p)
                return im.copy()
            finally:
                try:
                    os.remove(p)
                except OSError:
                    pass
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ---------------- Process one page ------------------------
    def _process_page_task(self, pdf_bytes: bytes, page_num: int, custom_prompt: Optional[str]) -> Tuple[int, str]:
        t0 = time.perf_counter()
        img = None
        try:
            img = self._render_single_page(pdf_bytes, page_num)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            prompt = custom_prompt or EXTRACTION_PROMPT
            text = self.transcribe_handwritten(img_bytes, prompt)
            text = self._strip_code_fences(text or "")
            return page_num, text
        finally:
            try:
                if img is not None:
                    img.close()
            except Exception:
                pass
            gc.collect()
            elapsed = time.perf_counter() - t0
            logger.info("Page %d done in %.2fs", page_num, elapsed)

    # ---------------- Process a PDF ---------------------------
    def process_pdf(self, pdf_bytes: bytes, custom_prompt: Optional[str] = None, max_pages: Optional[int] = None) -> Dict[int, str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        info = pdfinfo_from_bytes(pdf_bytes)
        total_pages = int(info.get("Pages", 0)) or 0
        if total_pages <= 0:
            raise ValueError("Could not determine page count (pdfinfo)")
        if max_pages is not None and max_pages > 0:
            num_pages = min(max_pages, total_pages)
        else:
            num_pages = total_pages

        workers = max(2, min(self.max_workers, num_pages))
        logger.info("OCR: pages=%d | workers=%d | ocr_concurrency=%d", num_pages, workers, self.max_ocr_workers)

        results: Dict[int, str] = {}

        heartbeat_every_pages = int(os.environ.get("OCR_PROGRESS_PAGES", "5"))
        heartbeat_every_secs = float(os.environ.get("OCR_PROGRESS_SECS", "20"))
        done = 0
        t_start = time.perf_counter()
        last_log = t_start

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._process_page_task, pdf_bytes, p, custom_prompt): p for p in range(1, num_pages + 1)}
            for fut in as_completed(futs):
                page, text = fut.result()
                results[page] = text
                done += 1
                now = time.perf_counter()
                if (done % heartbeat_every_pages == 0) or (now - last_log >= heartbeat_every_secs):
                    rate = done / max(1e-6, (now - t_start))
                    remaining = num_pages - done
                    eta_s = remaining / max(1e-6, rate)
                    logger.info("Progress: %d/%d pages (%.1f%%), ~%.2f p/s, ETA ~%.0fs",
                                done, num_pages, 100.0 * done / num_pages, rate, eta_s)
                    last_log = now

        total_s = time.perf_counter() - t_start
        logger.info("OCR complete: %d/%d pages in %.2fs (%.2f p/s)", done, num_pages, total_s, done / max(1e-6, total_s))
        return {k: results[k] for k in sorted(results)}

    # ---------------- Classification (NO truncation) ----------
    @staticmethod
    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("\n", 1)[0]
        s = s.strip()
        if s.lower().startswith("json"):
            s = s.split("\n", 1)[-1].strip()
        return s

    def _build_pages_array_no_trunc(self, extracted_per_page: Dict[int, str], kv_per_page: Optional[Dict[int, Any]] = None) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        kv_per_page = kv_per_page or {}
        for p in sorted(extracted_per_page):
            pages.append({"pageno": int(p), "rawtext": extracted_per_page[p] or "", "data": kv_per_page.get(p, {}) or {}})
        return pages

    def _classify_pages_no_trunc(self, pages: List[Dict[str, Any]], *, temperature: float = 0.1) -> str:
        if openai_client is None:
            raise RuntimeError("OpenAI client not available; set OPENAI_API_KEY or wire poc.client.api_key")
        pages_text = json.dumps(pages, ensure_ascii=False, separators=(",", ":"))
        prompt = classify_v3_prompt(pages_text).strip()
        delay = 0.5
        for attempt in range(1, CLS_MAX_RETRIES + 1):
            try:
                resp = openai_client.responses.create(
                    model=CLASSIFICATION_OPENAI_MODEL,
                    input=[
                        {"role": "system", "content": "Return only a single JSON array. No prose, no Markdown, no comments."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.output_text
            except (botocore.exceptions.ReadTimeoutError,
                    botocore.exceptions.ConnectTimeoutError,
                    botocore.exceptions.EndpointConnectionError) as e:
                if attempt < CLS_MAX_RETRIES:
                    sleep_for = delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                    logger.warning("Classification timeout on attempt %d: %s; retrying in %.2fs…", attempt, e, sleep_for)
                    time.sleep(sleep_for)
                    continue
                logger.error("Classification failed after %d attempts due to timeouts.", attempt)
                raise
            except Exception as e:
                if attempt < CLS_MAX_RETRIES:
                    sleep_for = delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                    logger.warning("Classification attempt %d failed (%s); retrying in %.2fs…", attempt, e, sleep_for)
                    time.sleep(sleep_for)
                else:
                    raise


    def prepare_payload(self, classification, filename: str, job_run_time: str):
        """
        Accepts classification as str | list | dict.
        Returns a list[dict] of flattened rows, one per document, while preserving the
        original doc JSON under 'classification' for the UI.
        """
        # 1) Parse if it's a JSON string

        if isinstance(classification, str):
            try:
                classification = json.loads(classification)
            except Exception:
                # Can't parse; return a single row with raw content
                return [{
                    "filename": filename,
                    "classification": classification,   # keep raw text
                    "bench_mark": "actuals",
                    "job_run_time": job_run_time,
                }]

        # 2) Normalize to a list of documents
        if isinstance(classification, list):
            docs = classification
        elif isinstance(classification, dict):
            docs = classification.get("documents", [])
            if not isinstance(docs, list):
                docs = []
        else:
            docs = []

        rows = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            # doc structure assumed: { "t": type, "pt": page_range, "d": { ...fields... } }
            d = (doc.get("d") or {}) if isinstance(doc.get("d"), dict) else {}

            row = {
                "filename": filename,
                "doc_type": doc.get("t", ""),
                "page_range": doc.get("pt", ""),
                "job_run_time": job_run_time,
            }

            # Optional field mapping
            for out_key, src_key in [
                ("patient_name",   "Patient Name"),
                ("subscriber_id",  "Subscriber ID"),
                ("date_of_service","Date of Service"),
                ("mr_id",          "MR ID"),
                ("accn_id",        "Accession ID"),
                ("claim_id",       "Claim ID"),
            ]:
                val = d.get(src_key)
                if val not in (None, "", []):
                    row[out_key] = str(val)

            rows.append(row)

        # If model returned no docs, still emit a row so UI has something to show
        if not rows:
            rows.append({
                "filename": filename,
                "classification": json.dumps(classification, ensure_ascii=False),
                "job_run_time": job_run_time,
            })
        return rows



    # ---------------- Batch: OCR + classify -------------------
    def batch_process_pdfs(self, pdf_list: List[str], is_classify: bool = True, max_pages_to_process: Optional[int] = None) -> List[Dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor

        def _load_pdf(path: str) -> bytes:
            return get_bytes_from_s3(path) if path.lower().startswith("s3://") else open(path, "rb").read()

        def _one_pdf(pdf_path: str) -> Dict[str, Any]:
            filename_ = os.path.basename(pdf_path)
            try:
                pdf_bytes = _load_pdf(pdf_path)
                logger.info("Processing %s", pdf_path)

                # 1) OCR for THIS PDF
                custom_prompt  = EXTRACTION_PROMPT
                extracted = self.process_pdf(pdf_bytes, custom_prompt, max_pages=max_pages_to_process)

                # 2) Build pages (NO truncation) + classify
                pages = self._build_pages_array_no_trunc(extracted)
                logger.info("Classify start: %s (pages=%d)", pdf_path, len(extracted))
                try:
                    with open(os.path.join(DEBUG_DIR, f"{filename_}.txt"), "w", encoding="utf-8") as file:
                        file.write(json.dumps(pages, ensure_ascii=False))
                except Exception as _e:
                    logger.debug("Could not write debug pages for %s: %s", filename_, _e)

                if is_classify:
                    classification_json = self._classify_pages_no_trunc(pages, temperature=0.1)
                else:
                    classification_json = "{}"
                logger.info("Classify done: %s", pdf_path)

                job_run_time = pd.Timestamp.utcnow().tz_convert("UTC")
                data = self.prepare_payload(classification_json,filename_,str(job_run_time))
                return data

            except Exception as e:
                logger.error("Error processing %s: %s", pdf_path, e)
                return {
                    "filename": str(filename_),
                    "classification": f"Error: {e}",                    "job_run_time": pd.NaT,
                }

        max_pdf_workers = int(os.environ.get("PDF_OCR_MAX_PDFS", "3"))
        logger.info("Starting batch of %d PDFs (parallel=%d)", len(pdf_list), max_pdf_workers)
        rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_pdf_workers) as ex:
            for r in ex.map(_one_pdf, pdf_list):
                if isinstance(r, list):
                    rows.extend(r)
                    fn = r[0]["filename"] if r else "unknown"
                else:
                    rows.append(r)
                    fn = r.get("filename", "unknown")
                logger.info("Completed PDF: %s", fn)
        logger.info("Batch finished: %d PDFs", len(rows))
        return rows


# ----------------------------- Local runner -------------------------------------------

def local_run():
    extractor = HandwritingExtractor(aws_region=AWS_REGION)
    try:
        input_folder = os.environ.get("INPUT_DIR", os.path.join(os.path.abspath(os.path.dirname(__file__)), "inputs"))
        files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
        if not files:
            logger.warning("No PDFs found in %s", input_folder)
            return
        results_rows = extractor.batch_process_pdfs(files, is_classify=True, max_pages_to_process=100)
        out_dir = os.environ.get("OUTPUT_DIR", os.path.join(os.path.abspath(os.path.dirname(__file__)), "outputs"))
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(results_rows).to_csv(os.path.join(out_dir, "all_transcriptions.csv"), index=False)
        logger.info("Wrote outputs/all_transcriptions.csv (%d rows)", len(results_rows))
    except Exception as e:
        logger.error("Local error: %s", e)
        raise


if __name__ == "__main__":
    local_run()
