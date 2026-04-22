"""
Retry failed tombstone records from a completed llm_resume_output_*.jsonl run.

Usage:
    python retry_failed_resumes.py
    python retry_failed_resumes.py --source llm_resume_output_full_v2.jsonl \
                                   --input  prompts_full_v2.jsonl \
                                   --out    retried_resumes.jsonl \
                                   --merged llm_resume_output_full_v2_final.jsonl

Steps:
  1. Scan --source for tombstone records (those with _error field).
  2. Look up their prompts from --input by matching 'id'.
  3. Re-call the LLM for each failed record.
  4. Write successes + surviving failures to --out.
  5. Merge: copy --source (skipping tombstones that now succeeded) + --out → --merged.
     The merged file is in original line order (id order from --input).
"""

import argparse
import json
import os
import re
import time
from typing import Optional, Tuple

import requests


# ================================
# LOAD ENV
# ================================
def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key   = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()
AICREDITS_API_KEY = os.getenv("LAMMA_API_KEY")
if not AICREDITS_API_KEY:
    raise ValueError("LAMMA_API_KEY not found in environment or .env file")


# ================================
# CONFIG  (shared with call_llm_resumes)
# ================================
DEFAULT_MODEL    = "meta-llama/llama-3.1-8b-instruct"
BASE_URL         = "https://api.aicredits.in/v1"
API_URL          = f"{BASE_URL}/chat/completions"
MAX_RETRIES      = 4          # one extra attempt for retries
TARGET_RPM       = 55
REQUEST_INTERVAL = 60.0 / TARGET_RPM
REQUEST_TIMEOUT  = (15, 120)


# ================================
# RATE LIMITER
# ================================
class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self.next_allowed = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self.next_allowed:
            time.sleep(self.next_allowed - now)
        self.next_allowed = max(now, self.next_allowed) + self.min_interval

    def defer(self, seconds: float) -> None:
        if seconds > 0:
            self.next_allowed = max(self.next_allowed, time.monotonic() + seconds)


class APICallError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None,
                 retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def clean_ws(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def extract_retry_after(headers) -> Optional[float]:
    val = headers.get("Retry-After") or headers.get("x-ratelimit-reset")
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except (TypeError, ValueError):
        return None


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return clean_ws(response.text) or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return clean_ws(val)
            if isinstance(val, dict):
                for nk in ("message", "detail"):
                    nv = val.get(nk)
                    if isinstance(nv, str) and nv.strip():
                        return clean_ws(nv)
    return clean_ws(response.text) or f"HTTP {response.status_code}"


def get_backoff(attempt: int, error: Exception) -> float:
    if isinstance(error, APICallError):
        if error.retry_after:
            return max(1.0, error.retry_after)
        if error.status_code == 429:
            return 5.0 * attempt
        if error.status_code and error.status_code >= 500:
            return min(12.0, 2.0 * attempt)
    return min(8.0, float(attempt))


def build_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "User-Agent":    "resume-generator/1.0",
    })
    return session


def looks_complete(text: str) -> bool:
    ending = text.rstrip()
    return bool(ending) and any(
        ending.endswith(s) for s in ("LPA", "Days", "days", "N/A", ".", ")", "]")
    )


def call_llm(session: requests.Session, rate_limiter: RateLimiter,
             prompt: str, model: str) -> Tuple[str, bool]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a resume writer. "
                    "Output ONLY the final resume text. "
                    "Do not explain or add any commentary. "
                    "Start directly with the candidate's name. "
                    "Use the exact company names provided — do NOT replace them with 'XYZ Corporation', 'ABC Company', or any placeholder. "
                    "Do NOT start the summary with 'Highly motivated', 'Results-driven', 'Dynamic', or similar generic openers — write a specific 2-line summary tied to the role and tech stack."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 800,
    }

    rate_limiter.wait()

    try:
        response = session.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise APICallError(f"Network error: {clean_ws(exc)}") from exc

    if response.status_code != 200:
        msg = extract_error_message(response)
        raise APICallError(msg, status_code=response.status_code,
                           retry_after=extract_retry_after(response.headers))

    try:
        data = response.json()
    except ValueError as exc:
        raise APICallError(f"Non-JSON response: {clean_ws(response.text[:300])}") from exc

    try:
        msg     = data["choices"][0]["message"]
        content = msg.get("content") or ""
        finish  = data["choices"][0].get("finish_reason", "")
    except (KeyError, IndexError, TypeError) as exc:
        raise APICallError(f"Unexpected response shape: {clean_ws(str(data)[:300])}") from exc

    resume = content.strip()
    if not resume:
        raise APICallError("Empty model response")

    is_complete = finish == "stop" and looks_complete(resume)
    return resume, is_complete


# ================================
# MERGE HELPERS
# ================================
def load_prompts_index(path: str) -> dict:
    """Return {id: record} from the prompts JSONL."""
    index = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            index[rec["id"]] = rec
    return index


def load_source_records(path: str) -> dict:
    """Return {id: record} from the existing output JSONL."""
    records = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


# ================================
# MAIN
# ================================
def parse_args():
    parser = argparse.ArgumentParser(description="Retry failed tombstone records.")
    parser.add_argument("--source", default="llm_resume_output_full_v2.jsonl",
                        help="Completed output file containing tombstone records.")
    parser.add_argument("--input",  default="prompts_full_v2.jsonl",
                        help="Original prompts file (to look up prompts for failures).")
    parser.add_argument("--out",    default="retried_resumes.jsonl",
                        help="Retry results (successes + remaining failures).")
    parser.add_argument("--merged", default="llm_resume_output_full_v2_final.jsonl",
                        help="Merged final file in original id order.")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--skip-merge", action="store_true",
                        help="Only retry; do not produce merged file.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading source: {args.source}")
    source_records = load_source_records(args.source)

    # Find tombstones
    failed_ids = [rid for rid, rec in source_records.items() if rec.get("_error")]
    print(f"Found {len(failed_ids)} failed records to retry.")

    if not failed_ids:
        print("Nothing to retry.")
    else:
        print(f"Loading prompts index: {args.input}")
        prompts_index = load_prompts_index(args.input)

        missing = [fid for fid in failed_ids if fid not in prompts_index]
        if missing:
            print(f"WARNING: {len(missing)} failed IDs not found in prompts file: {missing[:5]}")

        session      = build_session(AICREDITS_API_KEY)
        rate_limiter = RateLimiter(REQUEST_INTERVAL)
        retry_success = 0
        retry_failed  = 0

        try:
            with open(args.out, "w", encoding="utf-8") as outfile:
                for i, rid in enumerate(failed_ids, start=1):
                    if rid not in prompts_index:
                        continue

                    prompt_rec = prompts_index[rid]
                    orig_rec   = source_records[rid]
                    print(f"[{i:3d}/{len(failed_ids)}] {rid} ...", end=" ", flush=True)

                    last_error  = None
                    resume_text = None
                    is_complete = False

                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            resume_text, is_complete = call_llm(
                                session, rate_limiter, prompt_rec["prompt"], args.model
                            )
                            break
                        except APICallError as exc:
                            last_error = clean_ws(str(exc))
                            backoff    = get_backoff(attempt, exc)
                            if exc.retry_after:
                                rate_limiter.defer(exc.retry_after)
                            print(f"retry {attempt}/{MAX_RETRIES} ({last_error[:60]}) ...", end=" ", flush=True)
                            if attempt < MAX_RETRIES:
                                time.sleep(backoff)

                    if resume_text:
                        result = {
                            "id":          rid,
                            "jd_id":       orig_rec.get("jd_id"),
                            "preset":      orig_rec.get("preset"),
                            "contact":     orig_rec.get("contact", {}),
                            "resume_text": resume_text,
                            "truncated":   not is_complete,
                        }
                        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                        outfile.flush()
                        # Update in-memory source so merge step sees the fix
                        source_records[rid] = result
                        retry_success += 1
                        status = "ok" if is_complete else "ok (truncated)"
                        print(status)
                    else:
                        result = {
                            "id":          rid,
                            "jd_id":       orig_rec.get("jd_id"),
                            "preset":      orig_rec.get("preset"),
                            "contact":     orig_rec.get("contact", {}),
                            "resume_text": None,
                            "truncated":   True,
                            "_error":      last_error or "max retries exceeded",
                        }
                        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                        outfile.flush()
                        source_records[rid] = result
                        retry_failed += 1
                        print(f"FAILED — {last_error}")

        finally:
            session.close()

        print(f"\nRetry pass done. {retry_success} recovered, {retry_failed} still failed.")

    # ---- MERGE STEP ----
    if args.skip_merge:
        return

    print(f"\nBuilding merged output: {args.merged}")
    # Read original prompt order to preserve ordering
    ordered_ids = []
    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ordered_ids.append(json.loads(line)["id"])

    written = 0
    with open(args.merged, "w", encoding="utf-8") as mf:
        for rid in ordered_ids:
            rec = source_records.get(rid)
            if rec:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    total_success = sum(1 for rec in source_records.values() if not rec.get("_error"))
    total_failed  = sum(1 for rec in source_records.values() if rec.get("_error"))
    print(f"Merged {written} records → {args.merged}")
    print(f"Final: {total_success} good  |  {total_failed} still failed")


if __name__ == "__main__":
    main()
