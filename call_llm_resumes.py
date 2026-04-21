"""
Send prompts_for_sarvam.jsonl to AICredits API and collect generated resumes.

Output: generated_resumes.jsonl
Each record: { id, jd_id, preset, contact, resume_text, truncated }

- Rate-limited to 55 RPM (AICredits allows 60 RPM default)
- Resumes from last completed line on restart (append mode)
- Retries up to 3x with backoff on 429 / 5xx / network errors
- Free-text response — no JSON parsing needed

Model: meta-llama/llama-3.1-8b-instruct (cheapest capable model on AICredits)
Cost: ~₹10 per full 2940-resume run

Usage:
    python call_llm_resumes.py
    python call_llm_resumes.py --input prompts_for_sarvam.jsonl --out generated_resumes.jsonl
    python call_llm_resumes.py --model google/gemma-2-9b-it   # switch model if quality is bad
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
# CONFIG
# ================================
DEFAULT_MODEL    = "meta-llama/llama-3.1-8b-instruct"
BASE_URL         = "https://api.aicredits.in/v1"
API_URL          = f"{BASE_URL}/chat/completions"
MAX_RETRIES      = 3
TARGET_RPM       = 55
REQUEST_INTERVAL = 60.0 / TARGET_RPM   # ~1.09 s between requests
REQUEST_TIMEOUT  = (15, 120)           # (connect, read) seconds


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


# ================================
# HELPERS
# ================================
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
    """Rough check: resume ended at a natural stopping point."""
    ending = text.rstrip()
    return bool(ending) and any(
        ending.endswith(s) for s in ("LPA", "Days", "days", "N/A", ".", ")", "]")
    )


def call_llm(session: requests.Session, rate_limiter: RateLimiter,
             prompt: str, model: str) -> Tuple[str, bool]:
    """
    Returns (resume_text, is_complete).
    is_complete=False means output was truncated (hit token limit mid-resume).
    """
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


def count_output_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def append_line(fh, record: dict) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


# ================================
# MAIN
# ================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate resumes via AICredits API.")
    parser.add_argument("--input", default="prompts_full_v2.jsonl")
    parser.add_argument("--out",   default="llm_resume_output_full_v2.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model ID on AICredits (default: meta-llama/llama-3.1-8b-instruct)")
    return parser.parse_args()


def main():
    args      = parse_args()
    start_idx = count_output_lines(args.out)

    print(f"Model : {args.model}")
    print(f"Input : {args.input}")
    print(f"Output: {args.out}")
    if start_idx:
        print(f"Resuming from record #{start_idx + 1} (found {start_idx} already in output)")

    session      = build_session(AICREDITS_API_KEY)
    rate_limiter = RateLimiter(REQUEST_INTERVAL)
    success      = 0
    failed       = 0

    try:
        with open(args.input, "r", encoding="utf-8") as infile, \
             open(args.out,   "a", encoding="utf-8") as outfile:

            for idx, raw_line in enumerate(infile):
                if idx < start_idx:
                    continue

                record_num = idx + 1
                line = raw_line.strip()
                if not line:
                    continue

                rec    = json.loads(line)
                rec_id = rec.get("id", f"record_{record_num}")

                print(f"[{record_num:3d}] {rec_id} ...", end=" ", flush=True)

                last_error  = None
                resume_text = None
                is_complete = False

                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        resume_text, is_complete = call_llm(
                            session, rate_limiter, rec["prompt"], args.model
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
                    append_line(outfile, {
                        "id":          rec_id,
                        "jd_id":       rec.get("jd_id"),
                        "preset":      rec.get("preset"),
                        "contact":     rec.get("contact", {}),
                        "resume_text": resume_text,
                        "truncated":   not is_complete,
                    })
                    success += 1
                    status = "ok" if is_complete else "ok (truncated)"
                    print(status)
                else:
                    # Tombstone so resume-from-line works correctly
                    append_line(outfile, {
                        "id":          rec_id,
                        "jd_id":       rec.get("jd_id"),
                        "preset":      rec.get("preset"),
                        "contact":     rec.get("contact", {}),
                        "resume_text": None,
                        "truncated":   True,
                        "_error":      last_error or "max retries exceeded",
                    })
                    failed += 1
                    print(f"FAILED — {last_error}")

    finally:
        session.close()

    print(f"\nDone. {success} generated, {failed} failed.")


if __name__ == "__main__":
    main()
