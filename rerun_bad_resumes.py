"""
Re-run specific resume IDs that have quality issues (duplicate company entries
or XYZ/ABC placeholder hallucinations).

Reads ids_to_rerun.json, looks up prompts from prompts_full_v2.jsonl,
re-calls the LLM with a tighter system prompt, then patches
llm_resume_output_full_v2_final.jsonl in-place.

Usage:
    python rerun_bad_resumes.py
    python rerun_bad_resumes.py --ids ids_to_rerun.json \
                                --prompts prompts_full_v2.jsonl \
                                --final llm_resume_output_full_v2_final.jsonl
"""

import argparse
import json
import os
import re
import time
from collections import Counter
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
MAX_RETRIES      = 4
TARGET_RPM       = 55
REQUEST_INTERVAL = 60.0 / TARGET_RPM
REQUEST_TIMEOUT  = (15, 120)

# Tighter system prompt addressing both issues
SYSTEM_PROMPT = (
    "You are a resume writer. "
    "Output ONLY the final resume text. "
    "Do not explain or add any commentary. "
    "Start directly with the candidate's name. "
    "CRITICAL: Use the EXACT company names provided in the candidate's experience — "
    "do NOT replace them with 'XYZ', 'XYZ Corporation', 'ABC Company', 'ABC Corp', or any placeholder. "
    "CRITICAL: List each company EXACTLY ONCE in the Experience section. "
    "If the candidate has only one job, write only one Experience entry — do NOT duplicate it. "
    "Do NOT invent extra experience entries that are not in the candidate data. "
    "Do NOT start the summary with 'Highly motivated', 'Results-driven', 'Dynamic', "
    "'Seasoned professional', or similar generic openers — write a specific 2-line summary "
    "tied to the role and tech stack."
)


# ================================
# QUALITY CHECKS
# ================================
def has_xyz(text: str) -> bool:
    return bool(re.search(r'\bXYZ\b|ABC Company|ABC Corp', text, re.I))


def has_dupe_company(text: str) -> bool:
    companies_seen = Counter()
    for line in text.split('\n'):
        m = re.search(r'at (.+?) \(', line)
        if m:
            companies_seen[m.group(1).strip()] += 1
    return any(n > 1 for n in companies_seen.values())


def is_quality_ok(text: str) -> bool:
    return not has_xyz(text) and not has_dupe_company(text)


# ================================
# API HELPERS
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens":  800,
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
# MAIN
# ================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids",     default="ids_to_rerun.json")
    parser.add_argument("--prompts", default="prompts_full_v2.jsonl")
    parser.add_argument("--final",   default="llm_resume_output_full_v2_final.jsonl")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--max-quality-attempts", type=int, default=3,
                        help="Extra LLM attempts if quality check still fails")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.ids) as f:
        target_ids = set(json.load(f))
    print(f"IDs to fix: {len(target_ids)}")

    # Load prompts index
    prompts_index = {}
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            prompts_index[r["id"]] = r
    print(f"Prompts loaded: {len(prompts_index)}")

    # Load current final file
    all_records = {}
    ordered_ids = []
    with open(args.final) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            all_records[r["id"]] = r
            ordered_ids.append(r["id"])
    print(f"Existing records: {len(all_records)}")

    session      = build_session(AICREDITS_API_KEY)
    rate_limiter = RateLimiter(REQUEST_INTERVAL)

    fixed = 0; still_bad = 0; api_failed = 0
    target_list = sorted(target_ids)

    try:
        for i, rid in enumerate(target_list, start=1):
            if rid not in prompts_index:
                print(f"[{i:3d}/{len(target_list)}] {rid} — no prompt found, skipping")
                continue

            orig = all_records.get(rid, {})
            prompt = prompts_index[rid]["prompt"]
            print(f"[{i:3d}/{len(target_list)}] {rid} ...", end=" ", flush=True)

            best_text = None
            best_complete = False
            last_error = None

            # Try up to MAX_RETRIES API attempts; within each, check quality
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resume_text, is_complete = call_llm(session, rate_limiter, prompt, args.model)
                except APICallError as exc:
                    last_error = clean_ws(str(exc))
                    backoff = get_backoff(attempt, exc)
                    if exc.retry_after:
                        rate_limiter.defer(exc.retry_after)
                    print(f"api-err {attempt} ({last_error[:50]}) ...", end=" ", flush=True)
                    if attempt < MAX_RETRIES:
                        time.sleep(backoff)
                    continue

                if best_text is None:
                    best_text = resume_text
                    best_complete = is_complete

                if is_quality_ok(resume_text):
                    best_text = resume_text
                    best_complete = is_complete
                    break  # quality passed — done
                else:
                    # Keep track of best attempt even if quality failed
                    best_text = resume_text
                    best_complete = is_complete
                    if attempt < MAX_RETRIES:
                        issues = []
                        if has_xyz(resume_text): issues.append("xyz")
                        if has_dupe_company(resume_text): issues.append("dupe")
                        print(f"quality-fail({','.join(issues)}) retry{attempt} ...", end=" ", flush=True)

            if best_text is None:
                api_failed += 1
                print(f"API-FAILED — {last_error}")
                continue

            # Update record
            new_rec = {
                "id":          rid,
                "jd_id":       orig.get("jd_id"),
                "preset":      orig.get("preset"),
                "contact":     orig.get("contact", {}),
                "resume_text": best_text,
                "truncated":   not best_complete,
            }
            all_records[rid] = new_rec

            if is_quality_ok(best_text):
                fixed += 1
                status = "fixed" if best_complete else "fixed(truncated)"
            else:
                still_bad += 1
                issues = []
                if has_xyz(best_text): issues.append("xyz")
                if has_dupe_company(best_text): issues.append("dupe")
                status = f"still-bad({','.join(issues)})"
            print(status)

    finally:
        session.close()

    # Write patched final file in original order
    with open(args.final, "w", encoding="utf-8") as f:
        for rid in ordered_ids:
            rec = all_records.get(rid)
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. fixed={fixed}  still_bad={still_bad}  api_failed={api_failed}")
    print(f"Patched file written → {args.final}")


if __name__ == "__main__":
    main()
