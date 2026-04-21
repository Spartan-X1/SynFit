import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from skill_normalizer import normalize_skill_list


# ================================
# LOAD ENV
# ================================
def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')

            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY not found")


# ================================
# CONFIG
# ================================
INPUT_FILE = "fresh_naukri_jds.jsonl"
OUTPUT_FILE = "structured_jds.jsonl"
MODEL_NAME = "sarvam-m"
API_URL = "https://api.sarvam.ai/v1/chat/completions"

MAX_RETRIES = 3
TARGET_RPM = 56
REQUEST_INTERVAL_SECONDS = 60.0 / TARGET_RPM
REQUEST_TIMEOUT = (15, 120)

# Conservative token budgeting to avoid context overflow.
MODEL_CONTEXT_WINDOW_TOKENS = 8192
PROMPT_OVERHEAD_TOKENS = 650
OUTPUT_TOKEN_RESERVE = 350
INITIAL_JD_TOKEN_BUDGET = max(
    1200,
    MODEL_CONTEXT_WINDOW_TOKENS - PROMPT_OVERHEAD_TOKENS - OUTPUT_TOKEN_RESERVE,
)
MIN_JD_TOKEN_BUDGET = 1200
RETRY_TRIM_FACTORS = (1.0, 0.75, 0.55)

DEFAULT_OUTPUT = {
    "job_title": "",
    "min_experience_years": 0,
    "core_technical_skills": [],
    "domain": "Not Specified",
    "is_management_role": False,
    "education_requirement": "Not Specified",
    "max_notice_period_days": 90,
    "executive_summary": "",
}

REQUIRED_OUTPUT_KEYS = tuple(DEFAULT_OUTPUT.keys())


# ================================
# HELPERS
# ================================
class APICallError(Exception):
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
        context_overflow: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.context_overflow = context_overflow


class RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.next_allowed_at = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self.next_allowed_at:
            time.sleep(self.next_allowed_at - now)
        self.next_allowed_at = max(now, self.next_allowed_at) + self.min_interval_seconds

    def defer(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self.next_allowed_at = max(self.next_allowed_at, time.monotonic() + seconds)


def clean_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def trim_text_to_budget(text: str, token_budget: int) -> str:
    cleaned = clean_whitespace(text)
    if not cleaned:
        return ""

    max_chars = token_budget * 4
    if len(cleaned) <= max_chars:
        return cleaned

    head_chars = int(max_chars * 0.7)
    tail_chars = max_chars - head_chars - len(" [TRUNCATED] ")

    if tail_chars < 200:
        tail_chars = min(200, max_chars // 4)
        head_chars = max_chars - tail_chars - len(" [TRUNCATED] ")

    head = cleaned[:head_chars].rsplit(" ", 1)[0].strip()
    tail = cleaned[-tail_chars:].split(" ", 1)[-1].strip()

    if not head:
        return cleaned[:max_chars].strip()
    if not tail:
        return head

    return f"{head} [TRUNCATED] {tail}"


def build_prompt(job_title: str, raw_text: str, token_budget: int) -> str:
    trimmed_text = trim_text_to_budget(raw_text, token_budget)

    return f"""
You are an information extraction system.

Return exactly one valid JSON object and nothing else.

Schema:
{{
  "job_title": string,
  "min_experience_years": integer,
  "core_technical_skills": list,
  "domain": string,
  "is_management_role": boolean,
  "education_requirement": string,
  "max_notice_period_days": integer,
  "executive_summary": string
}}

Rules:
- Extract only explicitly stated information.
- Do not hallucinate missing facts.
- min_experience_years: lowest required number, else 0.
- max_notice_period_days: Immediate = 0, otherwise exact number, else 90.
- education_requirement: preserve required vs preferred, else "Not Specified".
- core_technical_skills: include only specific tools, languages, frameworks, databases, cloud platforms, or technologies.
- Exclude generic terms like AI, ML, analytics, programming, frameworks, leadership, communication.
- executive_summary must be concise, factual, and at most 2 sentences.
- No markdown, no code fences, no explanations.

Job Title:
{job_title}

Job Description:
{trimmed_text}
""".strip()


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {SARVAM_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "jd-structured-extractor/1.0",
        }
    )
    return session


def extract_retry_after(headers: requests.structures.CaseInsensitiveDict) -> Optional[float]:
    retry_after = headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        return None


def is_context_overflow_error(message: str) -> bool:
    lowered = message.lower()
    markers = (
        "context length",
        "maximum context",
        "maximum token",
        "too many tokens",
        "input too long",
        "input too large",
        "context window",
        "prompt is too long",
    )
    return any(marker in lowered for marker in markers)


def extract_api_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return clean_whitespace(response.text) or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return clean_whitespace(value)
            if isinstance(value, dict):
                for nested_key in ("message", "detail"):
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, str) and nested_value.strip():
                        return clean_whitespace(nested_value)

    return clean_whitespace(response.text) or f"HTTP {response.status_code}"


def call_sarvam(session: requests.Session, rate_limiter: RateLimiter, prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    rate_limiter.wait()

    try:
        response = session.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise APICallError(f"Network/API request failed: {clean_whitespace(exc)}") from exc

    if response.status_code != 200:
        message = extract_api_error_message(response)
        raise APICallError(
            message=message,
            status_code=response.status_code,
            retry_after=extract_retry_after(response.headers),
            context_overflow=is_context_overflow_error(message),
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise APICallError(f"API returned non-JSON response: {clean_whitespace(response.text[:500])}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise APICallError(f"Unexpected API response shape: {clean_whitespace(json.dumps(data)[:500])}") from exc

    content = clean_whitespace(content)
    if not content:
        raise APICallError("API returned empty model content")

    return content


# ================================
# JSON HANDLING
# ================================
def clean_model_response(text: str) -> str:
    cleaned = text.replace("\ufeff", "").strip()
    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```json", "```", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def coerce_json_candidate(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        best_dict = None
        best_score = -1
        for item in value:
            if isinstance(item, dict):
                score = score_candidate(item)
                if score > best_score:
                    best_dict = item
                    best_score = score
        return best_dict

    return None


def score_candidate(candidate: Dict[str, Any]) -> int:
    score = 0
    for key in REQUIRED_OUTPUT_KEYS:
        if key in candidate:
            score += 2
    score += min(len(candidate), 8)
    return score


def iter_json_substrings(text: str) -> List[str]:
    decoder = json.JSONDecoder()
    seen = set()
    candidates: List[str] = []

    stripped = text.strip()
    if stripped:
        seen.add(stripped)
        candidates.append(stripped)

    for block in re.findall(r"```(.*?)```", text, flags=re.DOTALL):
        candidate = block.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except ValueError:
            continue

        candidate = text[index : index + end].strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    return candidates


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = clean_model_response(text)
    best_candidate = None
    best_score = -1

    for candidate_text in iter_json_substrings(cleaned):
        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue

        candidate = coerce_json_candidate(parsed)
        if not candidate:
            continue

        score = score_candidate(candidate)
        if score > best_score:
            best_candidate = candidate
            best_score = score

    return best_candidate


def parse_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = clean_whitespace(value).lower()
    if not text:
        return default

    if "immediate" in text:
        return 0

    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))

    return default


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    text = clean_whitespace(value).lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return default


def normalize_skills(value: Any) -> List[str]:
    return normalize_skill_list(value)


def normalize_executive_summary(value: Any) -> str:
    summary = clean_whitespace(value)
    if not summary:
        return ""

    if len(summary) > 700:
        summary = summary[:700].rsplit(" ", 1)[0].strip()

    return summary


def normalize_extracted_record(parsed: Dict[str, Any], raw_jd: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(DEFAULT_OUTPUT)

    source_job_title = clean_whitespace(raw_jd.get("job_title", ""))
    source_url = clean_whitespace(raw_jd.get("source_url", "")) or "Unknown"

    record["job_title"] = clean_whitespace(parsed.get("job_title")) or source_job_title
    record["min_experience_years"] = max(0, parse_int(parsed.get("min_experience_years"), 0))
    record["core_technical_skills"] = normalize_skills(parsed.get("core_technical_skills"))
    record["domain"] = clean_whitespace(parsed.get("domain")) or "Not Specified"
    record["is_management_role"] = parse_bool(parsed.get("is_management_role"), False)
    record["education_requirement"] = (
        clean_whitespace(parsed.get("education_requirement")) or "Not Specified"
    )
    record["max_notice_period_days"] = max(0, parse_int(parsed.get("max_notice_period_days"), 90))
    record["executive_summary"] = normalize_executive_summary(parsed.get("executive_summary"))
    record["source_url"] = source_url

    return record


def build_skip_record(raw_jd: Optional[Dict[str, Any]], line_number: int, reason: str) -> Dict[str, Any]:
    source = raw_jd if isinstance(raw_jd, dict) else {}
    record = normalize_extracted_record({}, source)
    record["_status"] = "skipped"
    record["_line_number"] = line_number
    record["_error"] = clean_whitespace(reason)[:500] or "Unknown error"
    return record


def append_jsonl_line(outfile, record: Dict[str, Any]) -> None:
    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    outfile.flush()


def count_existing_output_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0

    with open(path, "r", encoding="utf-8") as output_file:
        return sum(1 for line in output_file if line.strip())


def get_retry_backoff_seconds(attempt: int, error: Exception) -> float:
    if isinstance(error, APICallError):
        if error.retry_after:
            return max(1.0, error.retry_after)
        if error.status_code == 429:
            return 5.0 * attempt
        if error.status_code and error.status_code >= 500:
            return min(12.0, 2.0 * attempt)
        if error.context_overflow:
            return 1.0
    return min(8.0, float(attempt))


# ================================
# MAIN PIPELINE
# ================================
def process_bulk() -> None:
    print("Starting bulk extraction...\n")

    if not os.path.exists(INPUT_FILE):
        print("Input file not found")
        return

    start_index = count_existing_output_lines(OUTPUT_FILE)
    if start_index:
        print(f"Resuming from input line #{start_index + 1}")

    session = build_session()
    rate_limiter = RateLimiter(REQUEST_INTERVAL_SECONDS)

    success_count = 0
    skipped_count = 0

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
            OUTPUT_FILE, "a", encoding="utf-8"
        ) as outfile:
            for index, raw_line in enumerate(infile):
                if index < start_index:
                    continue

                line_number = index + 1
                line = raw_line.strip()

                if not line:
                    append_jsonl_line(outfile, build_skip_record({}, line_number, "Empty input line"))
                    skipped_count += 1
                    print(f"Skipped line #{line_number}: empty input")
                    continue

                try:
                    raw_jd = json.loads(line)
                except json.JSONDecodeError as exc:
                    append_jsonl_line(
                        outfile,
                        build_skip_record({}, line_number, f"Invalid JSON input: {exc.msg}"),
                    )
                    skipped_count += 1
                    print(f"Skipped line #{line_number}: invalid JSON input")
                    continue

                if not isinstance(raw_jd, dict):
                    append_jsonl_line(
                        outfile,
                        build_skip_record({}, line_number, "Input line must contain a JSON object"),
                    )
                    skipped_count += 1
                    print(f"Skipped line #{line_number}: input is not a JSON object")
                    continue

                job_title = clean_whitespace(raw_jd.get("job_title", ""))
                jd_text = clean_whitespace(raw_jd.get("raw_jd_text", ""))

                if not jd_text:
                    append_jsonl_line(
                        outfile,
                        build_skip_record(raw_jd, line_number, "Missing or empty raw_jd_text"),
                    )
                    skipped_count += 1
                    print(f"Skipped line #{line_number}: empty JD text")
                    continue

                print(f"Processing JD #{line_number}")

                last_error: Optional[str] = None
                success = False

                for attempt in range(1, MAX_RETRIES + 1):
                    trim_factor = RETRY_TRIM_FACTORS[min(attempt - 1, len(RETRY_TRIM_FACTORS) - 1)]
                    token_budget = max(MIN_JD_TOKEN_BUDGET, int(INITIAL_JD_TOKEN_BUDGET * trim_factor))
                    prompt = build_prompt(job_title, jd_text, token_budget)

                    try:
                        response_text = call_sarvam(session, rate_limiter, prompt)
                        parsed = extract_json(response_text)

                        if parsed is None:
                            raise ValueError("Could not extract valid JSON from model response")

                        record = normalize_extracted_record(parsed, raw_jd)
                        append_jsonl_line(outfile, record)
                        success_count += 1
                        success = True
                        print(f"Saved JD #{line_number}")
                        break

                    except (APICallError, ValueError) as exc:
                        last_error = clean_whitespace(str(exc))
                        backoff_seconds = get_retry_backoff_seconds(attempt, exc)

                        if isinstance(exc, APICallError) and exc.retry_after:
                            rate_limiter.defer(exc.retry_after)

                        print(
                            f"Retry {attempt}/{MAX_RETRIES} for JD #{line_number}: "
                            f"{last_error or 'Unknown error'}"
                        )

                        if attempt < MAX_RETRIES:
                            time.sleep(backoff_seconds)

                if not success:
                    append_jsonl_line(
                        outfile,
                        build_skip_record(raw_jd, line_number, last_error or "Max retries exceeded"),
                    )
                    skipped_count += 1
                    print(f"Skipped JD #{line_number} after {MAX_RETRIES} attempts")

    finally:
        session.close()

    total_handled = success_count + skipped_count
    print(
        f"\nDone. Handled {total_handled} lines in this run "
        f"({success_count} saved, {skipped_count} skipped)."
    )


# ================================
# ENTRY
# ================================
if __name__ == "__main__":
    process_bulk()
