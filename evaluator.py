import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests


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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")


# ================================
# CONFIG
# ================================
INPUT_FILE = "evaluation_dataset.jsonl"
OUTPUT_FILE = "evaluation_results.jsonl"

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
REQUEST_DELAY = 6
REQUEST_TIMEOUT = (15, 90)
MAX_RETRIES = 3

BINARY_SCORE_KEYS = (
    "job_title",
    "min_experience_years",
    "domain",
    "is_management_role",
    "education_requirement",
    "max_notice_period_days",
)
CONTINUOUS_SCORE_KEYS = ("core_technical_skills", "overall_score")
REQUIRED_SCORE_KEYS = BINARY_SCORE_KEYS + CONTINUOUS_SCORE_KEYS


# ================================
# HELPERS
# ================================
def clean_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def trim_text(text: Any, max_chars: int = 2500) -> str:
    cleaned = clean_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned

    head_chars = int(max_chars * 0.72)
    tail_chars = max_chars - head_chars - len(" [TRUNCATED] ")

    if tail_chars < 200:
        tail_chars = 200
        head_chars = max_chars - tail_chars - len(" [TRUNCATED] ")

    head = cleaned[:head_chars].rsplit(" ", 1)[0].strip()
    tail = cleaned[-tail_chars:].split(" ", 1)[-1].strip()

    if not head or not tail:
        return cleaned[:max_chars].strip()

    return f"{head} [TRUNCATED] {tail}"


# ================================
# BUILD PROMPT
# ================================
def build_system_prompt() -> str:
    return (
        "You are a strict evaluation engine. "
        "Return exactly one valid JSON object and nothing else. "
        "Do not use markdown, code fences, commentary, notes, or extra keys."
    )


def build_prompt(raw_jd: str, structured: Dict[str, Any]) -> str:
    structured_view = {
        key: structured.get(key)
        for key in (
            "job_title",
            "min_experience_years",
            "core_technical_skills",
            "domain",
            "is_management_role",
            "education_requirement",
            "max_notice_period_days",
            "executive_summary",
        )
    }

    return f"""
You are an expert evaluator for a structured information extraction system.

Your task is to compare a RAW JOB DESCRIPTION with a STRUCTURED OUTPUT.

The structured output was generated using strict extraction rules.
You MUST evaluate using the SAME rules.

-------------------------------------
EXTRACTION LOGIC (CRITICAL)
-------------------------------------

- Only explicitly stated information is extracted
- No hallucinations allowed

min_experience_years:
- Must be the LOWEST required value
- If range exists (e.g., "3–7 years") → correct value = 3
- If not mentioned → 0 is correct

max_notice_period_days:
- Immediate → 0
- Explicit number → must match
- Not mentioned → 90 is CORRECT (default, do not penalize)

education_requirement:
- Must preserve required vs preferred
- If missing → "Not Specified"

core_technical_skills:
- Only specific tools, languages, frameworks, databases, cloud platforms
- Ignore order
- Do NOT require exact wording

is_management_role:
- TRUE if ANY of:
  - leading teams
  - managing people
  - mentoring
  - ownership of team output
- FALSE otherwise
- Do NOT require explicit word "manager"

domain:
- Accept strong inference from company or role
- Do NOT require exact phrase match

executive_summary:
- Context only (do NOT score directly)

-------------------------------------
EVALUATION RULES
-------------------------------------

1. SEMANTIC MATCHING:
- If meaning is correct → mark as CORRECT
- Ignore formatting differences:
  - capitalization
  - hyphens
  - spacing

2. DO NOT PENALIZE:
- Minor wording differences
- Equivalent phrasing
- Different formatting

3. MARK INCORRECT ONLY IF:
- Factually wrong
- Missing clearly present information
- Hallucinated (not in raw JD)
- Violates extraction rules above

4. FIELD SCORING:

- job_title:
  Correct if semantically matches

- min_experience_years:
  Must match lowest required value logic

- core_technical_skills:
  Score between 0 and 1 based on overlap

- domain:
  Correct if reasonably inferred or explicitly stated

- is_management_role:
  Evaluate based on responsibility signals (not keywords only)

- education_requirement:
  Must preserve required vs preferred correctly

- max_notice_period_days:
  Must follow extraction rule logic exactly

-------------------------------------
OUTPUT FORMAT (STRICT JSON)
-------------------------------------

Return ONLY JSON:

{{
  "job_title": 0 or 1,
  "min_experience_years": 0 or 1,
  "core_technical_skills": 0 to 1,
  "domain": 0 or 1,
  "is_management_role": 0 or 1,
  "education_requirement": 0 or 1,
  "max_notice_period_days": 0 or 1,
  "overall_score": 0 to 1
}}

-------------------------------------
IMPORTANT:
-------------------------------------

- Be FAIR, not overly harsh
- This is semantic evaluation, NOT exact string matching
- Do NOT penalize correct default values
- If unsure, prefer partial credit instead of full rejection

-------------------------------------
RAW JOB DESCRIPTION:
{raw_jd}

-------------------------------------
STRUCTURED OUTPUT:
{json.dumps(structured_view, ensure_ascii=False)}
"""


# ================================
# GROQ CALL
# ================================
def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "jd-evaluator/1.0",
        }
    )
    return session


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
                nested_message = value.get("message") or value.get("detail")
                if isinstance(nested_message, str) and nested_message.strip():
                    return clean_whitespace(nested_message)

    return clean_whitespace(response.text) or f"HTTP {response.status_code}"


def call_groq(session: requests.Session, prompt: str) -> str:
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 220,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network/API request failed: {clean_whitespace(exc)}") from exc

    if response.status_code != 200:
        raise RuntimeError(extract_api_error_message(response))

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"API returned non-JSON response: {clean_whitespace(response.text[:500])}"
        ) from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        preview = clean_whitespace(json.dumps(data)[:500])
        raise RuntimeError(f"Unexpected API response shape: {preview}") from exc

    content = clean_whitespace(content)
    if not content:
        raise RuntimeError("API returned empty model content")

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


def score_candidate(candidate: Dict[str, Any]) -> int:
    score = 0
    for key in REQUIRED_SCORE_KEYS:
        if key in candidate:
            score += 3
    score += min(len(candidate), 10)
    return score


def coerce_json_candidate(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        best_dict = None
        best_score = -1
        for item in value:
            if not isinstance(item, dict):
                continue
            candidate_score = score_candidate(item)
            if candidate_score > best_score:
                best_dict = item
                best_score = candidate_score
        return best_dict

    return None


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

        candidate_score = score_candidate(candidate)
        if candidate_score > best_score:
            best_candidate = candidate
            best_score = candidate_score

    return best_candidate


def parse_score_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0

    if isinstance(value, (int, float)):
        return float(value)

    text = clean_whitespace(value).lower()
    if not text:
        return None

    if text in {"correct", "yes", "true", "pass"}:
        return 1.0
    if text in {"incorrect", "no", "false", "fail"}:
        return 0.0

    percentage_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if percentage_match:
        return float(percentage_match.group(1)) / 100.0

    number_match = re.search(r"-?\d+(?:\.\d+)?", text)
    if number_match:
        return float(number_match.group(0))

    return None


def normalize_scores(parsed: Dict[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}

    for key in BINARY_SCORE_KEYS:
        raw_value = parse_score_value(parsed.get(key))
        if raw_value is None:
            raise ValueError(f"Missing or invalid score for '{key}'")
        normalized[key] = min(1.0, max(0.0, raw_value))

    raw_skills = parse_score_value(parsed.get("core_technical_skills"))
    if raw_skills is None:
        raise ValueError("Missing or invalid score for 'core_technical_skills'")
    normalized["core_technical_skills"] = min(1.0, max(0.0, raw_skills))

    overall_raw = parse_score_value(parsed.get("overall_score"))
    if overall_raw is None:
        component_values = [normalized[key] for key in BINARY_SCORE_KEYS]
        component_values.append(normalized["core_technical_skills"])
        overall_raw = sum(component_values) / len(component_values)

    normalized["overall_score"] = min(1.0, max(0.0, overall_raw))
    return normalized


# ================================
# MAIN PIPELINE
# ================================
def run_evaluation() -> None:
    print("Starting evaluation...\n")

    results: List[Dict[str, float]] = []

    if not os.path.exists(INPUT_FILE):
        print("Input file not found")
        return

    session = build_session()

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
            OUTPUT_FILE, "w", encoding="utf-8"
        ) as outfile:
            for index, raw_line in enumerate(infile, 1):
                line = raw_line.strip()
                if not line:
                    print(f"Skipped #{index}: empty input line")
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Skipped #{index}: invalid input JSON ({exc.msg})")
                    continue

                raw_jd = trim_text(data.get("raw_jd_text", ""))
                structured = data.get("structured_output", {})
                source_url = clean_whitespace(data.get("source_url", "")) or "Unknown"

                if not raw_jd:
                    print(f"Skipped #{index}: empty raw_jd_text")
                    continue

                if not isinstance(structured, dict):
                    print(f"Skipped #{index}: structured_output is not a JSON object")
                    continue

                prompt = build_prompt(raw_jd, structured)

                print(f"Evaluating #{index}")

                last_error = "Unknown error"
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        response_text = call_groq(session, prompt)
                        parsed = extract_json(response_text)

                        if parsed is None:
                            preview = clean_whitespace(response_text[:220])
                            raise ValueError(f"JSON parse failed. Model output preview: {preview}")

                        normalized_scores = normalize_scores(parsed)
                        result = {"source_url": source_url, "scores": normalized_scores}

                        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                        outfile.flush()

                        results.append(normalized_scores)
                        print(f"Saved #{index}")
                        break

                    except Exception as exc:
                        last_error = clean_whitespace(str(exc))
                        print(f"Retry {attempt}/{MAX_RETRIES} for #{index}: {last_error}")

                        if attempt < MAX_RETRIES:
                            time.sleep(2.5 * attempt)
                else:
                    print(f"Skipped #{index} after {MAX_RETRIES} attempts: {last_error}")

                time.sleep(REQUEST_DELAY)

    finally:
        session.close()

    print("\nComputing metrics...\n")
    compute_metrics(results)


# ================================
# METRICS
# ================================
def compute_metrics(results: List[Dict[str, float]]) -> None:
    if not results:
        print("No results to evaluate")
        return

    metrics: Dict[str, float] = {}

    for field in BINARY_SCORE_KEYS:
        field_scores = [record[field] for record in results if field in record]
        metrics[field] = sum(field_scores) / len(field_scores)

    skill_scores = [record["core_technical_skills"] for record in results]
    metrics["skills_avg"] = sum(skill_scores) / len(skill_scores)

    overall_scores = [record["overall_score"] for record in results]
    metrics["overall_score"] = sum(overall_scores) / len(overall_scores)

    print("FINAL METRICS:\n")
    for key, value in metrics.items():
        print(f"{key}: {round(value * 100, 2)}%")


# ================================
# ENTRY
# ================================
if __name__ == "__main__":
    run_evaluation()
