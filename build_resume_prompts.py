"""
Build Sarvam-ready prompt payloads from llm_resume_input.jsonl.

Each output record in prompts_for_sarvam.jsonl contains:
  - id, jd_id, preset  (for traceability)
  - contact            (Faker-generated, deterministic per id)
  - prompt             (the full text string to send to Sarvam)

The prompt embeds all candidate + JD data directly — Sarvam only needs
to receive the prompt string and return the resume text.

Usage:
    python build_resume_prompts.py
    python build_resume_prompts.py --input llm_resume_input.jsonl --out prompts_for_sarvam.jsonl
"""

import argparse
import json
import hashlib
import random

from faker import Faker


# Indian cities commonly seen in tech resumes
INDIAN_TECH_CITIES = [
    "Bengaluru", "Hyderabad", "Pune", "Mumbai", "Chennai",
    "Delhi", "Noida", "Gurugram", "Kolkata", "Ahmedabad",
]


def make_contact(record_id: str) -> dict:
    """
    Generate a deterministic but realistic Indian contact block.
    Seeds the Faker RNG from the record id so the same id always
    gets the same contact details across re-runs.
    """
    seed = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % (2**32)
    fake = Faker("en_IN")
    Faker.seed(seed)
    rng = random.Random(seed)

    name  = fake.name()
    city  = rng.choice(INDIAN_TECH_CITIES)
    # Indian mobile: 10 digits starting with 6-9
    phone = f"+91 {rng.randint(6,9)}{rng.randint(10**8, 10**9 - 1)}"
    # Realistic-looking email from the name
    slug  = name.lower().replace(" ", ".").replace("'", "")
    domain = rng.choice(["gmail.com", "outlook.com", "yahoo.com"])
    email = f"{slug}{rng.randint(1, 99)}@{domain}"

    return {"name": name, "city": city, "phone": phone, "email": email}


def format_experience(exp_list: list) -> str:
    lines = []
    for e in exp_list:
        stack = ", ".join(e.get("primary_tech_stack", []))
        lines.append(f"  - {e['role_title']} at {e['company']} ({e['duration_years']} years)")
        lines.append(f"    Tech: {stack}")
        for fa in e.get("focus_areas", []):
            lines.append(f"    • {fa}")
    return "\n".join(lines)


def format_projects(proj_list: list) -> str:
    lines = []
    for p in proj_list:
        tech = ", ".join(p.get("technologies_used", []))
        metrics = "; ".join(f"{k}: {v}" for k, v in p.get("quantitative_metrics", {}).items())
        signals = "; ".join(p.get("impact_signals", []))
        lines.append(f"  - {p['project_name']}")
        lines.append(f"    Problem: {p['business_problem']}")
        lines.append(f"    Architecture: {p.get('architecture_type', 'N/A')} | Tech: {tech}")
        if signals:
            lines.append(f"    Impact: {signals}")
        if metrics:
            lines.append(f"    Metrics: {metrics}")
    return "\n".join(lines)


def build_prompt(record: dict, contact: dict) -> str:
    jd   = record["jd"]
    cand = record["candidate"]
    edu  = cand.get("education", {})
    log  = cand.get("logistics", {})

    skills_str   = ", ".join(cand.get("skills", []))
    exp_str      = format_experience(cand.get("experience", []))
    proj_str     = format_projects(cand.get("projects", []))

    prompt = f"""You are a resume writer. Write a concise one-page Indian tech resume. Output only the resume — no explanations, no preamble.

Rules:
- Plain text only, no markdown, no asterisks. Use dash (-) for bullets.
- Sections in order: Name/Contact, Summary (2 lines), Skills (grouped), Experience (reverse-chron), Projects, Details.
- Experience: 2-3 bullets per role, action verb first, embed tech and one metric per bullet.
- Projects: 1 line problem + 1 line impact + "Tech: ..." — keep each project under 3 lines total.
- Summary: 2 lines max tying candidate to the target role.
- Details: Notice Period and Current CTC only.
- Keep total output under 600 words.

TARGET ROLE: {jd['job_title']} | Domain: {jd.get('domain', 'N/A')} | Key skills: {', '.join(jd.get('core_technical_skills', []))}

CANDIDATE:
{contact['name']} | {contact['city']} | {contact['phone']} | {contact['email']}
Education: {edu.get('degree', 'N/A')}, {edu.get('institution', 'N/A')} ({edu.get('graduation_year', 'N/A')})
Skills: {skills_str}

Experience:
{exp_str}

Projects:
{proj_str}

Notice Period: {log.get('notice_period', 'N/A')} | Current CTC: {log.get('current_ctc', 'N/A')}"""

    return prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Build Sarvam prompt payloads.")
    parser.add_argument("--input", default="llm_resume_input.jsonl")
    parser.add_argument("--out",   default="prompts_for_sarvam.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    records = []
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    output = []
    for rec in records:
        contact = make_contact(rec["id"])
        prompt  = build_prompt(rec, contact)
        output.append({
            "id":      rec["id"],
            "jd_id":   rec["jd_id"],
            "preset":  rec["preset"],
            "contact": contact,
            "prompt":  prompt,
        })

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in output:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Written {len(output)} prompt records to {args.out}")
    # Show one sample prompt
    sample = output[0]
    print(f"\n--- SAMPLE PROMPT (id={sample['id']}) ---\n")
    print(sample["prompt"])


if __name__ == "__main__":
    main()
