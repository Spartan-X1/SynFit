"""
Generate lean JD + candidate pairs for LLM resume rendering.

Outputs llm_resume_input.jsonl where each record contains only
the fields needed to render a natural-language resume — all scoring
metadata, skill-path internals, and domain routing signals are stripped.

Usage:
    python generate_llm_inputs.py                          # stratified 100-sample
    python generate_llm_inputs.py --limit 0               # all 735 JDs
    python generate_llm_inputs.py --limit 50 --seed 7
"""

import argparse
import json
import random
import sys
from collections import defaultdict

from Resume_Profile_Generator import generate_candidate, load_jds, load_json, load_skill_taxonomy
from assemble_corpus import generate_candidate_for_preset


PRESETS = ["full_match", "balanced", "hard_negative", "mismatch"]

# Maps a job_title (lowercased) keyword → family label used for stratification
ROLE_FAMILY_MAP = [
    ("android",           "Android Developer"),
    ("ios",               "iOS Developer"),
    ("machine learning",  "ML Engineer"),
    ("data scientist",    "Data Scientist"),
    ("data engineer",     "Data Engineer"),
    ("devops",            "DevOps/SRE"),
    ("site reliability",  "DevOps/SRE"),
    ("sre",               "DevOps/SRE"),
    ("qa",                "QA Engineer"),
    ("frontend",          "Frontend Developer"),
    ("front end",         "Frontend Developer"),
    ("ui/ux",             "UI/UX Designer"),
    ("ux",                "UI/UX Designer"),
    ("database",          "DBA"),
    ("java",              "Java Developer"),
    ("fullstack",         "Fullstack Developer"),
    ("full stack",        "Fullstack Developer"),
    ("cloud architect",   "Cloud Architect"),
    ("blockchain",        "Blockchain Developer"),
    ("backend",           "Backend Developer"),
    ("back end",          "Backend Developer"),
    ("engineering manager", "Engineering Manager"),
    ("product manager",   "Product Manager"),
]


def role_family(job_title: str) -> str:
    lower = job_title.lower()
    for key, family in ROLE_FAMILY_MAP:
        if key in lower:
            return family
    return "Other"


def stratified_sample(jds: list, n: int, seed: int) -> list:
    """
    Pick n JDs proportionally from each role family so that every
    role type is represented in the sample.
    """
    rng = random.Random(seed)
    by_family = defaultdict(list)
    for idx, jd in enumerate(jds):
        by_family[role_family(jd.get("job_title", ""))].append((idx, jd))

    # Proportional allocation — round down then top-up with largest remainders
    families = list(by_family.keys())
    total = len(jds)
    exact = {f: n * len(by_family[f]) / total for f in families}
    allocated = {f: int(exact[f]) for f in families}
    remainder = n - sum(allocated.values())
    # Distribute leftover slots to families with largest fractional parts
    families_by_remainder = sorted(families, key=lambda f: -(exact[f] - allocated[f]))
    for f in families_by_remainder[:remainder]:
        allocated[f] += 1

    sampled = []
    for f in families:
        pool = by_family[f]
        k = min(allocated[f], len(pool))
        sampled.extend(rng.sample(pool, k))

    # Shuffle final list so presets interleave across families
    rng.shuffle(sampled)
    return sampled


def lean_jd(jd: dict) -> dict:
    """Only what the LLM needs to understand the role."""
    return {
        "job_title": jd.get("job_title"),
        "domain": jd.get("domain"),
        "min_experience_years": jd.get("min_experience_years"),
        "core_technical_skills": jd.get("core_technical_skills", []),
        "education_requirement": jd.get("education_requirement"),
        "is_management_role": jd.get("is_management_role", False),
    }


def lean_experience(entry: dict) -> dict:
    """Strip internal routing fields; keep resume-renderable content."""
    return {
        "company": entry.get("company"),
        "role_title": entry.get("role_title"),
        "duration_years": entry.get("duration_years"),
        "primary_tech_stack": entry.get("primary_tech_stack", []),
        "focus_areas": entry.get("focus_areas", []),
    }


def lean_project(proj: dict) -> dict:
    """Keep everything needed to write a project section bullet."""
    return {
        "project_name": proj.get("project_name"),
        "business_problem": proj.get("business_problem"),
        "architecture_type": proj.get("architecture_type"),
        "technologies_used": proj.get("technologies_used", []),
        "impact_signals": proj.get("impact_signals", []),
        "quantitative_metrics": proj.get("quantitative_metrics", {}),
    }


def lean_candidate(sp: dict) -> dict:
    """Strip all scoring / metadata; keep only resume-buildable content."""
    return {
        "education": sp.get("education", {}),
        "skills": sp.get("skills", []),
        "experience": [lean_experience(e) for e in sp.get("experience", [])],
        "projects": [lean_project(p) for p in sp.get("projects", [])],
        "logistics": sp.get("logistics", {}),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate lean LLM resume inputs.")
    parser.add_argument("--jds",        default="structured_jds_normalize.jsonl")
    parser.add_argument("--taxonomy",   default="skill_pool_graph_v2.jsonl")
    parser.add_argument("--companies",  default="companies_pool.json")
    parser.add_argument("--colleges",   default="colleges_pool.json")
    parser.add_argument("--out",        default="llm_resume_input.jsonl")
    parser.add_argument("--limit",      type=int, default=100,
                        help="Number of JDs to sample (0 = all, stratified by role family).")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    companies = load_json(args.companies)
    colleges  = load_json(args.colleges)
    taxonomy  = load_skill_taxonomy(args.taxonomy)
    all_jds   = load_jds(args.jds)

    if args.limit > 0:
        sampled = stratified_sample(all_jds, args.limit, args.seed)
    else:
        sampled = list(enumerate(all_jds))   # (original_index, jd)

    records = []
    family_counts = defaultdict(int)
    company_usage = {}   # shared across all candidates — penalises overused companies

    for sample_num, (orig_idx, jd) in enumerate(sampled, start=1):
        family = role_family(jd.get("job_title", ""))
        family_counts[family] += 1
        jd_id = f"jd_{orig_idx + 1:05d}"

        for preset in PRESETS:
            candidate = generate_candidate_for_preset(
                jd, companies, colleges, taxonomy,
                preset, args.max_attempts,
                company_usage=company_usage,
            )
            sp = candidate.get("structured_profile", {})

            records.append({
                "id":      f"{jd_id}_{preset}",
                "jd_id":   jd_id,
                "preset":  preset,          # keep so you can filter/analyse later
                "jd":      lean_jd(jd),
                "candidate": lean_candidate(sp),
            })

        # Live progress every 10 JDs
        if sample_num % 10 == 0 or sample_num == len(sampled):
            print(f"  [{sample_num:3d}/{len(sampled)}] done", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "total_jds_sampled": len(sampled),
        "total_records": len(records),
        "presets": PRESETS,
        "output_file": args.out,
        "family_distribution": dict(sorted(family_counts.items(), key=lambda x: -x[1])),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
