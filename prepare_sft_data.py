"""
Clean pointwise_with_resumes.jsonl into a lean SFT-ready dataset.

Keeps:
  - resume_text            (model input)
  - jd: job_title, core_technical_skills, min_experience_years,
         domain, is_management_role, education_requirement
  - jd_role_key
  - candidate: education, skills,
               experience (company, duration_years, role_title,
                           primary_tech_stack, focus_areas),
               projects   (project_name, business_problem,
                           architecture_type, technologies_used,
                           complexity_tier, impact_signals,
                           quantitative_metrics)
  - score_breakdown        (7 axes — model output)
  - relevance_score        (0-10 scalar — model output)
  - score_rationale        (natural-language explanation — model output)

Drops everything else (IDs, logistics, skill_alignment, score_metadata,
dials, internal domain/path fields, redundant 0-to-1 score, preset, etc.)

Output: sft_pointwise.jsonl  (one record per line)

Usage:
    python prepare_sft_data.py
    python prepare_sft_data.py --input  training_with_resumes/pointwise_with_resumes.jsonl \
                               --output sft_pointwise.jsonl
"""

import argparse
import json
import sys

JD_KEEP = {"job_title", "core_technical_skills", "min_experience_years",
           "domain", "is_management_role", "education_requirement"}

EXP_KEEP = {"company", "duration_years", "role_title",
            "primary_tech_stack", "focus_areas"}

PROJ_KEEP = {"project_name", "business_problem", "architecture_type",
             "technologies_used", "complexity_tier",
             "impact_signals", "quantitative_metrics"}


def clean_experience(exp_list):
    return [
        {k: v for k, v in entry.items() if k in EXP_KEEP}
        for entry in exp_list
    ]


def clean_projects(proj_list):
    return [
        {k: v for k, v in entry.items() if k in PROJ_KEEP}
        for entry in proj_list
    ]


def clean_record(rec: dict) -> dict:
    cand_raw = rec.get("candidate", {})
    candidate = {
        "education":  cand_raw.get("education"),
        "skills":     cand_raw.get("skills", []),
        "experience": clean_experience(cand_raw.get("experience", [])),
        "projects":   clean_projects(cand_raw.get("projects", [])),
    }

    jd_raw = rec.get("jd", {})
    jd = {k: v for k, v in jd_raw.items() if k in JD_KEEP}

    return {
        "resume_text":     rec["resume_text"],
        "jd":              jd,
        "jd_role_key":     rec.get("jd_role_key"),
        "candidate":       candidate,
        "score_breakdown": rec.get("score_breakdown"),
        "relevance_score": rec.get("relevance_score"),
        "score_rationale": rec.get("score_rationale"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="training_with_resumes/pointwise_with_resumes.jsonl")
    p.add_argument("--output", default="sft_pointwise.jsonl")
    return p.parse_args()


def main():
    args = parse_args()

    written = 0
    with open(args.input, "r", encoding="utf-8") as inf, \
         open(args.output, "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cleaned = clean_record(rec)
            outf.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written {written} records → {args.output}")

    # Quick schema check on first record
    with open(args.output, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print("Top-level keys:", list(sample.keys()))
    print("JD keys:       ", list(sample["jd"].keys()))
    print("Candidate keys:", list(sample["candidate"].keys()))
    if sample["candidate"]["experience"]:
        print("Exp[0] keys:   ", list(sample["candidate"]["experience"][0].keys()))
    if sample["candidate"]["projects"]:
        print("Proj[0] keys:  ", list(sample["candidate"]["projects"][0].keys()))


if __name__ == "__main__":
    main()
