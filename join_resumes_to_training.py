"""
Join generated resume text into pointwise / pairwise / grouped training files.

The training files use candidate IDs like:
    jd_00001_cand_01_full_match
The resume output uses IDs like:
    jd_00001_full_match

Mapping: strip '_cand_NN_' from the training candidate_id to get the resume id.

Usage:
    python join_resumes_to_training.py
    python join_resumes_to_training.py \
        --resumes  llm_resume_output_full_v2_final.jsonl \
        --pointwise pointwise_training.jsonl \
        --pairwise  pairwise_training.jsonl \
        --grouped   grouped_training.jsonl \
        --out-dir   training_with_resumes/

Output files (in --out-dir):
    pointwise_with_resumes.jsonl
    pairwise_with_resumes.jsonl
    grouped_with_resumes.jsonl
    join_stats.json

Strips internal scoring/metadata fields that should not be in training data:
    skill_alignment, company_domain, score_metadata, dials,
    experience_signal_source, title_seniority_band, effective_required_years
"""

import argparse
import json
import os
import re
import sys


# Fields to drop from candidate sub-objects (scoring internals)
STRIP_FIELDS = {
    "skill_alignment",
    "company_domain",
    "score_metadata",
    "dials",
    "experience_signal_source",
    "title_seniority_band",
    "effective_required_years",
}


def load_resume_index(path: str) -> dict:
    """Return {resume_id: resume_text} — skip tombstones."""
    index = {}
    missing_text = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("resume_text")
            if text:
                index[rec["id"]] = text
            else:
                missing_text += 1
    print(f"Loaded {len(index)} resumes ({missing_text} tombstones skipped) from {path}")
    return index


def cand_id_to_resume_id(candidate_id: str) -> str:
    """
    jd_00001_cand_01_full_match  →  jd_00001_full_match
    jd_00001_cand_01_hard_negative  →  jd_00001_hard_negative
    """
    return re.sub(r"_cand_\d+_", "_", candidate_id)


def strip_internal(obj: dict) -> dict:
    """Remove scoring-internal fields from a candidate dict."""
    return {k: v for k, v in obj.items() if k not in STRIP_FIELDS}


def process_pointwise(src_path: str, resume_index: dict, out_path: str):
    added = skipped = tombstoned = 0
    with open(src_path, "r", encoding="utf-8") as inf, \
         open(out_path, "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            cid = rec.get("candidate_id", "")
            rid = cand_id_to_resume_id(cid)
            resume_text = resume_index.get(rid)

            if not resume_text:
                tombstoned += 1
                skipped += 1
                continue   # drop records whose resume failed

            # Build output record
            out = {
                "jd_id":        rec["jd_id"],
                "candidate_id": cid,
                "resume_id":    rid,
                "resume_text":  resume_text,
                "instruction":  rec.get("instruction"),
                "jd":           rec.get("jd"),
                "candidate":    strip_internal(rec.get("candidate", {})),
                "relevance_score":       rec.get("relevance_score"),
                "relevance_score_0_to_1": rec.get("relevance_score_0_to_1"),
                "score_rationale": rec.get("score_rationale"),
                "preset":        rec.get("preset"),
                "jd_role_key":   rec.get("jd_role_key"),
                "candidate_role_key": rec.get("candidate_role_key"),
                "score_breakdown": rec.get("score_breakdown"),
            }
            outf.write(json.dumps(out, ensure_ascii=False) + "\n")
            added += 1

    print(f"  pointwise: {added} written, {skipped} dropped (no resume)")
    return added, skipped


def process_pairwise(src_path: str, resume_index: dict, out_path: str):
    added = skipped = 0
    with open(src_path, "r", encoding="utf-8") as inf, \
         open(out_path, "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            chosen_id   = cand_id_to_resume_id(rec.get("chosen", {}).get("candidate_id", ""))
            rejected_id = cand_id_to_resume_id(rec.get("rejected", {}).get("candidate_id", ""))

            chosen_text   = resume_index.get(chosen_id)
            rejected_text = resume_index.get(rejected_id)

            if not chosen_text or not rejected_text:
                skipped += 1
                continue

            chosen_out   = strip_internal(rec["chosen"])
            rejected_out = strip_internal(rec["rejected"])
            chosen_out["resume_text"]   = chosen_text
            chosen_out["resume_id"]     = chosen_id
            rejected_out["resume_text"] = rejected_text
            rejected_out["resume_id"]   = rejected_id

            out = {
                "jd_id":      rec["jd_id"],
                "pair_type":  rec.get("pair_type"),
                "score_gap":  rec.get("score_gap"),
                "jd":         rec.get("jd"),
                "chosen":     chosen_out,
                "rejected":   rejected_out,
            }
            outf.write(json.dumps(out, ensure_ascii=False) + "\n")
            added += 1

    print(f"  pairwise: {added} written, {skipped} dropped (missing resume in pair)")
    return added, skipped


def process_grouped(src_path: str, resume_index: dict, out_path: str):
    added = skipped_groups = skipped_cands = 0
    with open(src_path, "r", encoding="utf-8") as inf, \
         open(out_path, "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            enriched_candidates = []
            for cand in rec.get("candidates", []):
                cid = cand.get("candidate_id", "")
                rid = cand_id_to_resume_id(cid)
                resume_text = resume_index.get(rid)
                if not resume_text:
                    skipped_cands += 1
                    continue
                c_out = strip_internal(cand)
                c_out["resume_text"] = resume_text
                c_out["resume_id"]   = rid
                enriched_candidates.append(c_out)

            if len(enriched_candidates) < 2:
                # Not enough candidates to rank — drop the group
                skipped_groups += 1
                continue

            out = {
                "jd_id":      rec["jd_id"],
                "jd":         rec.get("jd"),
                "candidates": enriched_candidates,
            }
            outf.write(json.dumps(out, ensure_ascii=False) + "\n")
            added += 1

    print(f"  grouped: {added} groups written, {skipped_groups} groups dropped (<2 candidates), "
          f"{skipped_cands} individual candidates skipped")
    return added, skipped_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Join resume text into training files.")
    parser.add_argument("--resumes",    default="llm_resume_output_full_v2_final.jsonl",
                        help="Final resume output (after retry pass).")
    parser.add_argument("--pointwise",  default="pointwise_training.jsonl")
    parser.add_argument("--pairwise",   default="pairwise_training.jsonl")
    parser.add_argument("--grouped",    default="grouped_training.jsonl")
    parser.add_argument("--out-dir",    default="training_with_resumes",
                        help="Output directory for enriched training files.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check source files exist
    for path in [args.resumes, args.pointwise, args.pairwise, args.grouped]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    resume_index = load_resume_index(args.resumes)

    pw_added, pw_skipped = process_pointwise(
        args.pointwise, resume_index,
        os.path.join(args.out_dir, "pointwise_with_resumes.jsonl")
    )
    pair_added, pair_skipped = process_pairwise(
        args.pairwise, resume_index,
        os.path.join(args.out_dir, "pairwise_with_resumes.jsonl")
    )
    grp_added, grp_skipped = process_grouped(
        args.grouped, resume_index,
        os.path.join(args.out_dir, "grouped_with_resumes.jsonl")
    )

    stats = {
        "resume_index_size": len(resume_index),
        "pointwise":  {"written": pw_added,   "dropped": pw_skipped},
        "pairwise":   {"written": pair_added,  "dropped": pair_skipped},
        "grouped":    {"written": grp_added,   "dropped": grp_skipped},
        "output_dir": args.out_dir,
    }
    stats_path = os.path.join(args.out_dir, "join_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print("\n" + json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
