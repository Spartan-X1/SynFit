import argparse
import json
import random

from Resume_Profile_Generator import generate_candidate, load_jds, load_json, load_skill_taxonomy


DEFAULT_PRESETS = ["full_match", "balanced", "hard_negative", "mismatch"]


PRESET_ACCEPTANCE_RULES = {
    "full_match": {"min_score": 0.72},
    "balanced": {"min_score": 0.50, "max_score": 0.82},
    "hard_negative": {"min_score": 0.32, "max_score": 0.72, "max_experience_fit": 0.85},
    "mismatch": {"max_score": 0.45, "require_role_mismatch": True},
}


def jd_view(jd):
    return {
        "job_title": jd.get("job_title"),
        "min_experience_years": jd.get("min_experience_years"),
        "core_technical_skills": jd.get("core_technical_skills", []),
        "domain": jd.get("domain"),
        "is_management_role": jd.get("is_management_role"),
        "education_requirement": jd.get("education_requirement"),
    }


def candidate_view(candidate):
    return {
        "structured_profile": candidate["structured_profile"],
        "profile_preset": candidate["profile_preset"],
        "candidate_role_key": candidate["candidate_role_key"],
        "jd_role_key": candidate["jd_role_key"],
        "overall_relevance_score": candidate["overall_relevance_score"],
        "score_breakdown": candidate["score_breakdown"],
        "score_rationale": candidate["score_rationale"],
        "score_metadata": candidate.get("score_metadata", {}),
        "effective_required_years": candidate.get("effective_required_years"),
        "experience_signal_source": candidate.get("experience_signal_source"),
        "title_seniority_band": candidate.get("title_seniority_band"),
        "dials": candidate.get("dials", {}),
    }


def build_pointwise_record(jd_id, jd, candidate_id, candidate):
    return {
        "jd_id": jd_id,
        "candidate_id": candidate_id,
        "instruction": "Given the job description and candidate profile, rate the candidate's fit on a scale from 0 to 10.",
        "jd": jd_view(jd),
        "candidate": candidate["structured_profile"],
        "relevance_score": round(candidate["overall_relevance_score"] * 10, 2),
        "relevance_score_0_to_1": candidate["overall_relevance_score"],
        "score_rationale": candidate["score_rationale"],
        "preset": candidate["profile_preset"],
        "jd_role_key": candidate["jd_role_key"],
        "candidate_role_key": candidate["candidate_role_key"],
        "score_breakdown": candidate["score_breakdown"],
        "score_metadata": candidate.get("score_metadata", {}),
        "effective_required_years": candidate.get("effective_required_years"),
        "experience_signal_source": candidate.get("experience_signal_source"),
        "title_seniority_band": candidate.get("title_seniority_band"),
        "dials": candidate.get("dials", {}),
    }


def build_grouped_record(jd_id, jd, ranked_candidates):
    return {
        "jd_id": jd_id,
        "jd": jd_view(jd),
        "candidates": [
            {
                "rank": index + 1,
                "candidate_id": candidate["candidate_id"],
                **candidate_view(candidate["candidate"]),
            }
            for index, candidate in enumerate(ranked_candidates)
        ],
    }


def candidate_fits_preset(candidate):
    rules = PRESET_ACCEPTANCE_RULES.get(candidate["profile_preset"], {})
    score = candidate["overall_relevance_score"]
    breakdown = candidate.get("score_breakdown", {})

    min_score = rules.get("min_score")
    if min_score is not None and score < min_score:
        return False

    max_score = rules.get("max_score")
    if max_score is not None and score > max_score:
        return False

    max_experience_fit = rules.get("max_experience_fit")
    if max_experience_fit is not None and breakdown.get("experience_fit", 1.0) > max_experience_fit:
        return False

    if rules.get("require_role_mismatch") and candidate["candidate_role_key"] == candidate["jd_role_key"]:
        return False

    return True


def generate_candidate_for_preset(jd, companies, colleges, taxonomy, preset, max_attempts, company_usage=None):
    chosen = None
    for _ in range(max_attempts):
        candidate = generate_candidate(
            jd,
            companies,
            colleges,
            taxonomy,
            profile_preset=preset,
            company_usage=company_usage,
        )
        chosen = candidate
        if candidate_fits_preset(candidate):
            return candidate
    return chosen


def build_pairwise_records(jd_id, jd, ranked_candidates, min_gap, clean_gap):
    records = []
    for left_index, chosen in enumerate(ranked_candidates):
        for rejected in ranked_candidates[left_index + 1 :]:
            gap = round(
                chosen["candidate"]["overall_relevance_score"] - rejected["candidate"]["overall_relevance_score"],
                3,
            )
            if gap < min_gap:
                continue

            pair_type = "clean_preference" if gap >= clean_gap else "hard_example"
            records.append(
                {
                    "jd_id": jd_id,
                    "pair_type": pair_type,
                    "score_gap": gap,
                    "jd": jd_view(jd),
                    "chosen": {
                        "candidate_id": chosen["candidate_id"],
                        **candidate_view(chosen["candidate"]),
                    },
                    "rejected": {
                        "candidate_id": rejected["candidate_id"],
                        **candidate_view(rejected["candidate"]),
                    },
                }
            )
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Assemble grouped JD-candidate training corpora.")
    parser.add_argument("--jds", default="structured_jds_normalize.jsonl")
    parser.add_argument("--taxonomy", default="skill_pool_graph_v2.jsonl")
    parser.add_argument("--companies", default="companies_pool.json")
    parser.add_argument("--colleges", default="colleges_pool.json")
    parser.add_argument("--pointwise-out", default="pointwise_training.jsonl")
    parser.add_argument("--pairwise-out", default="pairwise_training.jsonl")
    parser.add_argument("--grouped-out", default="grouped_training.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Optional JD limit for quick runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants-per-preset", type=int, default=1)
    parser.add_argument("--max-attempts-per-candidate", type=int, default=6)
    parser.add_argument("--pair-min-gap", type=float, default=1.0)
    parser.add_argument("--pair-clean-gap", type=float, default=3.0)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=DEFAULT_PRESETS,
        choices=DEFAULT_PRESETS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    companies = load_json(args.companies)
    colleges = load_json(args.colleges)
    taxonomy = load_skill_taxonomy(args.taxonomy)
    jds = load_jds(args.jds)

    if args.limit > 0:
        jds = jds[: args.limit]

    pointwise_records = []
    pairwise_records = []
    grouped_records = []

    for jd_index, jd in enumerate(jds, start=1):
        generated_candidates = []
        candidate_serial = 1

        for preset in args.presets:
            for _ in range(args.variants_per_preset):
                candidate = generate_candidate_for_preset(
                    jd,
                    companies,
                    colleges,
                    taxonomy,
                    preset,
                    args.max_attempts_per_candidate,
                )
                candidate_id = f"jd_{jd_index:05d}_cand_{candidate_serial:02d}_{preset}"
                generated_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate": candidate,
                    }
                )
                pointwise_records.append(build_pointwise_record(jd_index, jd, candidate_id, candidate))
                candidate_serial += 1

        ranked_candidates = sorted(
            generated_candidates,
            key=lambda item: item["candidate"]["overall_relevance_score"],
            reverse=True,
        )
        grouped_records.append(build_grouped_record(jd_index, jd, ranked_candidates))
        pairwise_records.extend(
            build_pairwise_records(
                jd_index,
                jd,
                ranked_candidates,
                min_gap=args.pair_min_gap / 10.0,
                clean_gap=args.pair_clean_gap / 10.0,
            )
        )

    with open(args.pointwise_out, "w", encoding="utf-8") as handle:
        for record in pointwise_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(args.pairwise_out, "w", encoding="utf-8") as handle:
        for record in pairwise_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(args.grouped_out, "w", encoding="utf-8") as handle:
        for record in grouped_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "jds": len(jds),
                "pointwise_records": len(pointwise_records),
                "pairwise_records": len(pairwise_records),
                "grouped_records": len(grouped_records),
                "presets": args.presets,
                "variants_per_preset": args.variants_per_preset,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
