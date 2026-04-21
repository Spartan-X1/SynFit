import argparse
import json
from typing import Dict, List

from skill_normalizer import normalize_record_skills


def normalize_file(path: str, output_path: str) -> Dict[str, int]:
    total = 0
    changed = 0
    removed = 0
    added = 0
    normalized_rows: List[Dict] = []

    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1

            original_skills = record.get("core_technical_skills", [])
            normalized_record = normalize_record_skills(record)
            normalized_skills = normalized_record.get("core_technical_skills", [])

            if original_skills != normalized_skills:
                changed += 1
                removed += max(0, len(original_skills) - len(normalized_skills))
                added += max(0, len(normalized_skills) - len(original_skills))

            normalized_rows.append(normalized_record)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for record in normalized_rows:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "total": total,
        "changed": changed,
        "removed": removed,
        "added": added,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize extracted core skills in JSONL files.")
    parser.add_argument("files", nargs="+", help="Input JSONL files to normalize")
    parser.add_argument("--in-place", action="store_true", help="Rewrite files in place")
    parser.add_argument("--output-suffix", default=".normalized", help="Suffix for output files when not in-place")
    args = parser.parse_args()

    for path in args.files:
        output_path = path if args.in_place else f"{path}{args.output_suffix}"
        stats = normalize_file(path, output_path)
        print(
            f"{path} -> {output_path} | "
            f"rows={stats['total']} changed={stats['changed']} removed={stats['removed']} added={stats['added']}"
        )


if __name__ == "__main__":
    main()
