import json

# ================================
# FILE PATHS
# ================================
STRUCTURED_SAMPLE_FILE = "sample_jds.jsonl"
RAW_FILE = "fresh_naukri_jds.jsonl"
OUTPUT_FILE = "evaluation_dataset.jsonl"

# ================================
# LOAD RAW DATA → BUILD MAP
# ================================
print("📦 Loading raw dataset...")

raw_map = {}

with open(RAW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            jd = json.loads(line)
            url = jd.get("source_url")
            if url:
                raw_map[url] = jd
        except:
            continue

print(f"✅ Loaded {len(raw_map)} raw JDs")

# ================================
# PROCESS SAMPLE + MATCH
# ================================
print("🔗 Matching structured samples with raw JDs...")

matched = 0
missing = 0

with open(STRUCTURED_SAMPLE_FILE, "r", encoding="utf-8") as sf, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for line in sf:
        try:
            structured = json.loads(line)
            url = structured.get("source_url")

            if not url:
                missing += 1
                continue

            raw = raw_map.get(url)

            if not raw:
                missing += 1
                continue

            combined = {
                "source_url": url,
                "raw_jd_text": raw.get("raw_jd_text", ""),
                "structured_output": structured
            }

            out.write(json.dumps(combined) + "\n")
            matched += 1

        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

# ================================
# SUMMARY
# ================================
print("\n📊 MATCHING COMPLETE")
print(f"✅ Matched: {matched}")
print(f"❌ Missing: {missing}")
print(f"📁 Output saved to: {OUTPUT_FILE}")