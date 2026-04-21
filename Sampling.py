import json
import random

INPUT_FILE = "structured_jds.jsonl"
SAMPLE_FILE = "sample_jds.jsonl"

# Load all structured data
with open(INPUT_FILE, "r") as f:
    data = [json.loads(line) for line in f]

# Random sample
sample_size = 75
sample = random.sample(data, sample_size)

# Save sample
with open(SAMPLE_FILE, "w") as f:
    for row in sample:
        f.write(json.dumps(row) + "\n")

print(f"Sampled {sample_size} JDs")