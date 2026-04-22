# SynFit

Synthetic resume generation pipeline + QLoRA fine-tuned Qwen2.5-3B for multi-axis JD-fit scoring.

**Scale:** 735 real scraped Naukri JDs → 720 sampled for training × 4 candidate presets = **2880 (JD, resume, score) triples**.

---

## What it does

Given a **job description** and a **resume**, the fine-tuned model extracts a structured candidate profile and scores the candidate's fit across 7 independent dimensions.

```
Input:  JD (structured) + Resume text (freeform)

Output: {
  "candidate":       { education, skills, experience, projects },
  "score_breakdown": {
    "skill_coverage":       0.33,
    "experience_fit":       1.00,
    "role_alignment":       1.00,
    "domain_alignment":     1.00,
    "education_fit":        0.45,
    "deployment_alignment": 0.50,
    "management_alignment": 1.00
  },
  "relevance_score": 6.94,
  "score_rationale": "Matches 1/3 JD skills including Python. Shows 1.5 years ..."
}
```

---

## Why synthetic data

Real resumes tied to real JDs would be a **data leak / privacy risk** and are practically impossible to collect at scale. Synthetic data lets us generate thousands of `(JD, resume, score)` triples with controlled label quality, zero PII, and guaranteed score-distribution coverage.

The 4 presets — `full_match`, `balanced`, `hard_negative`, `mismatch` — enforce balanced supervision per JD. Most ranker datasets have positive-heavy labels; this one is balanced by design.

---

## Pipeline overview

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Naukri.com JDs    │───▶│    LLM extractor    │───▶│   735 structured    │
│  (Playwright +      │    │    (Sarvam API)     │    │     JDs, then       │
│   stealth scrape)   │    │                     │    │    normalized       │
└─────────────────────┘    └─────────────────────┘    └──────────┬──────────┘
                                                                 │
                                                                 ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   2880 candidate    │◀───│  4 presets per JD   │◀───│  Profile generator  │
│     profiles        │    │  full/bal/hard/mis  │    │  (skill graph +     │
│                     │    │                     │    │   stratified sampl.)│
└──────────┬──────────┘    └─────────────────────┘    └─────────────────────┘
           │
           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Deterministic      │───▶│  Pointwise /        │───▶│    LLM (Llama 3.1   │
│  7-axis scorer      │    │  Pairwise /         │    │    8B via AICredits)│
│                     │    │  Grouped corpus     │    │    → resume text    │
└─────────────────────┘    └─────────────────────┘    └──────────┬──────────┘
                                                                 │
                                                                 ▼
                           ┌─────────────────────┐    ┌─────────────────────┐
                           │   Qwen2.5-3B QLoRA  │◀───│    Quality gates    │
                           │    fine-tune via    │    │   + retry + dedupe  │
                           │   Unsloth (T4 GPU)  │    │                     │
                           └─────────────────────┘    └─────────────────────┘
```

---

## Results

### JD extraction (Sarvam → structured fields)

Evaluated on a 75-JD random sample via LLM judge (Llama-4-Scout-17B on Groq), then spot-checked manually.

| Field                    | LLM-eval | Human-calibrated |
|--------------------------|----------|------------------|
| job_title                | 90.67%   | 95–98%           |
| min_experience_years     | 82.67%   | 88–93%           |
| core_technical_skills    | 78.81%   | 80–90%           |
| domain                   | 100.00%  | 95–100%          |
| is_management_role       | 54.67%   | 70–85%           |
| education_requirement    | 84.00%   | 85–92%           |
| max_notice_period_days   | 98.67%   | 98–100%          |
| **overall**              | **87.33%** | **90–94%**     |

95% CI on overall: ~83% – ~97%. LLM judge was systematically stricter than human raters on management-role classification and experience-years edge cases.

### Fine-tuned scorer (Qwen2.5-3B + QLoRA)

Evaluated on 100 held-out validation records (JD-stratified split, no JD leakage between train / val).

| Score axis                 | MAE     | Interpretation                  |
|----------------------------|---------|---------------------------------|
| skill_coverage             | 0.1011  | Good                            |
| experience_fit             | 0.1015  | Good                            |
| role_alignment             | 0.0160  | Excellent                       |
| domain_alignment           | 0.1755  | Acceptable                      |
| education_fit              | 0.0555  | Great                           |
| deployment_alignment       | 0.0866  | Good                            |
| management_alignment       | 0.0030  | Excellent                       |
| **relevance_score (0-10)** | **0.5866** | Off by ~0.59 points on average |

**JSON parse failure rate: 0/100** — every output was valid parseable JSON.

Training loss converged from **0.45 → 0.34** across 3 epochs on a single T4 GPU.

---

## Repository structure

All scripts sit flat at the repo root and are meant to run in sequence. Grouped by role:

**Scraping & JD extraction**
- `master_scraper.py` — Playwright + stealth, scrapes Naukri JDs across 19 role searches
- `extractor.py` — Sarvam API JD field extraction with token-budgeted prompt trimming + retries
- `normalize_structured_jds.py` — post-extraction skill & field normalization
- `skill_normalizer.py` — skill canonicalization + aliasing (e.g. `reactjs` → `React`)
- `evaluator.py` — LLM-as-judge quality check on extraction (Groq / Llama-4-Scout)
- `Sampling.py` — picks the 75-JD random evaluation sample

**Pool building**
- `pool.py` — builds `companies_pool.json` (from `Company_Lists.md`) and `colleges_pool.json` (from `colleges_raw.txt`)
- `Statistical_dataset.py` — matches raw ↔ structured JDs into evaluation-ready records

**Synthetic candidate generation**
- `Resume_Profile_Generator.py` — main candidate profile generator (profile presets, skill graph traversal, company/college sampling)
- `assemble_corpus.py` — builds pointwise / pairwise / grouped training corpora from candidates
- `generate_llm_inputs.py` — stratified JD-family sampling + lean input prep for resume prompts

**LLM resume generation & quality**
- `build_resume_prompts.py` — assembles Faker-generated Indian contact + candidate → prompt string
- `call_llm_resumes.py` — main LLM call loop (AICredits → Llama-3.1-8B-Instruct), 55 RPM, 800 max tokens, tombstone on failure
- `retry_failed_resumes.py` — retries tombstone records (network / API failures), merges into `*_final.jsonl`
- `rerun_bad_resumes.py` — quality-fix pass: regex checks for XYZ/ABC placeholders, duplicate company entries, generic openers → re-run with stricter system prompt

**Training**
- `join_resumes_to_training.py` — joins generated resume text back into pointwise / pairwise / grouped files
- `prepare_sft_data.py` — cleans `pointwise_with_resumes.jsonl` → lean `sft_pointwise.jsonl` (drops scoring internals, preset IDs, skill-path metadata)
- `qwen_finetune_lightning.ipynb` — QLoRA fine-tune notebook for Lightning.ai (Unsloth, SFTTrainer, Qwen2.5-3B 4-bit)

**Assets & reports**
- `Final_Metrix.txt` — JD extraction accuracy (automated + human-calibrated)
- `sample_jds_scores.csv` — sample scored JD-candidate pairs for inspection
- `Company_Lists.md` — company taxonomy by domain & CTC tier
- `colleges_raw.txt` — institution tier reference list

---

## The 7 scoring axes

Scores are deterministic — computed from the structured candidate profile against the JD using a weighted formula. The fine-tuned model learns to reproduce this function from resume text alone.

| Axis                  | Weight | What it measures                               |
|-----------------------|--------|------------------------------------------------|
| skill_coverage        | 0.38   | Overlap between JD skills and candidate skills |
| experience_fit        | 0.24   | Years-of-experience alignment                  |
| role_alignment        | 0.14   | Role family match (DS vs SWE vs DevOps etc.)   |
| domain_alignment      | 0.09   | Industry/domain match                          |
| education_fit         | 0.05   | Degree and institution tier                    |
| deployment_alignment  | 0.05   | Production-readiness signals                   |
| management_alignment  | 0.05   | Management role match                          |

Weights sum to 1.00 and are applied to produce the final `relevance_score` on a 0–10 scale.

---

## Quality gates on LLM-generated resumes

Llama-3.1-8B had well-known failure modes. Programmatic checks + selective re-runs closed the gaps:

| Failure mode                          | Detection                                | Fix                                     |
|---------------------------------------|------------------------------------------|-----------------------------------------|
| `XYZ Corporation` / `ABC Company`     | Regex: `\bXYZ\b\|ABC Company`            | Re-run with stricter system prompt      |
| Duplicate company entries             | Counter on `at <company>` lines          | Programmatic removal + re-run           |
| Duplicate section headers             | Line-level header tracking               | Remove duplicate header lines           |
| Generic openers ("Highly motivated")  | Regex scan                               | Re-run with stricter prompt             |
| Bracket placeholders `[Previous...]`  | Regex                                    | Re-run                                  |
| Phone/email formatting bleed          | Email regex over-capture                 | Reformat contact block                  |
| Network / API tombstones              | `_error` field on record                 | `retry_failed_resumes.py`               |

Final dataset: **2880 / 2880 resumes clean**, zero tombstones, zero placeholder hallucinations.

---

## Training configuration

```python
model          = "unsloth/Qwen2.5-3B-Instruct"    # 4-bit QLoRA
lora_r         = 16
lora_alpha     = 16
lora_dropout   = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

learning_rate           = 2e-4
lr_scheduler            = "cosine"
per_device_train_batch  = 2
gradient_accumulation   = 4                # effective batch = 8
num_epochs              = 3
warmup_steps            = 40
max_seq_length          = 2048
packing                 = True             # Unsloth-enabled
```

**Trainable parameters: 29.9M / 3.1B (0.96%)**

---

## Running it yourself

### 1. Install
```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Set API keys
```bash
cp .env.example .env
# Add SARVAM_API_KEY (for JD extraction) and LAMMA_API_KEY (AICredits, for resume generation)
# Optionally add GROQ_API_KEY if you want to re-run evaluator.py
```

### 3. Run the pipeline

```bash
# (a) Scrape JDs from Naukri (warning: slow, ~2–4 hours under anti-bot throttling)
python master_scraper.py

# (b) Extract structured fields and normalize
python extractor.py
python normalize_structured_jds.py structured_jds.jsonl --output-suffix _normalize.jsonl

# (c) Build company + college pools
python pool.py

# (d) Generate candidate profiles + score them, assemble training corpora
python assemble_corpus.py --limit 0 --seed 42

# (e) Stratified lean-input prep + build prompts + generate resume text
python generate_llm_inputs.py --limit 0
python build_resume_prompts.py --input llm_resume_input.jsonl --out prompts_full_v2.jsonl
python call_llm_resumes.py --input prompts_full_v2.jsonl --out llm_resume_output_full_v2.jsonl

# (f) Retry failures, then quality-fix bad resumes
python retry_failed_resumes.py
python rerun_bad_resumes.py

# (g) Join resumes back into training files, then produce SFT-ready file
python join_resumes_to_training.py
python prepare_sft_data.py

# (h) Fine-tune (upload sft_pointwise.jsonl to Lightning.ai / Colab / Kaggle)
# Open qwen_finetune_lightning.ipynb and run.
```

---

## Honest limitations

- **735 JDs × ~15 tech role types** — limited role diversity. Model may struggle on under-represented roles.
- **Synthetic resumes are more structured than real ones** — real resumes have inconsistent formatting, missing fields, different orderings. Model may not generalize to noisy real-world inputs.
- **Project descriptions overlap** — multiple resumes share similar `business_problem` templates (e.g. "Customer Churn Analysis"). Reduces lexical diversity.
- **Tech stack currency** — generated resumes don't explicitly prioritize 2024–2025 technologies (RAG, vLLM, LangChain). Not date-aware.
- **Score rationale is templated** — follows consistent phrasing. Model learns the template more than the reasoning.
- **Scorer is deterministic** — the 7-axis score is a weighted formula over the *structured* profile, not a learned label. The model is essentially learning to invert resume text → structured profile → score.
- **XGBoost LTR ranker is planned but not implemented** — pointwise and pairwise corpora are already built; only the training + eval loop is missing.

---

## Tech stack

- **Scraping**: Playwright, playwright-stealth, BeautifulSoup
- **LLM APIs**: Sarvam (JD extraction), AICredits / Llama-3.1-8B (resume generation), Groq / Llama-4-Scout-17B (extraction eval)
- **Training**: Unsloth, TRL SFTTrainer, PEFT, bitsandbytes, Transformers
- **Base model**: Qwen2.5-3B-Instruct (4-bit quantized)
- **Compute**: Single T4 GPU on Lightning.ai
- **Misc**: Faker (contact generation), requests, python-dotenv style env loader

---

## License

MIT
