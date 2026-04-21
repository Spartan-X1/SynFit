import json
import math
import random
import re
from collections import Counter
from datetime import date

from skill_normalizer import normalize_skill_list


DIAL_PRESETS = {
    "balanced": {
        "skill_coverage": 0.50,
        "experience_years": 0.52,
        "domain_relevance": 0.60,
        "deployment_skills": 0.36,
        "education_tier": 0.54,
        "notice_period": 0.50,   # ~50% 30 Days, ~35% Immediate, ~15% 90 Days
    },
    "full_match": {
        "skill_coverage": 0.95,
        "experience_years": 0.88,
        "domain_relevance": 0.90,
        "deployment_skills": 0.78,
        "education_tier": 0.72,
        "notice_period": 0.67,   # mostly Immediate, some 30 Days
    },
    "hard_negative": {
        "skill_coverage": 0.38,
        "experience_years": 0.15,
        "domain_relevance": 0.55,
        "deployment_skills": 0.12,
        "education_tier": 0.48,
        "notice_period": 0.38,   # mostly 30 Days, some 90 Days
    },
    "mismatch": {
        "skill_coverage": 0.05,
        "experience_years": 0.26,
        "domain_relevance": 0.18,
        "deployment_skills": 0.08,
        "education_tier": 0.44,
        "notice_period": 0.25,   # mostly 90 Days
    },
}


CONTENT_SCORE_WEIGHTS = {
    "skill_coverage": 0.38,
    "experience_fit": 0.24,
    "role_alignment": 0.14,
    "domain_alignment": 0.09,
    "education_fit": 0.05,
    "deployment_alignment": 0.05,
    "management_alignment": 0.05,
}


MATCH_RANGE_BY_PRESET = {
    "full_match": (0.78, 1.0),
    "balanced": (0.32, 0.52),
    "hard_negative": (0.20, 0.38),
    "mismatch": (0.0, 0.08),
}


ROLE_KEYWORD_HINTS = {
    "full_stack_developer": ["full stack", "fullstack", "mern", "mean", "web developer"],
    "frontend_developer": ["front end", "frontend", "ui developer", "react developer"],
    "backend_developer": ["backend", "api developer", "services engineer"],
    "java_spring_boot_developer": ["spring", "java microservices", "spring boot", "java backend"],
    "machine_learning_engineer": ["machine learning", "ml engineer", "ai engineer", "artificial intelligence"],
    "data_scientist": ["data scientist", "decision scientist", "applied scientist"],
    "data_engineer": ["data engineer", "etl engineer", "analytics engineer"],
    "devops_engineer": ["devops", "site reliability", "sre", "platform engineer"],
    "cloud_architect": ["cloud architect", "solution architect", "platform architect"],
    "android_developer": ["android"],
    "ios_developer": ["ios", "swift"],
    "qa_automation_engineer": ["qa", "automation", "test engineer", "sdet"],
    "cyber_security_analyst": ["security", "soc", "cyber"],
    "product_manager": ["product manager", "product owner", "technical product"],
    "database_administrator": ["database administrator", "dba", "sql dba", "postgresql administrator"],
    "enterprise_platform_developer": ["duck creek", "guidewire", "billing platform", "insurance platform"],
}


ROLE_MISMATCH_MAP = {
    "full_stack_developer": [
        "machine_learning_engineer",
        "cyber_security_analyst",
        "database_administrator",
        "android_developer",
        "product_manager",
    ],
    "frontend_developer": [
        "data_engineer",
        "devops_engineer",
        "cyber_security_analyst",
        "database_administrator",
    ],
    "backend_developer": [
        "frontend_developer",
        "android_developer",
        "data_scientist",
        "product_manager",
    ],
    "java_spring_boot_developer": [
        "frontend_developer",
        "android_developer",
        "data_scientist",
        "product_manager",
    ],
    "machine_learning_engineer": [
        "frontend_developer",
        "java_spring_boot_developer",
        "qa_automation_engineer",
        "database_administrator",
    ],
    "data_scientist": [
        "frontend_developer",
        "devops_engineer",
        "android_developer",
        "enterprise_platform_developer",
    ],
    "data_engineer": [
        "frontend_developer",
        "android_developer",
        "ios_developer",
        "cyber_security_analyst",
    ],
    "devops_engineer": [
        "frontend_developer",
        "data_scientist",
        "product_manager",
        "ios_developer",
    ],
    "cloud_architect": [
        "frontend_developer",
        "android_developer",
        "qa_automation_engineer",
        "data_scientist",
    ],
    "android_developer": [
        "devops_engineer",
        "database_administrator",
        "data_engineer",
        "enterprise_platform_developer",
    ],
    "ios_developer": [
        "devops_engineer",
        "database_administrator",
        "data_engineer",
        "enterprise_platform_developer",
    ],
    "qa_automation_engineer": [
        "machine_learning_engineer",
        "data_engineer",
        "product_manager",
        "cloud_architect",
    ],
    "cyber_security_analyst": [
        "frontend_developer",
        "ios_developer",
        "product_manager",
        "data_scientist",
    ],
    "product_manager": [
        "devops_engineer",
        "database_administrator",
        "android_developer",
        "cyber_security_analyst",
    ],
    "database_administrator": [
        "frontend_developer",
        "product_manager",
        "android_developer",
        "qa_automation_engineer",
    ],
    "enterprise_platform_developer": [
        "frontend_developer",
        "machine_learning_engineer",
        "devops_engineer",
        "product_manager",
    ],
}


ROLE_ADJACENCY = {
    "full_stack_developer": {"frontend_developer", "backend_developer", "java_spring_boot_developer"},
    "frontend_developer": {"full_stack_developer"},
    "backend_developer": {"full_stack_developer", "java_spring_boot_developer"},
    "java_spring_boot_developer": {"backend_developer", "full_stack_developer"},
    "machine_learning_engineer": {"data_scientist", "data_engineer"},
    "data_scientist": {"machine_learning_engineer", "data_engineer"},
    "data_engineer": {"machine_learning_engineer", "data_scientist"},
    "devops_engineer": {"cloud_architect"},
    "cloud_architect": {"devops_engineer"},
    "android_developer": {"ios_developer"},
    "ios_developer": {"android_developer"},
}


DOMAIN_CONTEXT_BANK = {
    "financial": {
        "labels": ["Payments", "Lending", "Risk", "Collections"],
        "audiences": ["operations analysts", "relationship managers", "customers", "risk teams"],
    },
    "bank": {
        "labels": ["Core Banking", "Payments", "Customer Service", "Treasury"],
        "audiences": ["business users", "customers", "branch teams", "support teams"],
    },
    "retail": {
        "labels": ["Catalog", "Checkout", "Fulfillment", "Merchandising"],
        "audiences": ["shoppers", "merchandising teams", "operations teams", "support agents"],
    },
    "internet": {
        "labels": ["Growth", "Platform", "Engagement", "Content"],
        "audiences": ["end users", "ops teams", "business stakeholders", "platform teams"],
    },
    "artificial intelligence": {
        "labels": ["Inference", "Model Ops", "Recommendation", "Training"],
        "audiences": ["analysts", "data teams", "platform teams", "internal users"],
    },
    "ai/ml": {
        "labels": ["Inference", "Recommendation", "Forecasting", "Feature Platform"],
        "audiences": ["data scientists", "analysts", "internal users", "product teams"],
    },
    "education": {
        "labels": ["Learning", "Classroom", "Assessment", "Content"],
        "audiences": ["students", "teachers", "ops teams", "admins"],
    },
    "health": {
        "labels": ["Patient Operations", "Clinical Data", "Claims", "Provider Portal"],
        "audiences": ["care teams", "operations teams", "admins", "providers"],
    },
    "insurance": {
        "labels": ["Policy", "Claims", "Billing", "Underwriting"],
        "audiences": ["claims analysts", "underwriters", "customers", "ops teams"],
    },
    "consult": {
        "labels": ["Enterprise Platform", "Internal Ops", "Client Delivery", "Audit"],
        "audiences": ["client teams", "delivery managers", "operations teams", "support teams"],
    },
}


PROJECT_CONSTRAINT_BANK = {
    "entry": [
        "with a strong focus on basic reliability and maintainability",
        "while keeping onboarding and handoff simple for internal teams",
        "with clear workflows and lightweight operational checks",
    ],
    "mid": [
        "while reducing latency under growing production traffic",
        "with stronger observability and release safety in place",
        "while balancing delivery speed with service reliability",
    ],
    "senior": [
        "while meeting strict reliability and scale expectations across teams",
        "with clear ownership boundaries, observability, and safe rollout controls",
        "while supporting cross-team integrations and high-volume production usage",
    ],
}


FOCUS_AREA_BANK = {
    "stage": {
        "entry": [
            "implemented scoped features with strong code quality and handoff discipline",
            "contributed to iterative delivery and production issue resolution",
            "picked up well-defined tasks independently and shipped with minimal rework",
            "ramped quickly on the codebase and resolved high-priority bugs under guidance",
            "wrote unit tests and participated actively in code reviews",
            "assisted senior engineers in breaking down complex requirements into executable tickets",
            "learned production workflows through on-call rotations and incident postmortems",
            "delivered assigned modules on schedule while maintaining test coverage standards",
            "fixed flaky tests and improved CI pass rates for the team's main pipeline",
            "shadowed production deployments and contributed to runbook documentation",
            "built internal scripts that automated repetitive developer chores",
            "responded to triage tickets and reduced open bug backlog within first quarter",
            "set up local dev environments and standardised onboarding documentation",
            "paired with senior engineers to ship three production features in first six months",
            "wrote integration tests that caught two regressions before they reached staging",
        ],
        "mid": [
            "owned end-to-end delivery for medium-complexity product surfaces",
            "balanced feature delivery with performance and reliability improvements",
            "identified and resolved recurring technical debt that was slowing team velocity",
            "partnered with product and design to scope and refine feature requirements",
            "drove improvements to the team's deployment confidence through better test coverage",
            "led technical execution on a multi-sprint initiative with minimal escalations",
            "diagnosed and fixed performance bottlenecks that affected user-facing latency",
            "acted as the primary reviewer for junior engineers and shaped coding standards",
            "took ownership of a critical service and reduced its p99 latency by 40%",
            "migrated a legacy module to a newer stack with zero downtime",
            "championed observability improvements that reduced mean time to detect by half",
            "negotiated scope with product to protect engineering quality without delaying launch",
            "introduced contract testing between services to catch integration failures early",
            "reduced on-call burden by converting reactive alerts into proactive monitors",
            "drove a refactor that cut the service's memory footprint by 35%",
            "coordinated a phased rollout across three regions with automated canary checks",
        ],
        "senior": [
            "drove architectural decisions and technical tradeoffs across modules",
            "improved engineering execution through stronger observability and release practices",
            "defined long-term technical direction in close alignment with product leadership",
            "reduced systemic failure points by redesigning core infrastructure components",
            "led cross-team initiatives that required coordinating dependencies and timelines",
            "built internal tooling that measurably improved developer productivity",
            "established incident review processes that shortened mean time to resolution",
            "influenced hiring decisions and shaped engineering culture through structured mentorship",
            "decomposed a monolith into independent services enabling parallel team execution",
            "introduced SLO-based alerting that shifted the team from reactive to proactive ops",
            "authored the platform migration RFC adopted by all product teams",
            "reduced infrastructure spend by 28% through right-sizing and reserved capacity planning",
            "built a self-service deployment portal that eliminated ops bottlenecks for ten teams",
            "led a post-incident review programme that surfaced three systemic risks",
            "defined coding standards and automated enforcement through custom linter rules",
            "coordinated a multi-quarter programme across five teams with weekly steering reviews",
        ],
    },
    "management": [
        "mentored engineers and coordinated delivery across cross-functional stakeholders",
        "helped break down roadmap work into technically coherent execution plans",
        "guided review quality and execution standards for complex deliverables",
        "ran sprint planning and retrospectives to improve team delivery predictability",
        "worked with product managers to translate business goals into engineering priorities",
        "created growth paths for direct reports and supported their performance reviews",
        "conducted bi-weekly one-on-ones and drove individual development plans for six reports",
        "reduced team attrition by restructuring on-call rotation and workload distribution",
        "partnered with recruiting to close three senior engineer roles within one quarter",
        "introduced a technical roadmap process that aligned engineering priorities with OKRs",
        "built a culture of blameless postmortems that improved incident learning loops",
    ],
    "deployment": [
        "worked closely with CI/CD, observability, and deployment workflows in production",
        "improved rollout safety through automation, monitoring, and operational guardrails",
        "reduced deployment failures by introducing staged rollouts and automated smoke tests",
        "owned on-call responsibilities and drove resolution of production incidents",
        "consolidated pipeline configurations to cut average build time from 18 to 7 minutes",
        "introduced feature flags enabling safe same-day production releases",
        "built GitOps-based delivery workflows that removed manual approval gates",
        "set up distributed tracing that cut root cause analysis time during incidents",
    ],
}


PROJECT_NAME_SUFFIXES = [
    "Platform",
    "Console",
    "Engine",
    "Portal",
    "Workflow",
    "Service",
    "Hub",
    "Suite",
    "Framework",
    "Toolkit",
    "Gateway",
    "Pipeline",
    "Layer",
    "Module",
    "System",
    "Accelerator",
    "Fabric",
    "Runtime",
    "Agent",
    "Mesh",
]


GENERIC_DOMAIN_FALLBACKS = [
    "Internet",
    "Financial Services",
    "Retail",
    "E-Learning / EdTech",
    "Healthcare Technology",
    "IT Services & Consulting",
]


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jds(filepath):
    """Loads valid JDs, skipping broken or skipped lines."""
    jds = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if record.get("_status") != "skipped":
                record["core_technical_skills"] = normalize_skill_list(record.get("core_technical_skills", []))
                jds.append(record)
    return jds


def load_skill_taxonomy(filepath):
    """Loads role profiles from the JSONL skill graph."""
    taxonomy = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            role = data.pop("role")
            taxonomy[role] = data
    return taxonomy


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def compact_text(value):
    return normalize_text(value).replace(" ", "")


def unique_in_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, value))


def jitter(value, spread=0.06):
    return clamp(random.gauss(value, spread))


def sample_subset_in_order(items, count):
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    selected_indexes = sorted(random.sample(range(len(items)), count))
    return [items[index] for index in selected_indexes]


def build_dials(profile_preset="balanced", dial_overrides=None):
    base = DIAL_PRESETS.get(profile_preset, DIAL_PRESETS["balanced"])
    dials = {key: jitter(value) for key, value in base.items()}

    if dial_overrides:
        for key, value in dial_overrides.items():
            if key in dials:
                dials[key] = clamp(float(value))

    return dials


def sample_truncated_gaussian(mean, deviation, lower, upper, attempts=12):
    for _ in range(attempts):
        value = random.gauss(mean, deviation)
        if lower <= value <= upper:
            return value
    return clamp(mean, lower, upper)


def skill_matches(left, right):
    left_compact = compact_text(left)
    right_compact = compact_text(right)
    if not left_compact or not right_compact:
        return False
    if left_compact == right_compact:
        return True
    if len(left_compact) >= 4 and left_compact in right_compact:
        return True
    if len(right_compact) >= 4 and right_compact in left_compact:
        return True
    return False


def infer_title_seniority(job_title):
    title = normalize_text(job_title)
    if any(token in title for token in ["principal", "staff", "architect", "manager", "head of"]):
        return "principal"
    if any(token in title for token in ["lead", "senior", "sr "]):
        return "senior"
    if any(token in title for token in ["ii", "iii"]):
        return "mid"
    return "unspecified"


def infer_target_experience(jd):
    explicit_years = jd.get("min_experience_years", 0) or 0
    if explicit_years > 0:
        return explicit_years, "explicit"

    title_band = infer_title_seniority(jd.get("job_title", ""))
    inferred_years = title_band_default_years(title_band)
    if inferred_years > 0:
        return inferred_years, "title_inferred"
    return 0, "unknown"


def title_band_default_years(title_band):
    return {
        "principal": 8,
        "senior": 5,
        "mid": 3,
        "unspecified": 0,
    }.get(title_band, 0)


def score_role_against_title(role, profile, normalized_title):
    score = 0
    aliases = [role.replace("_", " ")] + profile.get("aliases", [])
    for alias in aliases:
        normalized_alias = normalize_text(alias)
        if not normalized_alias:
            continue
        if normalized_alias in normalized_title or normalized_title in normalized_alias:
            score = max(score, len(normalized_alias.split()) * 4)
        else:
            overlap = len(set(normalized_alias.split()) & set(normalized_title.split()))
            score = max(score, overlap * 2)

    for keyword in ROLE_KEYWORD_HINTS.get(role, []):
        if normalize_text(keyword) in normalized_title:
            score += 4

    return score


def map_jd_to_role(job_title, taxonomy, is_management_role=False):
    """Maps a JD title to the most relevant role profile without overriding technical leads to PM."""
    normalized_title = normalize_text(job_title)
    best_role = "generic_engineering"
    best_score = -1

    for role, profile in taxonomy.items():
        score = score_role_against_title(role, profile, normalized_title)
        if score > best_score:
            best_role = role
            best_score = score

    if best_score <= 0:
        title = normalized_title
        if any(token in title for token in ["duck creek", "guidewire", "billing developer"]):
            return "enterprise_platform_developer"
        if "database administrator" in title or "dba" in title:
            return "database_administrator"
        if "full stack" in title or "fullstack" in title:
            return "full_stack_developer"
        if "front end" in title or "frontend" in title:
            return "frontend_developer"
        if "backend" in title:
            return "backend_developer"
        if "spring" in title:
            return "java_spring_boot_developer"
        if "cloud architect" in title:
            return "cloud_architect"
        if "devops" in title or "sre" in title:
            return "devops_engineer"
        if "machine learning" in title or title.startswith("ai ") or "artificial intelligence" in title:
            return "machine_learning_engineer"
        if "data engineer" in title:
            return "data_engineer"
        if "data scientist" in title:
            return "data_scientist"
        if "android" in title:
            return "android_developer"
        if "ios" in title:
            return "ios_developer"
        if "qa" in title or "test" in title:
            return "qa_automation_engineer"
        if "security" in title:
            return "cyber_security_analyst"

    if "product" in normalized_title and is_management_role:
        return "product_manager"

    return best_role


def infer_stage(experience_years):
    if experience_years >= 7:
        return "senior"
    if experience_years >= 2.5:
        return "mid"
    return "entry"


def get_role_skill_catalog(role_profile):
    skills = []
    for path in role_profile.get("skill_paths", []):
        skills.extend(path)
    skills.extend(role_profile.get("supporting_skills", []))
    for template in role_profile.get("project_templates", []):
        skills.extend(template.get("skills", []))
    return unique_in_order(skills)


def best_catalog_match(skill, catalog):
    for candidate in catalog:
        if skill_matches(skill, candidate):
            return candidate
    return None


def path_score(path, matched_skills):
    score = 0
    for skill in path:
        if any(skill_matches(skill, matched) for matched in matched_skills):
            score += 3
    return score


def is_delivery_path(path):
    delivery_keywords = {
        "docker",
        "kubernetes",
        "terraform",
        "jenkins",
        "ci cd",
        "aws",
        "azure",
        "gcp",
        "gitops",
        "monitoring",
    }
    return any(any(token in normalize_text(skill) for token in delivery_keywords) for skill in path)


def choose_candidate_role_key(target_role_key, taxonomy, profile_preset):
    if profile_preset != "mismatch":
        return target_role_key

    choices = ROLE_MISMATCH_MAP.get(target_role_key)
    if choices:
        valid_choices = [role for role in choices if role in taxonomy]
        if valid_choices:
            return random.choice(valid_choices)

    fallback = [role for role in taxonomy if role not in {target_role_key, "generic_engineering"}]
    return random.choice(fallback) if fallback else target_role_key


def build_non_jd_pool(filler_profile, jd_skills):
    """Collect all skills from filler profile that do NOT fuzzy-match any JD skill."""
    raw = []
    for path in filler_profile.get("skill_paths", []):
        raw.extend(path)
    for template in filler_profile.get("project_templates", []):
        raw.extend(template.get("skills", []))
    raw.extend(filler_profile.get("supporting_skills", []))
    return unique_in_order([
        skill for skill in raw
        if not any(skill_matches(skill, jd_skill) for jd_skill in jd_skills)
    ])


def choose_skill_bundle(
    jd_skills,
    target_role_profile,
    candidate_role_profile,
    skill_dial,
    deployment_dial,
    strength_dial,
    profile_preset="balanced",
    filler_role_profile=None,
):
    # filler_role_profile is used as the non-JD skill source for hard_negative,
    # giving the candidate a laterally-adjacent skill set rather than a thinner
    # version of the target role.
    filler_profile = (
        filler_role_profile
        if (profile_preset == "hard_negative" and filler_role_profile is not None)
        else candidate_role_profile
    )

    # --- Loop 1: JD-matched skills ---
    # Find which JD skills exist in the candidate role's catalog, then keep
    # exactly as many as the preset allows (MATCH_RANGE_BY_PRESET).
    catalog = get_role_skill_catalog(candidate_role_profile)
    all_matched_skills = []
    unmatched_jd_skills = []
    for jd_skill in jd_skills:
        matched = best_catalog_match(jd_skill, catalog)
        if matched:
            all_matched_skills.append(matched)
        else:
            unmatched_jd_skills.append(jd_skill)
    all_matched_skills = unique_in_order(all_matched_skills)
    jd_count = determine_match_count(len(all_matched_skills), skill_dial, profile_preset)
    matched_skills = sample_subset_in_order(all_matched_skills, jd_count)

    # --- Loop 2: non-JD filler skills ---
    # Explicitly built from skills that have zero fuzzy-overlap with any JD skill.
    # For hard_negative this comes from the adjacent filler role; for other presets
    # it comes from candidate_role_profile paths/templates (same as before).
    max_skills = 6
    if strength_dial > 0.45:
        max_skills = 8
    if strength_dial > 0.75:
        max_skills = 10
    if profile_preset == "mismatch":
        max_skills = min(max_skills, 7)

    non_jd_slots = max(0, max_skills - len(matched_skills))
    non_jd_pool = build_non_jd_pool(filler_profile, jd_skills)
    # Shuffle so selection isn't always the same path order
    non_jd_sample = non_jd_pool[:]
    random.shuffle(non_jd_sample)
    non_jd_skills = non_jd_sample[:non_jd_slots]

    # --- Assemble & track path/template metadata (used downstream) ---
    # selected_paths is still needed by generate_experience_history for
    # primary_tech_stack composition, so we derive it from the filler profile.
    scored_paths = sorted(
        filler_profile.get("skill_paths", []),
        key=lambda path: (path_score(path, matched_skills), len(path), random.random()),
        reverse=True,
    )
    selected_paths = select_scored_paths(
        scored_paths,
        matched_skills,
        strength_dial,
        deployment_dial,
        profile_preset,
    )

    final_skills = unique_in_order(matched_skills + non_jd_skills)

    # Pad with unmatched JD skills if pool ran dry (rare, but possible for small roles)
    if len(final_skills) < 4 and profile_preset != "mismatch":
        final_skills = unique_in_order(final_skills + unmatched_jd_skills)

    return final_skills[:max_skills], matched_skills, selected_paths, all_matched_skills


def choose_domain(jd, target_role_profile, candidate_role_profile, domain_relevance, profile_preset="balanced"):
    jd_domain = jd.get("domain", "").strip()

    # Probability of staying in-domain is driven by the domain_relevance dial.
    # full_match: dial ~0.90 → ~95% chance of jd_domain (stays tight).
    # balanced:   dial ~0.60 → ~65% chance.
    # hard_negative: dial ~0.55 → ~40% chance (frequent domain drift).
    # mismatch:   dial ~0.18 → ~10% chance (almost always out-of-domain).
    in_domain_prob = {
        "full_match":    0.50 + domain_relevance * 0.50,   # [0.50, 1.00]
        "balanced":      0.20 + domain_relevance * 0.70,   # [0.20, 0.90]
        "hard_negative": 0.10 + domain_relevance * 0.55,   # [0.10, 0.645]
        "mismatch":      domain_relevance * 0.25,           # [0.00, 0.25]
    }.get(profile_preset, domain_relevance)

    if jd_domain and random.random() < in_domain_prob:
        return jd_domain

    candidate_preferences = candidate_role_profile.get("jd_domain_preferences", []) or GENERIC_DOMAIN_FALLBACKS
    filtered = [domain for domain in candidate_preferences if normalize_text(domain) != normalize_text(jd_domain)]
    if filtered:
        return random.choice(filtered)

    fallback = [d for d in GENERIC_DOMAIN_FALLBACKS if normalize_text(d) != normalize_text(jd_domain)]
    return random.choice(fallback) if fallback else random.choice(GENERIC_DOMAIN_FALLBACKS)


def company_domain_matches(company_domain, preferred_domains):
    normalized_company_domain = normalize_text(company_domain)
    for preferred_domain in preferred_domains:
        normalized_preference = normalize_text(preferred_domain)
        if not normalized_preference:
            continue
        if normalized_preference in normalized_company_domain or normalized_company_domain in normalized_preference:
            return True
        if len(set(normalized_preference.split()) & set(normalized_company_domain.split())) >= 2:
            return True
    return False


def derive_actual_experience(required_years, experience_fit, profile_preset="balanced"):
    preset_shift = {
        "full_match": 0.18,
        "balanced": -0.08,
        "hard_negative": -0.52,
        "mismatch": -0.30,
    }.get(profile_preset, 0.0)

    if required_years > 0:
        mean = required_years * (1.0 + preset_shift + ((experience_fit - 0.5) * 0.22))
        spread = max(0.8, required_years * 0.16)
        if profile_preset == "hard_negative":
            lower = max(0.8, required_years * 0.3)
        elif profile_preset == "mismatch":
            lower = max(0.8, required_years * 0.45)
        else:
            lower = max(0.8, required_years * 0.6)
        upper = max(required_years + 5.0, required_years * 1.7)
        return round(sample_truncated_gaussian(mean, spread, lower, upper), 1)

    # Fresher/unspecified role: target ranges that align with the scorer's
    # overqualification curve (<=2yr = ideal, 2-5yr = penalty, >5yr = heavy penalty).
    # full_match must land in the ideal band; hard_negative must land in the penalty band.
    fresher_config = {
        "full_match":    (1.0, 0.5, 0.5, 2.0),   # mean, spread, lower, upper
        "balanced":      (2.5, 0.7, 1.0, 4.0),
        "hard_negative": (4.5, 0.8, 3.0, 7.0),
        "mismatch":      (2.0, 1.0, 0.5, 5.0),
    }.get(profile_preset, (2.0, 0.9, 0.5, 6.0))
    mean, spread, lower, upper = fresher_config
    return round(sample_truncated_gaussian(mean, spread, lower, upper), 1)


def _usage_weights(candidates, company_usage):
    """
    Return a weight list for `candidates` that penalises frequently-used
    companies so the selection spreads evenly across the pool.

    Weight formula: 1 / (1 + log(1 + usage_count))
    A company used 0 times → weight 1.0
    Used 5 times          → weight ~0.59
    Used 20 times         → weight ~0.40
    Used 100 times        → weight ~0.27
    """
    weights = []
    for c in candidates:
        count = company_usage.get(c.get("name", ""), 0)
        weights.append(1.0 / (1.0 + math.log1p(count)))
    return weights


def choose_company(companies, role_profile, overall_strength, stage,
                   target_domain="", excluded_names=None, company_usage=None):
    excluded_names = set(excluded_names or [])
    company_usage  = company_usage or {}

    if stage == "senior":
        tier_sequence = ["tier_1", "tier_2", "tier_3"]
    elif stage == "mid":
        tier_sequence = ["tier_2", "tier_3", "tier_4"]
    else:
        tier_sequence = ["tier_3", "tier_4", "tier_5"]

    if overall_strength < 0.35:
        tier_sequence = list(reversed(tier_sequence))

    preferred_domains = unique_in_order(role_profile.get("preferred_company_domains", []) + [target_domain])

    for tier in tier_sequence:
        pool = companies.get(tier, [])
        domain_matches = [
            company
            for company in pool
            if company.get("name") not in excluded_names
            and company_domain_matches(company.get("domain", ""), preferred_domains)
        ]
        if domain_matches:
            # If the domain-matched pool is too small, blend in tier-mates so
            # a single company doesn't win by default.
            if len(domain_matches) < 3:
                non_domain = [
                    c for c in pool
                    if c.get("name") not in excluded_names and c not in domain_matches
                ]
                domain_matches = domain_matches + random.sample(non_domain, min(4, len(non_domain)))
            weights = _usage_weights(domain_matches, company_usage)
            return random.choices(domain_matches, weights=weights, k=1)[0]

    for tier in tier_sequence:
        fallback_pool = [c for c in companies.get(tier, []) if c.get("name") not in excluded_names]
        if fallback_pool:
            weights = _usage_weights(fallback_pool, company_usage)
            return random.choices(fallback_pool, weights=weights, k=1)[0]

    for tier, pool in companies.items():
        fallback_pool = [c for c in pool if c.get("name") not in excluded_names]
        if fallback_pool:
            weights = _usage_weights(fallback_pool, company_usage)
            return random.choices(fallback_pool, weights=weights, k=1)[0]

    return {"name": "Confidential Company", "domain": "Technology", "estimated_ctc": "₹10 LPA"}


def education_pool_from_requirement(education_requirement):
    requirement = normalize_text(education_requirement)
    if any(token in requirement for token in ["iit", "nit", "iiit", "bits", "master", "m tech", "mca", "mba", "ms", "m sc"]):
        return "tier_1"
    if any(token in requirement for token in ["bachelor", "b tech", "b e", "degree", "graduate"]):
        return "tier_2"
    return None


def choose_degree(role_profile, education_requirement="", stage="mid"):
    degrees = role_profile.get("education_options") or ["B.Tech Computer Science", "B.E Information Technology", "MCA"]
    requirement = normalize_text(education_requirement)
    advanced_degrees = [
        degree for degree in degrees
        if any(token in normalize_text(degree) for token in ["m.tech", "mca", "mba", "m.s", "ms", "m.sc"])
    ]

    if advanced_degrees and any(
        token in requirement for token in ["master", "m tech", "mca", "mba", "ms", "m sc", "postgraduate"]
    ):
        if "preferred" not in requirement or stage == "senior" or random.random() < 0.65:
            return random.choice(advanced_degrees)
    return random.choice(degrees)


def choose_role_title(jd_title, role_profile, stage, honor_jd_title=True):
    stage_titles = role_profile.get("stage_titles", {})
    titles = stage_titles.get(stage) or [jd_title or "Software Engineer"]
    if honor_jd_title and stage == "senior" and "senior" in normalize_text(jd_title):
        return jd_title
    return random.choice(titles)


def previous_stage_for(stage):
    if stage == "senior":
        return "mid"
    if stage == "mid":
        return "entry"
    return "entry"


def determine_match_count(total_matches, skill_dial, profile_preset):
    if total_matches <= 0:
        return 0

    lower_ratio, upper_ratio = MATCH_RANGE_BY_PRESET.get(profile_preset, MATCH_RANGE_BY_PRESET["balanced"])

    if profile_preset == "mismatch":
        mismatch_cap = min(total_matches, 1 if skill_dial < 0.1 else 2)
        return random.randint(0, mismatch_cap)

    lower_count = max(1, round(total_matches * lower_ratio))
    upper_count = max(lower_count, round(total_matches * upper_ratio))
    dial_ratio = lower_ratio + ((upper_ratio - lower_ratio) * clamp(skill_dial))
    target_count = max(lower_count, round(total_matches * dial_ratio))

    range_floor = max(lower_count, target_count - 1)
    range_ceiling = min(total_matches, max(upper_count, target_count + 1))
    return random.randint(range_floor, max(range_floor, range_ceiling))


def select_scored_paths(scored_paths, matched_skills, strength_dial, deployment_dial, profile_preset):
    if not scored_paths:
        return []

    seed_pool = scored_paths[: min(4, len(scored_paths))]
    positive_seed_pool = [path for path in seed_pool if path_score(path, matched_skills) > 0]
    selected_paths = [random.choice(positive_seed_pool or seed_pool)]

    desired_paths = 1
    if profile_preset in {"balanced", "full_match"} or strength_dial > 0.58:
        desired_paths = 2
    if (profile_preset == "full_match" and strength_dial > 0.7) or (
        profile_preset == "balanced" and strength_dial > 0.82
    ):
        desired_paths = 3

    delivery_path = None
    if deployment_dial > 0.58:
        for path in scored_paths:
            if path not in selected_paths and is_delivery_path(path):
                delivery_path = path
                break

    remaining = [path for path in scored_paths if path not in selected_paths]
    while len(selected_paths) < desired_paths and remaining:
        if delivery_path and delivery_path in remaining and random.random() < 0.8:
            selected_paths.append(delivery_path)
            remaining.remove(delivery_path)
            delivery_path = None
            continue

        choice_pool = remaining[: min(3, len(remaining))]
        chosen = random.choice(choice_pool)
        selected_paths.append(chosen)
        remaining = [path for path in remaining if path != chosen]

    return selected_paths


def build_primary_stack(selected_paths, final_skills, limit=5):
    weighted_stack = []
    for path in selected_paths:
        weighted_stack.extend(path)
    weighted_stack.extend(final_skills)
    return unique_in_order(weighted_stack)[:limit] or ["Python"]


def pick_domain_context(domain):
    normalized_domain = normalize_text(domain)
    for fragment, config in DOMAIN_CONTEXT_BANK.items():
        if fragment in normalized_domain:
            return random.choice(config["labels"]), random.choice(config["audiences"])
    return random.choice(["Platform", "Operations", "Workflow", "Customer"]), random.choice(
        ["internal teams", "business users", "customers", "operations teams"]
    )


def compose_focus_areas(
    role_profile,
    matched_skills,
    domain,
    stage,
    is_management_role,
    deployment_dial,
    primary_stack,
    excluded_themes=None,
):
    themes = role_profile.get("experience_themes", ["delivered product features end to end"])
    # Earlier stages draw from the foundational end of the theme list to signal
    # a narrower, more execution-focused scope — this creates a visible career arc
    # when the previous job is at entry/mid and the current job is at senior.
    if stage == "entry" and len(themes) > 4:
        theme_pool = themes[: max(5, len(themes) // 2)]
    elif stage == "mid" and len(themes) > 6:
        theme_pool = themes[: max(7, (len(themes) * 3) // 4)]
    else:
        theme_pool = themes
    # Exclude themes already used in a more-recent experience entry so consecutive
    # entries don't repeat the same focus area sentences.
    if excluded_themes:
        filtered_pool = [t for t in theme_pool if t not in excluded_themes]
        theme_pool = filtered_pool if len(filtered_pool) >= 2 else theme_pool
    selected = []
    selected.extend(random.sample(theme_pool, min(2, len(theme_pool))))
    selected.append(random.choice(FOCUS_AREA_BANK["stage"][stage]))

    if matched_skills:
        sampled_skill = random.choice(matched_skills[: min(3, len(matched_skills))])
        selected.append(f"applied {sampled_skill} to production-facing workflows in the {domain} domain")
        # Second stack-specific line when there are enough distinct matched skills
        remaining = [s for s in matched_skills if s != sampled_skill]
        if remaining and len(matched_skills) >= 2:
            second_skill = random.choice(remaining[: min(2, len(remaining))])
            stack_phrases = [
                f"leveraged {second_skill} for performance-critical components across the service layer",
                f"used {second_skill} to improve reliability and observability of core systems",
                f"integrated {second_skill} into the team's standard delivery and testing workflow",
                f"built reusable modules with {second_skill} adopted across the wider engineering team",
            ]
            selected.append(random.choice(stack_phrases))
    elif primary_stack:
        selected.append(f"worked deeply with {random.choice(primary_stack)} across production delivery workflows")

    if deployment_dial > 0.55:
        selected.append(random.choice(FOCUS_AREA_BANK["deployment"]))

    if is_management_role:
        selected.append(random.choice(FOCUS_AREA_BANK["management"]))

    return unique_in_order(selected)


def generate_experience_history(
    jd,
    candidate_role_profile,
    companies,
    target_domain,
    overall_strength,
    stage,
    actual_years,
    final_skills,
    matched_skills,
    selected_paths,
    deployment_dial,
    is_management_role,
    honor_jd_title=True,
    company_usage=None,
):
    _usage = company_usage if company_usage is not None else {}
    current_company = choose_company(
        companies,
        candidate_role_profile,
        overall_strength,
        stage,
        target_domain=target_domain,
        company_usage=_usage,
    )
    if company_usage is not None:
        company_usage[current_company.get("name", "")] = company_usage.get(current_company.get("name", ""), 0) + 1
    current_stack = build_primary_stack(selected_paths, final_skills, limit=5)
    experience_history = []

    if actual_years >= 5.0 and stage in {"mid", "senior"}:
        recent_years = round(max(1.5, min(actual_years - 1.0, actual_years * random.uniform(0.45, 0.62))), 1)
        previous_years = round(max(1.0, actual_years - recent_years), 1)
        prev_stage = previous_stage_for(stage)
        previous_company = choose_company(
            companies,
            candidate_role_profile,
            overall_strength * 0.88,
            prev_stage,
            target_domain=target_domain,
            excluded_names={current_company.get("name")},
            company_usage=_usage,
        )
        if company_usage is not None:
            company_usage[previous_company.get("name", "")] = company_usage.get(previous_company.get("name", ""), 0) + 1
        previous_stack = build_primary_stack(selected_paths[:1], final_skills[:5], limit=4)

        current_focus_areas = compose_focus_areas(
            candidate_role_profile,
            matched_skills,
            target_domain,
            stage,
            is_management_role,
            deployment_dial,
            current_stack,
        )[:3]
        experience_history.append(
            {
                "company": current_company["name"],
                "company_domain": current_company.get("domain", "Technology"),
                "role_domain": target_domain,
                "duration_years": recent_years,
                "role_title": choose_role_title(
                    jd.get("job_title", "Software Engineer"),
                    candidate_role_profile,
                    stage,
                    honor_jd_title=honor_jd_title,
                ),
                "primary_tech_stack": current_stack,
                "focus_areas": current_focus_areas,
            }
        )
        experience_history.append(
            {
                "company": previous_company["name"],
                "company_domain": previous_company.get("domain", "Technology"),
                "role_domain": target_domain,
                "duration_years": previous_years,
                "role_title": choose_role_title(
                    jd.get("job_title", "Software Engineer"),
                    candidate_role_profile,
                    prev_stage,
                    honor_jd_title=False,
                ),
                "primary_tech_stack": previous_stack,
                "focus_areas": compose_focus_areas(
                    candidate_role_profile,
                    matched_skills[:2],
                    target_domain,
                    prev_stage,
                    is_management_role and prev_stage != "entry",
                    max(0.0, deployment_dial - 0.1),
                    previous_stack,
                    excluded_themes=current_focus_areas,
                )[:2],
            }
        )
        current_ctc = current_company.get("estimated_ctc", "₹10 LPA")
    else:
        experience_history.append(
            {
                "company": current_company["name"],
                "company_domain": current_company.get("domain", "Technology"),
                "role_domain": target_domain,
                "duration_years": max(1.0, actual_years),
                "role_title": choose_role_title(
                    jd.get("job_title", "Software Engineer"),
                    candidate_role_profile,
                    stage,
                    honor_jd_title=honor_jd_title,
                ),
                "primary_tech_stack": current_stack,
                "focus_areas": compose_focus_areas(
                    candidate_role_profile,
                    matched_skills,
                    target_domain,
                    stage,
                    is_management_role,
                    deployment_dial,
                    current_stack,
                )[:3],
            }
        )
        current_ctc = current_company.get("estimated_ctc", "₹10 LPA")

    return experience_history, current_ctc


def select_project_templates(role_profile, selected_skills):
    templates = role_profile.get("project_templates", [])
    return sorted(
        templates,
        key=lambda template: (
            sum(1 for skill in template.get("skills", []) if any(skill_matches(skill, selected) for selected in selected_skills)),
            random.random(),
        ),
        reverse=True,
    )


def generate_quantitative_metrics(stage, architecture_type, domain):
    nd = normalize_text(domain)

    # Domain-specific metric pools keyed by (stage)
    domain_pools = {}

    if any(x in nd for x in ["fintech", "financial", "payment", "banking", "trading", "insurance",
                              "bank", "bfsi", "nbfc", "lending", "wealth", "capital", "fund",
                              "property casualty", "p c insurance"]):
        domain_pools = {
            "entry":  [
                {"transactions_processed": "50k+/day", "error_rate": "<0.5%"},
                {"payment_flows_tested": "30+", "test_coverage": "82%"},
                {"reconciliation_jobs": "daily", "accuracy": "99.8%"},
            ],
            "mid":    [
                {"transactions": "500k+/day", "fraud_detection_accuracy": "98.2%"},
                {"payment_success_rate": "99.4%", "latency": "<120ms"},
                {"settlement_cycles": "4/day", "failure_rate": "<0.3%"},
                {"api_calls": "3M+/month", "p99_latency": "<200ms"},
            ],
            "senior": [
                {"tpv": "₹200Cr+/month", "uptime": "99.99%"},
                {"peak_tps": "12k+", "latency_p99": "<80ms"},
                {"fraud_loss_reduction": "34%", "model_precision": "97.6%"},
                {"compliance_audits_passed": "4/year", "sla": "99.95%"},
            ],
        }
    elif any(x in nd for x in ["mobile", "android", "ios", "app",
                               "consumer app", "mobile app development"]):
        domain_pools = {
            "entry":  [
                {"screens_shipped": "8", "crash_rate": "<1%"},
                {"unit_tests_added": "120+", "test_pass_rate": "97%"},
                {"ui_components_built": "15", "accessibility_score": "AA"},
            ],
            "mid":    [
                {"dau": "200k+", "crash_rate": "<0.3%"},
                {"app_store_rating": "4.6", "monthly_active_users": "800k+"},
                {"cold_start_time": "<1.2s", "retention_d30": "41%"},
                {"api_response_time": "<300ms", "offline_sync_success": "99.1%"},
            ],
            "senior": [
                {"dau": "2M+", "crash_free_sessions": "99.7%"},
                {"release_cadence": "bi-weekly", "force_update_rate": "<2%"},
                {"sdk_integrations": "6", "p90_launch_time": "<800ms"},
                {"platform_coverage": "Android + iOS", "shared_code": "72%"},
            ],
        }
    elif any(x in nd for x in ["data", "analytics", "ml", "machine learning", "ai", "science",
                               "kpo", "research", "data science", "big data", "bi ", "business intelligence",
                               "artificial intelligence"]):
        domain_pools = {
            "entry":  [
                {"datasets_cleaned": "12", "pipeline_accuracy": "94%"},
                {"notebooks_productionised": "4", "data_quality_score": "91%"},
                {"etl_jobs": "8/day", "failure_rate": "<2%"},
            ],
            "mid":    [
                {"model_accuracy": "91.4%", "inference_latency": "<50ms"},
                {"data_processed": "500GB+/day", "pipeline_sla": "99.5%"},
                {"features_served": "120+", "feature_store_freshness": "<5min"},
                {"experiment_velocity": "12 A/B tests/quarter", "lift": "8%"},
            ],
            "senior": [
                {"data_processed": "terabytes/day", "model_retraining": "weekly"},
                {"platform_users": "40+ data scientists", "job_success_rate": "99.2%"},
                {"cost_per_prediction": "reduced 60%", "p99_latency": "<30ms"},
                {"data_contracts_enforced": "35+", "schema_drift_alerts": "real-time"},
            ],
        }
    elif any(x in nd for x in ["devops", "infra", "cloud", "platform", "sre", "reliability",
                               "cloud computing", "it services", "it infrastructure",
                               "managed services", "consulting", "internet", "telecom",
                               "software product", "saas", "paas", "it & information"]):
        domain_pools = {
            "entry":  [
                {"pipelines_automated": "6", "avg_build_time": "8 min"},
                {"infra_modules_written": "10+", "drift_incidents": "0"},
                {"runbooks_documented": "15", "on_call_incidents_resolved": "8"},
            ],
            "mid":    [
                {"deployment_frequency": "12+/week", "change_failure_rate": "<3%"},
                {"mttr": "<22min", "alert_noise_reduction": "55%"},
                {"infra_cost_saved": "₹18L/year", "resource_utilisation": "74%"},
                {"pipeline_uptime": "99.8%", "avg_deploy_time": "<6min"},
            ],
            "senior": [
                {"clusters_managed": "14", "uptime": "99.97%"},
                {"infra_spend_reduction": "28%", "engineer_toil_saved": "6hr/week"},
                {"slo_compliance": "99.95%", "incident_count_reduction": "40%"},
                {"platform_teams_served": "9", "self_service_adoption": "80%"},
            ],
        }
    elif any(x in nd for x in ["security", "cyber", "appsec", "pentest",
                               "information security", "cybersecurity"]):
        domain_pools = {
            "entry":  [
                {"vulnerabilities_triaged": "40+/quarter", "false_positive_rate": "<8%"},
                {"security_tests_added": "60+", "owasp_coverage": "top 10"},
                {"dependency_audits": "monthly", "critical_cves_patched": "100%"},
            ],
            "mid":    [
                {"critical_vulns_remediated": "25+/year", "mean_time_to_patch": "<72hr"},
                {"penetration_tests_conducted": "8", "findings_resolved": "94%"},
                {"sast_scan_coverage": "100%", "secrets_leaked": "0"},
                {"compliance_controls": "SOC2 Type II", "audit_findings": "0 critical"},
            ],
            "senior": [
                {"security_incidents": "0 breaches", "threat_models_reviewed": "30+"},
                {"red_team_exercises": "2/year", "detection_rate": "98.5%"},
                {"security_programme_coverage": "12 teams", "risk_score_reduction": "45%"},
                {"zero_trust_rollout": "90% complete", "lateral_movement_blocked": "100%"},
            ],
        }
    elif any(x in nd for x in ["frontend", "ui", "web", "design",
                               "advertising", "marketing", "media", "ott",
                               "e learning", "edtech", "education"]):
        domain_pools = {
            "entry":  [
                {"components_built": "20+", "accessibility_score": "WCAG AA"},
                {"lighthouse_score": "88+", "cross_browser_issues": "0"},
                {"storybook_stories": "35+", "visual_regression_tests": "passed"},
            ],
            "mid":    [
                {"page_load_time": "<1.8s", "lcp": "<2.1s"},
                {"bundle_size_reduction": "38%", "ttfb": "<400ms"},
                {"component_reuse_rate": "65%", "design_parity": "95%"},
                {"a11y_violations": "0 critical", "monthly_active_users": "300k+"},
            ],
            "senior": [
                {"design_system_components": "80+", "adoption": "6 product teams"},
                {"core_web_vitals": "all green", "monthly_sessions": "5M+"},
                {"build_time_reduction": "52%", "shared_component_usage": "78%"},
                {"performance_budget_compliance": "100%", "regression_rate": "<1%"},
            ],
        }
    elif any(x in nd for x in ["game", "gaming", "entertainment", "sports", "fantasy"]):
        domain_pools = {
            "entry":  [
                {"game_features_shipped": "5", "bug_escape_rate": "<2%"},
                {"unit_tests": "80+", "frame_rate_target": "60fps maintained"},
            ],
            "mid":    [
                {"dau": "150k+", "session_length": "22min avg"},
                {"in_app_purchase_conversion": "4.2%", "churn_d7": "28%"},
                {"server_tick_rate": "20Hz", "lag_complaints_reduced": "60%"},
            ],
            "senior": [
                {"concurrent_players": "50k+", "uptime": "99.9%"},
                {"live_ops_events": "8/quarter", "revenue_uplift": "18%"},
                {"anti_cheat_detections": "2k+/month", "false_positives": "<0.1%"},
            ],
        }

    elif any(x in nd for x in ["retail", "e commerce", "ecommerce", "marketplace",
                               "d2c", "direct to consumer", "fmcg", "grocery", "food",
                               "consumer goods", "fashion", "apparel"]):
        domain_pools = {
            "entry":  [
                {"sku_catalogue_tested": "500+", "search_result_accuracy": "93%"},
                {"checkout_flows_automated": "8", "cart_abandonment_tracked": True},
                {"integrations_tested": "5", "regression_pass_rate": "98%"},
            ],
            "mid":    [
                {"gmv_supported": "₹50Cr+/month", "checkout_latency": "<800ms"},
                {"catalogue_size": "2M+ SKUs", "search_p99": "<350ms"},
                {"order_success_rate": "99.1%", "return_processing_time": "<24hr"},
                {"conversion_uplift": "12%", "a_b_tests_run": "8/quarter"},
            ],
            "senior": [
                {"peak_orders": "80k+/day", "platform_uptime": "99.95%"},
                {"gmv_uplift": "22%", "recommendation_ctr": "8.4%"},
                {"seller_onboarding_time": "reduced 60%", "catalogue_ingestion": "1M/day"},
                {"logistics_partners_integrated": "12", "sla_breach_rate": "<0.5%"},
            ],
        }
    elif any(x in nd for x in ["health", "pharma", "medical", "life science", "hospital", "clinical",
                               "pharmaceutical", "biotech", "diagnostics"]):
        domain_pools = {
            "entry":  [
                {"test_cases_written": "100+", "hipaa_controls_reviewed": "15"},
                {"data_pipelines_validated": "6", "record_accuracy": "99.2%"},
                {"integrations_built": "4", "hl7_messages_processed": "10k+/day"},
            ],
            "mid":    [
                {"patient_records_managed": "500k+", "data_accuracy": "99.7%"},
                {"ehr_integrations": "6", "api_uptime": "99.9%"},
                {"claims_processed": "200k+/month", "adjudication_accuracy": "98.5%"},
                {"clinical_trials_supported": "3", "data_compliance": "HIPAA + GDPR"},
            ],
            "senior": [
                {"facilities_integrated": "18", "platform_uptime": "99.97%"},
                {"regulatory_audits_passed": "3", "critical_findings": "0"},
                {"data_processed": "10M+ patient events/month", "latency": "<2s"},
                {"cost_per_claim_reduced": "18%", "fraud_detection_precision": "96%"},
            ],
        }
    elif any(x in nd for x in ["recruitment", "staffing", "hr tech", "human resource",
                               "bpm", "bpo", "outsourc"]):
        domain_pools = {
            "entry":  [
                {"workflows_automated": "5", "manual_effort_saved": "8hr/week"},
                {"reports_built": "12", "data_refresh_frequency": "daily"},
                {"integrations_tested": "6", "uptime": "99.5%"},
            ],
            "mid":    [
                {"candidates_processed": "10k+/month", "matching_accuracy": "87%"},
                {"workflows_automated": "18", "processing_time_reduced": "45%"},
                {"clients_served": "40+", "sla_compliance": "98%"},
                {"data_quality_score": "94%", "pipeline_latency": "<5min"},
            ],
            "senior": [
                {"platform_users": "500+", "uptime": "99.9%"},
                {"cost_per_hire_reduced": "32%", "time_to_fill_reduced": "25%"},
                {"automations_deployed": "30+", "annual_savings": "₹1.2Cr"},
                {"integrations_live": "20+", "data_accuracy": "98.5%"},
            ],
        }
    elif any(x in nd for x in ["manufactur", "industrial", "automotive", "auto component",
                               "semiconductor", "hardware", "electronic", "equipment"]):
        domain_pools = {
            "entry":  [
                {"test_scripts_written": "60+", "defect_detection_rate": "92%"},
                {"data_pipelines_built": "4", "sensor_feeds_integrated": "8"},
                {"dashboards_delivered": "5", "data_refresh": "real-time"},
            ],
            "mid":    [
                {"oee_improvement": "8%", "downtime_alerts": "real-time"},
                {"sensor_streams": "200+", "anomaly_detection_accuracy": "94%"},
                {"supply_chain_nodes": "12", "forecast_accuracy": "89%"},
                {"defect_reduction": "22%", "scrap_rate_improvement": "15%"},
            ],
            "senior": [
                {"plants_connected": "8", "data_volume": "5TB+/month"},
                {"predictive_maintenance_savings": "₹2.5Cr/year", "downtime_reduced": "35%"},
                {"digital_twin_coverage": "4 production lines", "oee": "82%"},
                {"erp_integrations": "SAP + Oracle", "data_latency": "<10s"},
            ],
        }

    # Generic fallback used when no domain matches
    generic_pools = {
        "entry": [
            {"users": "internal teams", "status": "pilot"},
            {"records": "50k+/day", "status": "beta"},
            {"tickets_resolved": "40+/sprint", "reopen_rate": "<5%"},
            {"integrations_built": "4", "uptime": "99.5%"},
            {"test_cases_written": "90+", "coverage": "78%"},
        ],
        "mid": [
            {"users": "10k+", "uptime": "99.9%"},
            {"requests": "2M+/month", "latency": "<250ms"},
            {"jobs_processed": "500k+/day", "sla": "99.5%"},
            {"active_customers": "8k+", "support_tickets_reduced": "30%"},
            {"api_uptime": "99.8%", "avg_response_time": "<180ms"},
            {"monthly_events": "15M+", "processing_lag": "<3s"},
            {"integrations_live": "12", "error_rate": "<0.4%"},
        ],
        "senior": [
            {"users": "1M+", "uptime": "99.95%"},
            {"requests": "50M+/month", "latency": "<180ms"},
            {"data_processed": "terabytes/day", "sla": "99.95%"},
            {"teams_served": "8+", "platform_uptime": "99.97%"},
            {"peak_rps": "25k+", "p99_latency": "<120ms"},
            {"cost_optimisation": "₹30L+/year", "engineer_productivity": "+25%"},
            {"global_regions": "4", "failover_tested": "quarterly"},
        ],
    }

    pool = domain_pools if domain_pools else generic_pools
    selected = dict(random.choice(pool[stage]))

    if "event" in normalize_text(architecture_type):
        selected.setdefault("events", "high-volume async traffic")
    if "serverless" in normalize_text(architecture_type):
        selected.setdefault("autoscaling", "burst-ready")
    return selected


def build_project_name(template_name, domain):
    context_label, _ = pick_domain_context(domain)
    base = template_name.title()

    # Words already present in the base name (normalised) — used to avoid repetition
    base_words = set(normalize_text(base).split())
    context_words = set(normalize_text(context_label).split())
    label_overlaps_base = bool(base_words & context_words)

    # Suffix must not duplicate the context label or the last word of base
    base_last_word = normalize_text(base.split()[-1]) if base.split() else ""
    suffix_pool = [
        s for s in PROJECT_NAME_SUFFIXES
        if normalize_text(s) != normalize_text(context_label)
        and normalize_text(s) != base_last_word
    ]
    suffix = random.choice(suffix_pool) if suffix_pool else "Hub"

    variants = [
        base,                                                          # e.g. "Customer Churn Analysis"
        f"{base} {suffix}",                                            # e.g. "Customer Churn Analysis Pipeline"
        f"{base} for {context_label}" if not label_overlaps_base       # skip if "Customer" already in base
            else f"{base} {suffix}",
        f"{context_label} {suffix}" if not label_overlaps_base         # e.g. "Retail Hub"
            else f"{base} Engine",
    ]
    return random.choice(unique_in_order(variants))


def build_business_problem(template, domain, stage):
    base = template.get("business_problem", "Built a production-facing solution aligned with the target role.")
    context_label, audience = pick_domain_context(domain)
    # Avoid "Workflow workflows" / "Operations operations" — suffix templates
    # no longer append "workflows" or "operations" after the context_label.
    domain_suffix = {
        "entry":  f"Targeted at {audience} in the {context_label} space.",
        "mid":    f"Served {audience} within the {context_label} domain under production load.",
        "senior": f"Deployed for {audience} supporting {context_label} at scale.",
    }[stage]
    return f"{base} {domain_suffix}"


_EXTRA_IMPACT_POOL = {
    "entry": [
        "improved adoption by internal users",
        "reduced manual handoffs",
        "improved test coverage",
        "reduced bug escape rate",
        "reduced onboarding friction",
        "improved code review turnaround",
        "improved local development experience",
        "reduced CI flakiness",
        "improved runbook accuracy",
        "reduced duplicate support tickets",
    ],
    "mid": [
        "reduced issue resolution time",
        "improved release confidence",
        "improved system observability",
        "reduced p99 latency",
        "reduced on-call alert volume",
        "improved API reliability",
        "reduced time to first meaningful paint",
        "improved data freshness",
        "reduced engineer toil",
        "improved feature adoption",
        "reduced rollback frequency",
        "improved test suite speed",
        "reduced technical debt backlog",
        "improved deployment frequency",
        "reduced unplanned downtime",
    ],
    "senior": [
        "improved cross-team delivery predictability",
        "reduced operational risk at scale",
        "improved platform scalability",
        "reduced infrastructure cost",
        "improved engineering org velocity",
        "reduced critical incident frequency",
        "improved SLO compliance",
        "reduced mean time to recovery",
        "improved developer experience",
        "reduced system complexity",
        "improved capacity planning accuracy",
        "reduced security exposure surface",
        "improved data quality at scale",
        "reduced time to market for new features",
        "improved team retention through better tooling",
    ],
}


def build_impact_signals(template, stage, used_impacts=None):
    base_outcomes = template.get("outcomes", ["improved platform reliability", "reduced delivery time"])
    extra_pool = _EXTRA_IMPACT_POOL.get(stage, _EXTRA_IMPACT_POOL["mid"])
    # Shuffle the extra pool so we don't always pick the first two
    extra_sample = random.sample(extra_pool, min(4, len(extra_pool)))
    candidates = unique_in_order(base_outcomes + extra_sample)
    if used_impacts:
        candidates = [c for c in candidates if c not in used_impacts]
    if not candidates:
        candidates = extra_pool
    return random.sample(candidates, min(2, len(candidates)))


def generate_structured_projects(
    role_profile,
    selected_skills,
    matched_skills,
    domain,
    stage,
    deployment_dial,
    is_management_role,
    experience_history=None,
):
    project_count = 3 if stage == "senior" else 2
    ranked_templates = select_project_templates(role_profile, selected_skills)
    chosen_templates = ranked_templates[:project_count] or role_profile.get("project_templates", [])[:project_count]
    recent_stack = experience_history[0]["primary_tech_stack"] if experience_history else selected_skills

    projects = []
    used_tech: set = set()
    used_impacts: set = set()
    for template in chosen_templates:
        # Loop 1: pick template-matched skills not already used in a prior project
        technologies = []
        for skill in template.get("skills", []):
            if any(skill_matches(skill, selected) for selected in recent_stack):
                if not any(skill_matches(skill, seen) for seen in used_tech):
                    technologies.append(skill)

        # Loop 2: fill remaining slots from recent_stack then selected_skills,
        # strictly excluding anything already committed to an earlier project
        for pool in (recent_stack, selected_skills):
            for candidate in pool:
                if len(unique_in_order(technologies)) >= 4:
                    break
                if not any(skill_matches(candidate, existing) for existing in technologies):
                    if not any(skill_matches(candidate, seen) for seen in used_tech):
                        technologies.append(candidate)
            if len(unique_in_order(technologies)) >= 4:
                break

        deduped_tech = unique_in_order(technologies)[:6]

        # Fallback: if all skills were consumed by prior projects, pull directly from
        # the template's own skill list (ignoring used_tech) so no project is empty.
        if not deduped_tech:
            template_skills = unique_in_order(template.get("skills", []))
            deduped_tech = template_skills[:4] or unique_in_order(selected_skills)[:3]

        used_tech.update(deduped_tech)

        architecture_candidates = list(template.get("architecture", ["Microservices"]))
        if deployment_dial > 0.65:
            architecture_candidates.extend(["Event-Driven", "Serverless"])
        architecture_type = random.choice(unique_in_order(architecture_candidates))

        linked_skills = matched_skills[:2] if matched_skills else selected_skills[:2]
        link_reason = "Extends the same stack highlighted in the candidate's experience."
        if linked_skills:
            link_reason = f"Extends the {', '.join(linked_skills)} stack already established in the candidate's experience."
        if is_management_role:
            link_reason += " The scope also reflects coordination and delivery responsibilities."

        impact_signals = build_impact_signals(template, stage, used_impacts=used_impacts)
        used_impacts.update(impact_signals)

        projects.append(
            {
                "project_name": build_project_name(template.get("name", "Production Project"), domain),
                "project_domain": domain,
                "business_problem": build_business_problem(template, domain, stage),
                "architecture_type": architecture_type,
                "technologies_used": deduped_tech,
                "complexity_tier": {"entry": "Internal Tool", "mid": "Production Grade", "senior": "Enterprise Scale"}[stage],
                "impact_signals": impact_signals,
                "quantitative_metrics": generate_quantitative_metrics(stage, architecture_type, domain),
                "resume_fit_reason": link_reason,
            }
        )
    return projects


def collect_candidate_skill_inventory(structured_profile):
    inventory = []
    inventory.extend(structured_profile.get("skills", []))
    for exp in structured_profile.get("experience", []):
        inventory.extend(exp.get("primary_tech_stack", []))
        inventory.extend(exp.get("focus_areas", []))
    for project in structured_profile.get("projects", []):
        inventory.extend(project.get("technologies_used", []))
        inventory.extend(project.get("impact_signals", []))
        inventory.append(project.get("architecture_type", ""))
    return unique_in_order(inventory)


def fuzzy_match_count(required_items, observed_items):
    matched = []
    for required in required_items:
        if any(skill_matches(required, observed) for observed in observed_items):
            matched.append(required)
    return len(unique_in_order(matched)), unique_in_order(matched)


def role_alignment_score(target_role_key, candidate_role_key):
    if target_role_key == candidate_role_key:
        return 1.0
    if candidate_role_key in ROLE_ADJACENCY.get(target_role_key, set()):
        return 0.6
    return 0.2


def domain_alignment_score(jd_domain, structured_profile):
    if not jd_domain:
        return 0.5
    normalized_target = normalize_text(jd_domain)
    observed_domains = [structured_profile.get("experience", [{}])[0].get("role_domain", "")]
    observed_domains.extend(exp.get("company_domain", "") for exp in structured_profile.get("experience", []))
    observed_domains.extend(project.get("project_domain", "") for project in structured_profile.get("projects", []))
    for observed in observed_domains:
        normalized_observed = normalize_text(observed)
        if not normalized_observed:
            continue
        if normalized_target in normalized_observed or normalized_observed in normalized_target:
            return 1.0
        if len(set(normalized_target.split()) & set(normalized_observed.split())) >= 2:
            return 0.8
    return 0.35


def education_alignment_score(education_requirement, education):
    requirement = normalize_text(education_requirement)
    degree_text = normalize_text(education.get("degree", ""))
    institution_text = normalize_text(education.get("institution", ""))

    score = 0.5
    if any(token in requirement for token in ["master", "m tech", "mca", "mba", "ms", "m sc", "postgraduate"]):
        score = 1.0 if any(token in degree_text for token in ["m tech", "mca", "mba", "ms", "m sc"]) else 0.45
    elif any(token in requirement for token in ["bachelor", "graduate", "b tech", "b e", "degree"]):
        score = 1.0 if degree_text else 0.6

    if any(token in requirement for token in ["iit", "nit", "iiit", "bits"]):
        elite_markers = ["iit", "iiit", "nit", "bits", "international institute", "technology"]
        if any(marker in institution_text for marker in elite_markers):
            score = min(1.0, score + 0.1)
        else:
            score = min(score, 0.55)

    return score


def management_alignment_score(jd_requires_management, generated_management_role):
    if not jd_requires_management:
        return 1.0 if not generated_management_role else 0.7
    return 1.0 if generated_management_role else 0.45


def deployment_alignment_score(jd_skills, observed_items):
    deployment_terms = [
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "terraform",
        "ci/cd",
        "jenkins",
        "helm",
        "gitops",
        "monitoring",
    ]
    required_deploy = [
        skill for skill in jd_skills
        if any(token in normalize_text(skill) for token in deployment_terms)
    ]
    if not required_deploy:
        return 0.5
    matched_count, _ = fuzzy_match_count(required_deploy, observed_items)
    return matched_count / max(1, len(required_deploy))


def compute_content_relevance(jd, structured_profile, target_role_key, candidate_role_key, generated_management_role):
    jd_skills = normalize_skill_list(jd.get("core_technical_skills", []))
    inventory = collect_candidate_skill_inventory(structured_profile)
    matched_skill_count, matched_skill_list = fuzzy_match_count(jd_skills, inventory)
    skill_score = matched_skill_count / max(1, len(jd_skills)) if jd_skills else 0.5

    required_years, experience_signal = infer_target_experience(jd)
    total_years = round(sum(exp.get("duration_years", 0.0) for exp in structured_profile.get("experience", [])), 1)
    if required_years > 0:
        experience_score = min(1.0, total_years / required_years)
    else:
        # Fresher/unspecified role: 0–2 yrs = ideal, 3–5 yrs = slight overqualification, >5 yrs = overqualified
        if total_years <= 2.0:
            experience_score = 1.0
        elif total_years <= 5.0:
            experience_score = max(0.5, 1.0 - (total_years - 2.0) * 0.1)
        else:
            experience_score = max(0.3, 0.7 - (total_years - 5.0) * 0.08)

    role_score = role_alignment_score(target_role_key, candidate_role_key)
    domain_score = domain_alignment_score(jd.get("domain", ""), structured_profile)
    education_score = education_alignment_score(
        jd.get("education_requirement", ""),
        structured_profile.get("education", {}),
    )
    management_score = management_alignment_score(jd.get("is_management_role", False), generated_management_role)
    deployment_score = deployment_alignment_score(jd_skills, inventory)

    overall = (
        CONTENT_SCORE_WEIGHTS["skill_coverage"] * skill_score
        + CONTENT_SCORE_WEIGHTS["experience_fit"] * experience_score
        + CONTENT_SCORE_WEIGHTS["role_alignment"] * role_score
        + CONTENT_SCORE_WEIGHTS["domain_alignment"] * domain_score
        + CONTENT_SCORE_WEIGHTS["education_fit"] * education_score
        + CONTENT_SCORE_WEIGHTS["deployment_alignment"] * deployment_score
        + CONTENT_SCORE_WEIGHTS["management_alignment"] * management_score
    )

    rationale_parts = []
    if jd_skills:
        lead_skills = ", ".join(matched_skill_list[:3])
        if lead_skills:
            rationale_parts.append(f"matches {matched_skill_count}/{len(jd_skills)} JD skills including {lead_skills}")
        else:
            rationale_parts.append(f"matches {matched_skill_count}/{len(jd_skills)} JD skills")
    if required_years > 0:
        rationale_parts.append(f"shows {total_years:.1f} years against a {required_years}-year target")
    else:
        if total_years > 5.0:
            rationale_parts.append(f"shows {total_years:.1f} years which may be overqualified for a fresher role")
        elif total_years > 2.0:
            rationale_parts.append(f"shows {total_years:.1f} years, slightly above fresher expectations")
        else:
            rationale_parts.append(f"shows {total_years:.1f} years which fits the fresher role expectation")
    role_phrase = "same role family" if role_score == 1.0 else "adjacent role family" if role_score >= 0.6 else "different role family"
    rationale_parts.append(role_phrase)
    rationale_parts.append(f"domain alignment {domain_score:.2f}")
    if jd.get("is_management_role", False):
        rationale_parts.append("management scope aligned" if management_score >= 0.8 else "limited management evidence")
    rationale = ". ".join(part[0].upper() + part[1:] if part else part for part in rationale_parts) + "."

    return {
        "overall": round(overall, 3),
        "breakdown": {
            "skill_coverage": round(skill_score, 3),
            "experience_fit": round(experience_score, 3),
            "role_alignment": round(role_score, 3),
            "domain_alignment": round(domain_score, 3),
            "education_fit": round(education_score, 3),
            "deployment_alignment": round(deployment_score, 3),
            "management_alignment": round(management_score, 3),
        },
        "weights": CONTENT_SCORE_WEIGHTS,
        "matched_jd_skills": matched_skill_list,
        "effective_required_years": required_years,
        "score_rationale": rationale,
    }


def generate_candidate(jd, companies, colleges, taxonomy, profile_preset="balanced", dial_overrides=None, company_usage=None):
    dials = build_dials(profile_preset, dial_overrides)
    overall_strength = (dials["education_tier"] + dials["skill_coverage"]) / 2

    target_role_key = map_jd_to_role(
        jd.get("job_title", ""),
        taxonomy,
        is_management_role=jd.get("is_management_role", False),
    )
    candidate_role_key = choose_candidate_role_key(target_role_key, taxonomy, profile_preset)

    target_role_profile = taxonomy.get(target_role_key, taxonomy["generic_engineering"])
    candidate_role_profile = taxonomy.get(candidate_role_key, taxonomy["generic_engineering"])

    # For hard_negative: pick one adjacent role's profile as the filler source so the
    # candidate's non-JD skills come from a neighbouring discipline (lateral mover signal)
    # rather than just being fewer skills from the same role pool.
    filler_role_profile = None
    if profile_preset == "hard_negative":
        adjacent_keys = list(ROLE_ADJACENCY.get(target_role_key, set()))
        valid_adjacent = [k for k in adjacent_keys if k in taxonomy and k != target_role_key]
        if valid_adjacent:
            filler_role_key = random.choice(valid_adjacent)
            filler_role_profile = taxonomy[filler_role_key]

    required_years, experience_signal = infer_target_experience(jd)
    actual_years = derive_actual_experience(required_years, dials["experience_years"], profile_preset=profile_preset)
    stage = infer_stage(actual_years)
    title_band = infer_title_seniority(jd.get("job_title", ""))

    inferred_pool = education_pool_from_requirement(jd.get("education_requirement", ""))
    if inferred_pool:
        tier_edu = inferred_pool
    elif dials["education_tier"] > 0.8:
        tier_edu = "tier_1"
    elif dials["education_tier"] > 0.4:
        tier_edu = "tier_2"
    else:
        tier_edu = "tier_3"

    graduation_buffer = 1 if stage in {"mid", "senior"} else 0
    if title_band == "principal":
        graduation_buffer = 2
    education = {
        "institution": random.choice(colleges[tier_edu]),
        "degree": choose_degree(candidate_role_profile, jd.get("education_requirement", ""), stage=stage),
        "graduation_year": int(date.today().year - max(1, round(actual_years + graduation_buffer))),
    }

    final_skills, matched_skills, selected_paths, all_matched_skills = choose_skill_bundle(
        normalize_skill_list(jd.get("core_technical_skills", [])),
        target_role_profile,
        candidate_role_profile,
        dials["skill_coverage"],
        dials["deployment_skills"],
        overall_strength,
        profile_preset=profile_preset,
        filler_role_profile=filler_role_profile,
    )

    target_domain = choose_domain(
        jd,
        target_role_profile,
        candidate_role_profile,
        dials["domain_relevance"],
        profile_preset=profile_preset,
    )

    generated_management_role = jd.get("is_management_role", False) or title_band == "principal"

    experience, current_ctc = generate_experience_history(
        jd,
        candidate_role_profile,
        companies,
        target_domain,
        overall_strength,
        stage,
        actual_years,
        final_skills,
        matched_skills,
        selected_paths,
        dials["deployment_skills"],
        generated_management_role,
        honor_jd_title=(candidate_role_key == target_role_key and profile_preset != "mismatch"),
        company_usage=company_usage,
    )

    notice = "Immediate" if dials["notice_period"] > 0.65 else "30 Days" if dials["notice_period"] > 0.35 else "90 Days"

    structured_profile = {
        "education": education,
        "skills": final_skills,
        "skill_alignment": {
            "available_jd_skill_matches": all_matched_skills,
            "matched_jd_skills": matched_skills,
            "skill_paths_used": selected_paths,
        },
        "experience": experience,
        "projects": generate_structured_projects(
            candidate_role_profile,
            final_skills,
            matched_skills,
            target_domain,
            stage,
            dials["deployment_skills"],
            generated_management_role,
            experience_history=experience,
        ),
        "logistics": {
            "notice_period": notice,
            "current_ctc": current_ctc,
        },
    }

    content_score = compute_content_relevance(
        jd,
        structured_profile,
        target_role_key,
        candidate_role_key,
        generated_management_role,
    )

    return {
        "jd_target": jd.get("job_title"),
        "profile_preset": profile_preset,
        "jd_role_key": target_role_key,
        "candidate_role_key": candidate_role_key,
        "overall_relevance_score": content_score["overall"],
        "score_breakdown": content_score["breakdown"],
        "score_rationale": content_score["score_rationale"],
        "score_metadata": {
            "scoring_version": "content_v2",
            "weights": content_score["weights"],
            "matched_jd_skills": content_score["matched_jd_skills"],
        },
        "effective_required_years": required_years,
        "experience_signal_source": experience_signal,
        "title_seniority_band": title_band,
        "dials": {key: round(value, 3) for key, value in dials.items()},
        "structured_profile": structured_profile,
    }


if __name__ == "__main__":
    print("Loading pools...")
    random.seed(42)
    colleges = load_json("colleges_pool.json")
    companies = load_json("companies_pool.json")
    taxonomy = load_skill_taxonomy("skill_pool_graph_v2.jsonl")
    jds = load_jds("sample_jds.jsonl")

    if not jds:
        print("Error: No valid JDs found in sample_jds.jsonl")
    else:
        test_jd = jds[0]
        print(
            f"\nRunning test on single JD: {test_jd.get('job_title')} "
            f"(Required Experience: {test_jd.get('min_experience_years')} years)"
        )
        sample_candidate = generate_candidate(test_jd, companies, colleges, taxonomy)
        print("\nGeneration Complete. Here is the output:\n")
        print(json.dumps(sample_candidate, indent=4))

        with open("test_single_output.json", "w", encoding="utf-8") as f:
            json.dump(sample_candidate, f, indent=4)
        print("\nSaved to 'test_single_output.json'")
