import re
from typing import Any, Dict, Iterable, List


EXACT_MULTI_MAP = {
    "javascript/typescript": ["JavaScript", "TypeScript"],
    "ecs/eks": ["ECS", "EKS"],
    "alb/nlb": ["ALB", "NLB"],
    ".net (c#, asp.net)": [".NET", "C#", "ASP.NET"],
    "awsazure devops": ["AWS", "Azure DevOps"],
    "awsazure devops/ado": ["AWS", "Azure DevOps"],
}


EXACT_CANONICAL_MAP = {
    "java": "Java",
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "react": "React",
    "react js": "React",
    "reactjs": "React",
    "react.js": "React",
    "node.js": "Node.js",
    "node js": "Node.js",
    "nodejs": "Node.js",
    "spring boot": "Spring Boot",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "git": "Git",
    "nextjs": "Next.js",
    "next js": "Next.js",
    "swift ui": "SwiftUI",
    "swiftui": "SwiftUI",
    "objective c": "Objective-C",
    "objective-c": "Objective-C",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "pyspark": "PySpark",
    "hugging face": "Hugging Face",
    "scikit learn": "scikit-learn",
    "scikitlearn": "scikit-learn",
    "fast api": "FastAPI",
    "restful api": "REST APIs",
    "restful apis": "REST APIs",
    "rest api development": "REST APIs",
    "rest api integration": "REST APIs",
    "rest api": "REST APIs",
    "ci/cd": "CI/CD",
    "ci cd": "CI/CD",
    "cicd": "CI/CD",
    "ci/cd pipelines": "CI/CD",
    "ci cd pipelines": "CI/CD",
    "shell": "Shell Scripting",
    "shell scripting": "Shell Scripting",
    "unix shell scripting": "Shell Scripting",
    "google cloud platform": "GCP",
    "google cloud services": "GCP",
    "microsoft azure": "Azure",
    "azure cloud": "Azure",
    "azure cloud services": "Azure",
    "amazon web services": "AWS",
    "aws services": "AWS",
    "aws cloud": "AWS",
    "aws cloud services": "AWS",
    "helm charts": "Helm",
    "helm chart": "Helm",
    "power bi": "Power BI",
    "qlikview": "QlikView",
    "mlflow": "MLflow",
    "ml flow": "MLflow",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "aws code pipeline": "AWS CodePipeline",
    "github actions": "GitHub Actions",
    "github copilot": "GitHub Copilot",
    "aws opensearch": "Amazon OpenSearch",
    "aws open search": "Amazon OpenSearch",
    "azure devops/ado": "Azure DevOps",
    "oracle database administration (dba)": "Oracle Database",
    "oracle database administration dba": "Oracle Database",
    "oracle (9i, 10g, 11g, 12c, 19c)": "Oracle Database",
    "ios sdk": "iOS SDK",
    "postgres": "PostgreSQL",
    "chatgpt": "OpenAI APIs",
    "vector databases": "Vector Databases",
}


GENERIC_EXACT = {
    "ai",
    "ml",
    "analytics",
    "frameworks",
    "programming",
    "project management",
    "cloud technologies",
    "data pipelines",
    "enterprise applications",
    "machine learning algorithms",
    "classification/regression techniques",
    "business analytics",
    "data engineering",
    "cloud architectures",
    "application development",
    "cloud platform",
    "data structures and algorithms",
}


GENERIC_CONTAINS = {
    "enterprise applications",
    "cloud technologies",
    "machine learning algorithms",
    "project management",
    "application development frameworks",
    "transaction management frameworks",
    "data access frameworks",
    "data quality frameworks",
    "modern test frameworks",
    "public, private and hybrid cloud technologies",
}


GENERIC_SUFFIXES = (
    " frameworks",
    " methodologies",
)


WHITELIST_ANALYTICS = {
    "azure synapse analytics",
    "adobe analytics",
}


def clean_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_key(value: Any) -> str:
    text = clean_whitespace(value).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def titleize_words(text: str) -> str:
    parts = []
    for token in clean_whitespace(text).split():
        if token.isupper() and len(token) <= 5:
            parts.append(token)
        else:
            parts.append(token.capitalize())
    return " ".join(parts)


def is_generic_skill(skill: str) -> bool:
    key = normalize_key(skill)
    if not key:
        return True
    if key in WHITELIST_ANALYTICS:
        return False
    if key in GENERIC_EXACT:
        return True
    if any(key.endswith(suffix) for suffix in GENERIC_SUFFIXES):
        return True
    if any(fragment in key for fragment in GENERIC_CONTAINS):
        return True
    if key in {"architecture", "leadership", "communication", "problem solving"}:
        return True
    return False


def split_compound_skill(skill: str) -> List[str]:
    key = normalize_key(skill)
    if key in EXACT_MULTI_MAP:
        return EXACT_MULTI_MAP[key]

    if "/" in skill and normalize_key(skill) not in {"ci/cd"}:
        parts = [part.strip() for part in skill.split("/") if part.strip()]
        if 1 < len(parts) <= 3:
            return parts

    return [skill]


def canonicalize_skill(skill: str) -> str:
    cleaned = clean_whitespace(skill)
    if not cleaned:
        return ""

    key = normalize_key(cleaned)
    if key in EXACT_CANONICAL_MAP:
        return EXACT_CANONICAL_MAP[key]

    if re.fullmatch(r"aws\s*\(.*\)", key):
        return "AWS"
    if re.fullmatch(r"azure\s*\(.*\)", key):
        return "Azure"
    if re.fullmatch(r"gcp\s*\(.*\)", key):
        return "GCP"
    if key.startswith("oracle ") and "database" in key:
        return "Oracle Database"
    if key == "rest":
        return "REST APIs"

    if key in {"aws", "gcp", "sql", "nlp", "dbt", "dvc", "iam", "api", "apis"}:
        return cleaned.upper() if key != "apis" else "APIs"

    if key in {"bash", "docker", "kubernetes", "terraform", "grafana", "prometheus", "ansible", "jenkins"}:
        return titleize_words(cleaned)

    return cleaned


def normalize_skill_list(value: Any) -> List[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.split(r"[,;\n\r|]+", value)
    else:
        raw_items = []

    normalized: List[str] = []
    seen = set()

    for item in raw_items:
        for candidate in split_compound_skill(clean_whitespace(item)):
            skill = canonicalize_skill(candidate)
            if not skill or is_generic_skill(skill):
                continue

            dedupe_key = normalize_key(skill)
            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            normalized.append(skill)

    return normalized


def normalize_record_skills(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    normalized["core_technical_skills"] = normalize_skill_list(record.get("core_technical_skills", []))
    return normalized


def normalize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_record_skills(record) for record in records]
