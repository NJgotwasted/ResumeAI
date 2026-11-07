"""
prepare_dataset.py - Dataset Preparation Pipeline for Resume NER.

Generates 100 high-fidelity, synthetic annotated resumes with word-level BIO tags:
- B-PER, I-PER
- B-SKILL, I-SKILL
- B-DEGREE, I-DEGREE
- B-COMPANY, I-COMPANY
- O

Splits:
- 85 resumes for training & validation:
    - 70 train resumes (data/processed/train.json)
    - 15 validation resumes (data/processed/val.json)
- 15 independent resumes for final benchmark evaluation (data/processed/test.json)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Set fixed seed for 100% reproducible dataset generation
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Exact required BIO label schema
LABEL_LIST: List[str] = [
    "O",
    "B-PER",
    "I-PER",
    "B-SKILL",
    "I-SKILL",
    "B-DEGREE",
    "I-DEGREE",
    "B-COMPANY",
    "I-COMPANY",
]

LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABEL_LIST)}

# Entity pools for synthetic data generation
FIRST_NAMES = [
    "Sarah", "Alex", "David", "Priya", "Elena", "Marcus", "Mei", "Carlos", "Amina", "Liam",
    "Sophia", "Rohan", "Hannah", "Chen", "Gabriel", "Fatima", "Vikram", "Zoe", "Kenji", "Chloe",
    "Arjun", "Lucas", "Maya", "Nikita", "Ananya", "Dmitri", "Jordan", "Tara", "Samuel", "Fatima",
    "Kavita", "Mateo", "Leila", "Devon", "Aarav", "Yuki", "Nadia", "Tariq", "Oscar", "Isabella",
]

LAST_NAMES = [
    "Connor", "Rivera", "Kim", "Sharma", "Rostova", "Vance", "Lin", "Rodriguez", "Diallo", "Smith",
    "Patel", "Chen", "Dubois", "Tanaka", "Silva", "Al-Mansoor", "Kowalski", "Mueller", "Singh", "Novak",
    "Mehta", "Gomez", "Nakamura", "Morales", "Johansson", "O'Connor", "Zhang", "Ibrahim", "Castillo", "Taylor",
    "Gupta", "Santos", "Lombardi", "Kovacs", "Fernandez", "Adler", "Bauer", "Popov", "Rahman", "Mendoza",
]

DEGREES = [
    "Bachelor of Science in Computer Science",
    "Master of Science in Artificial Intelligence",
    "Doctor of Philosophy in Machine Learning",
    "Bachelor of Engineering in Information Technology",
    "Master of Science in Data Science",
    "Bachelor of Science in Electrical Engineering",
    "Master of Business Administration",
    "Bachelor of Science in Software Engineering",
    "Master of Science in Computational Linguistics",
    "Bachelor of Technology in Computer Engineering",
    "Doctor of Philosophy in Computer Science",
    "Master of Science in Cybersecurity",
    "Bachelor of Science in Mathematics and Statistics",
    "Master of Science in Robotics and Automation",
    "Bachelor of Science in Informatics",
    "AWS Certified Solutions Architect",
    "TensorFlow Developer Certificate",
]

COMPANIES = [
    "Google", "Microsoft", "Amazon Web Services", "Apple", "Meta Platforms",
    "NVIDIA", "DeepMind", "Netflix", "Databricks", "Snowflake",
    "Salesforce", "Stripe", "Palantir Technologies", "Uber Technologies", "Airbnb",
    "Cyberdyne Systems", "Skynet Robotics", "Acme Corporation", "Quantum Leap AI", "Cognitive Systems",
    "InnoTech Labs", "Nexus Cloud Systems", "Pinnacle Software", "BioHealth Informatics", "FinTech Dynamics",
    "Cisco Systems", "Oracle Corporation", "Intel Corporation", "IBM Research", "Spotify",
    "Stanford University", "MIT Lincoln Laboratory", "Carnegie Mellon University", "Harvard Medical Lab",
]

SKILLS_POOL = [
    "Python", "PyTorch", "TensorFlow", "HuggingFace Transformers", "BERT",
    "Large Language Models", "Natural Language Processing", "Computer Vision", "Scikit-Learn",
    "Docker", "Kubernetes", "AWS", "Google Cloud Platform", "Microsoft Azure", "Terraform",
    "SQL", "PostgreSQL", "MongoDB", "Redis", "Apache Kafka", "Apache Spark",
    "FastAPI", "Flask", "Django", "React", "TypeScript", "JavaScript", "Node.js",
    "C++", "Java", "Go", "Rust", "Linux", "Git", "CI/CD", "REST APIs", "GraphQL",
    "Pandas", "NumPy", "OpenCV", "Airflow", "Elasticsearch", "Prompt Engineering",
]

JOB_TITLES = [
    "Senior Machine Learning Engineer",
    "Lead AI Research Scientist",
    "Principal Software Architect",
    "Data Scientist",
    "NLP Engineer",
    "Full Stack Developer",
    "Cloud Infrastructure Engineer",
    "Computer Vision Specialist",
    "Data Engineering Lead",
    "Backend Systems Developer",
]

CITIES = [
    "San Francisco, CA", "Seattle, WA", "New York, NY", "Boston, MA", "Austin, TX",
    "London, UK", "Toronto, Canada", "Berlin, Germany", "Bengaluru, India", "Zurich, Switzerland",
]


def tokenize_text(text: str) -> List[str]:
    """Splits text into word and punctuation tokens."""
    # Split on whitespace while keeping punctuation detached
    raw_tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return raw_tokens


def create_annotated_resume(resume_id: int) -> Dict[str, Any]:
    """
    Constructs a coherent synthetic resume and produces aligned word tokens and BIO tags.
    """
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    candidate_name = f"{first_name} {last_name}"
    title = random.choice(JOB_TITLES)
    city = random.choice(CITIES)

    # Pick 2 companies
    selected_companies = random.sample(COMPANIES, 2)
    company1, company2 = selected_companies[0], selected_companies[1]

    # Pick 1-2 degrees
    num_deg = random.choice([1, 2])
    selected_degrees = random.sample(DEGREES, num_deg)

    # Pick 6-10 skills
    num_skills = random.randint(6, 10)
    selected_skills = random.sample(SKILLS_POOL, num_skills)

    # Email & Phone (PII elements to be tagged O so NER model focuses on PER, SKILL, DEGREE, COMPANY)
    email = f"{first_name.lower()}.{last_name.lower()}@domain.org"
    phone = f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

    # Build sentences with known entity spans
    # Sentence 1: Header
    # e.g., "Sarah Connor , Senior Machine Learning Engineer , San Francisco, CA ."
    segments: List[Tuple[str, str]] = []

    # Segment tuple: (text_fragment, entity_type)
    # entity_type in ["PER", "SKILL", "DEGREE", "COMPANY", "O"]

    segments.append((candidate_name, "PER"))
    segments.append((f" , {title} in {city} . Contact : {email} | {phone} .", "O"))

    # Sentence 2: Summary
    skill_sample_sum = random.sample(selected_skills, 2)
    segments.append((f"Summary : Passionate engineer with extensive background in", "O"))
    segments.append((skill_sample_sum[0], "SKILL"))
    segments.append(("and", "O"))
    segments.append((skill_sample_sum[1], "SKILL"))
    segments.append((f"building robust distributed architectures .", "O"))

    # Sentence 3: Experience 1
    segments.append((f"Experience : Currently employed as {title} at", "O"))
    segments.append((company1, "COMPANY"))
    skill_exp1 = random.sample(selected_skills, 2)
    segments.append((f"delivering high - impact solutions using", "O"))
    segments.append((skill_exp1[0], "SKILL"))
    segments.append(("and", "O"))
    segments.append((skill_exp1[1], "SKILL"))
    segments.append((".", "O"))

    # Sentence 4: Experience 2
    segments.append((f"Previously worked at", "O"))
    segments.append((company2, "COMPANY"))
    skill_exp2 = random.sample(selected_skills, 2)
    segments.append((f"specializing in", "O"))
    segments.append((skill_exp2[0], "SKILL"))
    segments.append(("with", "O"))
    segments.append((skill_exp2[1], "SKILL"))
    segments.append(("for production systems .", "O"))

    # Sentence 5: Education
    segments.append((f"Education : Holds a", "O"))
    segments.append((selected_degrees[0], "DEGREE"))
    if len(selected_degrees) > 1:
        segments.append(("and a", "O"))
        segments.append((selected_degrees[1], "DEGREE"))
    segments.append((".", "O"))

    # Sentence 6: Skills Section
    remaining_skills = [s for s in selected_skills if s not in (skill_sample_sum + skill_exp1 + skill_exp2)]
    if not remaining_skills:
        remaining_skills = random.sample(SKILLS_POOL, 3)

    segments.append((f"Core competencies and technical skills include", "O"))
    for idx, sk in enumerate(remaining_skills):
        segments.append((sk, "SKILL"))
        if idx < len(remaining_skills) - 1:
            segments.append((",", "O"))
        else:
            segments.append((".", "O"))

    # Tokenize and create word-level BIO tags
    tokens: List[str] = []
    ner_tags: List[int] = []
    tag_strings: List[str] = []

    for fragment, ent_type in segments:
        frag_tokens = tokenize_text(fragment)
        if not frag_tokens:
            continue

        if ent_type == "O":
            for tok in frag_tokens:
                tokens.append(tok)
                ner_tags.append(LABEL2ID["O"])
                tag_strings.append("O")
        else:
            # First token is B-TYPE, subsequent are I-TYPE
            b_tag = f"B-{ent_type}"
            i_tag = f"I-{ent_type}"

            tokens.append(frag_tokens[0])
            ner_tags.append(LABEL2ID[b_tag])
            tag_strings.append(b_tag)

            for tok in frag_tokens[1:]:
                tokens.append(tok)
                ner_tags.append(LABEL2ID[i_tag])
                tag_strings.append(i_tag)

    full_text = " ".join(tokens)

    return {
        "id": f"resume_{resume_id:03d}",
        "candidate_name": candidate_name,
        "text": full_text,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "tag_strings": tag_strings,
        "ground_truth_entities": {
            "PERSON": [candidate_name],
            "SKILL": list(set(selected_skills)),
            "DEGREE": selected_degrees,
            "COMPANY": selected_companies,
        },
    }


def generate_and_save_dataset():
    """Generates the 100 labeled resumes and saves train/val/test splits."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating 100 synthetic annotated resumes...")
    all_resumes = [create_annotated_resume(i) for i in range(100)]

    # Split: 85 train+val (70 train, 15 val), 15 test
    train_set = all_resumes[0:70]
    val_set = all_resumes[70:85]
    test_set = all_resumes[85:100]

    # Save splits
    with open(OUTPUT_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_set, f, indent=2)

    with open(OUTPUT_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2)

    with open(OUTPUT_DIR / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2)

    # Save schema mapping
    labels_meta = {
        "label_list": LABEL_LIST,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "num_labels": len(LABEL_LIST),
    }
    with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_meta, f, indent=2)

    # Summary statistics
    def count_entities(data):
        counts = {"B-PER": 0, "B-SKILL": 0, "B-DEGREE": 0, "B-COMPANY": 0, "TOTAL_TOKENS": 0}
        for r in data:
            counts["TOTAL_TOKENS"] += len(r["tokens"])
            for t in r["tag_strings"]:
                if t in counts:
                    counts[t] += 1
        return counts

    summary = {
        "total_resumes": len(all_resumes),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
        "train_stats": count_entities(train_set),
        "val_stats": count_entities(val_set),
        "test_stats": count_entities(test_set),
    }
    with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset generated successfully in: {OUTPUT_DIR}")
    print(f"  • Training:   {len(train_set)} resumes")
    print(f"  • Validation: {len(val_set)} resumes")
    print(f"  • Test:       {len(test_set)} resumes")
    print(f"  • Total labels: {len(LABEL_LIST)} {LABEL_LIST}")


if __name__ == "__main__":
    generate_and_save_dataset()

