import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

INPUT_FILE = "dog_cat_behavior_dataset_optimized.json"
OUTPUT_FILE = "dog_cat_behavior_dataset_final.json"

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

MIN_LENGTH = 140
SIM_THRESHOLD = 0.88
MAX_RECORDS_PER_URL = 4

# ❌ Remove these completely
BANNED_TERMS = [
    "supplement", "product", "brand", "chew",
    "capsule", "tablet", "purina", "melatonin",
    "cbd", "alpha-casozepine", "buy", "order"
]

# ❌ Citation patterns
CITATION_REGEX = r"(journal|doi\.org|et al\.|\d{4};\d+)"

# ✅ Mechanism-level signals
MECHANISM_TERMS = [
    "instinct", "learning", "conditioning",
    "reinforcement", "stress response",
    "neuro", "dopamine", "serotonin",
    "communication", "body language",
    "territorial", "fear response",
    "social behavior", "development"
]

def normalize(t):
    return re.sub(r"\s+", " ", t.lower()).strip()

def is_citation_only(t):
    return bool(re.search(CITATION_REGEX, t.lower())) and len(t.split()) < 60

def contains_banned(t):
    return any(b in t.lower() for b in BANNED_TERMS)

def has_mechanism(t):
    return any(m in t.lower() for m in MECHANISM_TERMS)

# ---------------- LOAD ----------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"🔹 Starting records: {len(data)}")

# ---------------- STRICT FILTER ----------------

filtered = []
for r in data:
    text = r["content"].strip()
    if len(text) < MIN_LENGTH:
        continue
    if is_citation_only(text):
        continue
    if contains_banned(text):
        continue
    if not has_mechanism(text):
        continue
    filtered.append(r)

print(f"🔍 After strict filtering: {len(filtered)}")

# ---------------- PER-URL DIVERSITY CAP ----------------

by_url = defaultdict(list)
for r in filtered:
    by_url[r["source_url"]].append(r)

balanced = []
for url, records in by_url.items():
    balanced.extend(records[:MAX_RECORDS_PER_URL])

print(f"⚖️ After URL balancing: {len(balanced)}")

# ---------------- SEMANTIC DEDUP ----------------

texts = [normalize(r["content"]) for r in balanced]
embeddings = MODEL.encode(texts, convert_to_tensor=True)

keep = []
used = set()

for i in range(len(balanced)):
    if i in used:
        continue
    keep.append(balanced[i])
    sims = util.cos_sim(embeddings[i], embeddings)[0]
    for j, s in enumerate(sims):
        if j != i and s >= SIM_THRESHOLD:
            used.add(j)

# ---------------- FINAL SAVE ----------------

for i, r in enumerate(keep, start=1):
    r["id"] = i

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(keep, f, indent=2, ensure_ascii=False)

print("\n✅ FINAL behavior dataset ready")
print(f"Final records: {len(keep)}")
print(f"Saved to: {OUTPUT_FILE}")
