import json
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================

INPUT_FILE = "dog_cat_behavior_dataset.json"
OUTPUT_FILE = "dog_cat_behavior_dataset_optimized.json"

MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.86   # STRONG dedup for behavior
MIN_LENGTH = 120

# ===============================
# FILTER RULES
# ===============================

QUESTION_PATTERNS = [
    r"\bwhy does\b",
    r"\bwhy do dogs\b",
    r"\bwhy do cats\b",
    r"\bwhat does it mean\b",
    r"\bshould i worry\b",
    r"\bis it normal\b",
    r"\bcan dogs\b",
    r"\bcan cats\b"
]

AD_PATTERNS = [
    "buy now", "shop", "subscribe", "discount",
    "supplement", "product", "brand", "order",
    "sponsored", "affiliate"
]

MECHANISM_KEYWORDS = [
    "instinct", "conditioning", "reinforcement",
    "learning", "behavioral", "neuro",
    "dopamine", "serotonin", "stress",
    "anxiety", "fear", "communication",
    "body language", "ethology", "territorial",
    "social hierarchy", "development",
    "habit formation"
]

# ===============================
# HELPERS
# ===============================

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def is_question_like(text):
    return any(re.search(p, text.lower()) for p in QUESTION_PATTERNS)

def is_ad_like(text):
    return any(p in text.lower() for p in AD_PATTERNS)

def has_mechanism_signal(text):
    return any(k in text.lower() for k in MECHANISM_KEYWORDS)

# ===============================
# LOAD DATA
# ===============================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"🧹 Raw records collected: {len(raw_data)}")

# ===============================
# STRICT CONTENT FILTERING
# ===============================

filtered = []

for r in raw_data:
    content = r.get("content", "").strip()
    if len(content) < MIN_LENGTH:
        continue
    if is_question_like(content):
        continue
    if is_ad_like(content):
        continue
    if not has_mechanism_signal(content):
        continue

    filtered.append(r)

print(f"🔍 After rule-based filtering: {len(filtered)}")

# ===============================
# SEMANTIC DEDUPLICATION
# ===============================

model = SentenceTransformer(MODEL_NAME)

texts = [normalize(r["content"]) for r in filtered]
embeddings = model.encode(texts, convert_to_tensor=True)

keep = []
used = set()

for i in tqdm(range(len(filtered)), desc="🧠 Deduplicating"):
    if i in used:
        continue

    keep.append(filtered[i])
    sims = util.cos_sim(embeddings[i], embeddings)[0]

    for j, s in enumerate(sims):
        if j != i and s >= SIMILARITY_THRESHOLD:
            used.add(j)

# ===============================
# FINAL SAVE
# ===============================

for idx, r in enumerate(keep, start=1):
    r["id"] = idx

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(keep, f, indent=2, ensure_ascii=False)

print("\n✅ Behavior dataset optimized")
print(f"   Final records: {len(keep)}")
print(f"   Saved to: {OUTPUT_FILE}")
