import requests
import json
import re
import time
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# CONFIG
# ==================================================

OUTPUT_FILE = "dog_cat_behavior_dataset.json"

SEED_URLS = [
    # Dogs
    "https://www.akc.org/expert-advice/dog-behavior/",
    "https://www.petmd.com/dog/behavior",
    "https://vcahospitals.com/know-your-pet/dog-behavior",
    "https://www.aspca.org/pet-care/dog-care/dog-behavior",

    # Cats
    "https://www.petmd.com/cat/behavior",
    "https://vcahospitals.com/know-your-pet/cat-behavior",
    "https://www.aspca.org/pet-care/cat-care/cat-behavior"
]

BEHAVIOR_SIGNALS = [
    "behavior", "body language", "aggression", "fear",
    "anxiety", "stress", "socialization", "communication",
    "territorial", "play", "vocalization", "scratching",
    "biting", "barking", "growling", "hissing"
]

JUNK_TERMS = [
    "register", "login", "shop", "cart",
    "program", "course", "class", "trainer",
    "disease", "symptom", "treatment",
    "insurance", "appointment", "review",
    "why does my dog", "why does my cat"
]

DOG_TERMS = ["dog", "canine", "puppy", "bark", "growl"]
CAT_TERMS = ["cat", "feline", "kitten", "meow", "hiss"]

HEADERS = {"User-Agent": "PAWS-AgenticRAG/Behavior"}
MAX_PAGES = 180
MIN_LEN = 180
MAX_LEN = 600
SIM_THRESHOLD = 0.92

# ==================================================

visited = set()
raw_records = []
record_id = 1

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==================================================
# HELPERS
# ==================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def has_behavior_signal(text):
    t = text.lower()
    return any(k in t for k in BEHAVIOR_SIGNALS)

def is_junk(text):
    t = text.lower()
    return any(k in t for k in JUNK_TERMS)

def detect_species(text):
    t = text.lower()
    dog = any(k in t for k in DOG_TERMS)
    cat = any(k in t for k in CAT_TERMS)
    if dog and cat:
        return "both"
    if dog:
        return "dog"
    if cat:
        return "cat"
    return None

def infer_subtopic(text):
    t = text.lower()
    if "aggression" in t:
        return "aggression"
    if "fear" in t or "anxiety" in t:
        return "fear_anxiety"
    if "play" in t:
        return "play_behavior"
    if "communication" in t or "body language" in t:
        return "communication"
    if "territorial" in t:
        return "territorial_behavior"
    if "vocal" in t or "bark" in t or "meow" in t:
        return "vocalization"
    return "general_behavior"

# ==================================================
# CRAWLER
# ==================================================

def crawl(url):
    global record_id

    if url in visited or len(visited) >= MAX_PAGES:
        return []

    visited.add(url)
    print(f"🔍 Crawling: {url}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
            tag.decompose()

        for p in soup.find_all("p"):
            text = clean_text(p.get_text())
            if len(text) < MIN_LEN or len(text) > MAX_LEN:
                continue
            if not has_behavior_signal(text):
                continue
            if is_junk(text):
                continue

            species = detect_species(text)
            if species is None:
                continue

            raw_records.append({
                "id": record_id,
                "topic": "behavior",
                "species": species,
                "subtopic": infer_subtopic(text),
                "content": text,
                "source_url": url
            })
            record_id += 1

        links = []
        for a in soup.find_all("a", href=True):
            nxt = urljoin(url, a["href"])
            if nxt.startswith("http") and nxt not in visited:
                if any(k in nxt.lower() for k in ["behavior", "aggression", "anxiety"]):
                    links.append(nxt)

        return links

    except:
        return []

# ==================================================
# RUN CRAWL
# ==================================================

queue = SEED_URLS.copy()
while queue and len(visited) < MAX_PAGES:
    current = queue.pop(0)
    queue.extend(crawl(current))
    time.sleep(1)

print(f"\n🧹 Raw records collected: {len(raw_records)}")

# ==================================================
# DEDUPLICATION
# ==================================================

texts = [r["content"] for r in raw_records]
embeddings = model.encode(texts, show_progress_bar=True)
sim = cosine_similarity(embeddings)

final = []
removed = set()

for i in range(len(raw_records)):
    if i in removed:
        continue
    final.append(raw_records[i])
    for j in range(i + 1, len(raw_records)):
        if sim[i][j] > SIM_THRESHOLD:
            removed.add(j)

for i, r in enumerate(final, start=1):
    r["id"] = i

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

print(f"\n✅ Behavior dataset ready")
print(f"   Final records: {len(final)}")
print(f"   Saved to: {OUTPUT_FILE}")
