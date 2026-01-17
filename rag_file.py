import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

index = faiss.read_index("./faiss_index/index.faiss")

chunks = []
with open("./faiss_index/chunks.txt", encoding="utf-8") as f:
    for line in f:
        source, text = line.split("|||", 1)
        chunks.append({"source": source, "text": text})

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

with open("events.json", "r", encoding="utf-8") as f:
    EVENTS = json.load(f)["events"]

def retrieve_rag_context(query, top_k=3):
    query_emb = embed_model.encode([query])
    _, indices = index.search(np.array(query_emb), top_k)

    contexts = []
    for idx in indices[0]:
        contexts.append(chunks[idx]["text"])

    return "\n\n".join(contexts)

def match_events(user_city, user_state, category):
    matched = []

    for e in EVENTS:
        if category in e["categories"]:
            if user_city in e["cities"] or user_state in e["states"] or "All" in e["states"]:
                matched.append(e)

    return matched[:2]  

