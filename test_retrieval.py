import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("./faiss_index/index.faiss")

chunks = []
with open("./faiss_index/chunks.txt", encoding="utf-8") as f:
    for line in f:
        source, text = line.split("|||", 1)
        chunks.append({"source": source, "text": text})

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results

if __name__ == "__main__":
    query = "Is traditional pottery outdated compared to modern décor items?"
    results = search(query)

    print("\ nQuery:", query)
    print("\n Retrieved chunks:\n")

    for r in results:
        print("SOURCE:", r["source"])
        print(r["text"][:500])
        print("-" * 80)
