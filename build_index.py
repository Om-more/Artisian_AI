import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm


def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

            documents.append({
                "source": file,
                "text": text
            })

        elif file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "source": file,
                "text": text
            })

    return documents


def chunk_text(text, chunk_size=800, overlap=150):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def prepare_chunks(documents):
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({
                "source": doc["source"],
                "text": chunk
            })

    return all_chunks

def build_faiss_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks

def save_index(index, chunks):
    os.makedirs("./faiss_index", exist_ok=True)

    faiss.write_index(index, "./faiss_index/index.faiss")

    with open("./faiss_index/chunks.txt", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"{c['source']}|||{c['text']}\n")


if __name__ == "__main__":
    docs = load_documents("./Knowledge")
    chunks = prepare_chunks(docs)
    index, chunks = build_faiss_index(chunks)
    save_index(index, chunks)

    print("RAG index built successfully")
