import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",  # THIS is your vector DB
        anonymized_telemetry=False
    )
)
collection = chroma_client.get_or_create_collection(
    name="sanskrit_docs",
    metadata={"hnsw:space": "cosine"}
)
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.json")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)


len(chunks)

texts = [c["text"] for c in chunks]

metadatas = [
    {
        "chunk_id": c["chunk_id"],
        "language": c["language"],
        "script": c["script"],
        "section_id": c["section_id"],
        "position": c["position"]
    }
    for c in chunks
]

ids = [c["chunk_id"] for c in chunks]

embeddings = embed_model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

collection.add(
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)


collection.count()
def rerank(results):
    reranked = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = 1 - dist

        # Penalize very short chunks (titles)
        if len(doc) < 30:
            score *= 0.6

        reranked.append({
            "score": round(score, 3),
            "language": meta["language"],
            "text": doc,
            "section_id": meta["section_id"]
        })

    return sorted(reranked, key=lambda x: x["score"], reverse=True)

from collections import Counter

def filter_by_section(reranked, top_n=5):
    section_counts = Counter(r["section_id"] for r in reranked)
    best_section = section_counts.most_common(1)[0][0]

    return [
        r for r in reranked
        if r["section_id"] == best_section
    ][:top_n]

def extract_keywords(query):
    # keep long, distinctive tokens
    return [w for w in query.split() if len(w) >= 5]
def keyword_filter_raw(raw, keywords):
    if not keywords:
        return raw

    filtered_docs = []
    filtered_metas = []
    filtered_dists = []

    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0]
    ):
        if any(k in doc for k in keywords):
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            filtered_dists.append(dist)

    # if nothing matched, return original (fallback)
    if not filtered_docs:
        return raw

    return {
        "documents": [filtered_docs],
        "metadatas": [filtered_metas],
        "distances": [filtered_dists]
    }
def final_query(query):
    keywords = extract_keywords(query)

    # 1. Broad dense recall
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    raw = collection.query(
        query_embeddings=q_emb,
        n_results=40
    )

    # 2. EARLY keyword anchoring
    raw = keyword_filter_raw(raw, keywords)

    # 3. Quality rerank
    reranked = rerank(raw)

    # 4. Section coherence
    final = filter_by_section(reranked)

    return final


