import time
import numpy as np
import faiss
import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import json

# Load models
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading rerank model...")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load documents
print("Loading documents...")
with open("documents.json", "r") as f:
    docs = json.load(f)

doc_texts = [d["content"] for d in docs]
doc_ids = [d["id"] for d in docs]

# Compute embeddings
print("Computing embeddings...")
doc_embeddings = embed_model.encode(doc_texts, normalize_embeddings=True)

# FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

print("FAISS index ready")

# API setup
app = FastAPI()


@app.get("/")
def home():
    return {"message": "API is running successfully!"}


class QueryRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5


def normalize(scores):
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(float(s) - float(min_s)) / (float(max_s) - float(min_s)) for s in scores]


@app.post("/search")
def search(req: QueryRequest):

    start = time.time()

    query_emb = embed_model.encode([req.query], normalize_embeddings=True)

    scores, indices = index.search(query_emb, req.k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        candidates.append({
            "id": int(doc_ids[idx]),
            "content": doc_texts[idx],
            "score": float(score),
            "metadata": docs[idx]["metadata"]
        })

    reranked = False

    if req.rerank:

        pairs = [(req.query, c["content"]) for c in candidates]
        rerank_scores = rerank_model.predict(pairs)

        norm_scores = normalize(rerank_scores)

        for i, c in enumerate(candidates):
            c["score"] = float(norm_scores[i])

        candidates.sort(key=lambda x: x["score"], reverse=True)

        candidates = candidates[:req.rerankK]

        reranked = True

    latency = int((time.time() - start) * 1000)

    return {
        "results": candidates,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": len(docs)
        }
    }


# IMPORTANT: Render needs this
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
