import time
import os
import json
import faiss
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer, CrossEncoder

# ------------------------
# Load Models (cached in memory)
# ------------------------

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranking model...")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------
# Load documents
# ------------------------

print("Loading documents...")
with open("documents.json", "r") as f:
    docs = json.load(f)

doc_texts = [d["content"] for d in docs]
doc_ids = [d["id"] for d in docs]

# ------------------------
# Compute embeddings ONCE (cached)
# ------------------------

print("Computing embeddings...")
doc_embeddings = embed_model.encode(
    doc_texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

# ------------------------
# Build FAISS index
# ------------------------

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

print(f"FAISS ready with {len(docs)} docs")

# ------------------------
# FastAPI app
# ------------------------

app = FastAPI(title="Semantic Search API")


@app.get("/")
def home():
    return {"status": "running", "docs": len(docs)}


# ------------------------
# Request schema
# ------------------------

class QueryRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5


# ------------------------
# Normalize scores 0–1
# ------------------------

def normalize(scores):

    scores = np.array(scores)

    if scores.max() == scores.min():
        return np.ones(len(scores))

    return (scores - scores.min()) / (scores.max() - scores.min())


# ------------------------
# SEARCH ENDPOINT
# ------------------------

@app.post("/search")
def search(req: QueryRequest):

    start_time = time.time()

    # embed query
    query_embedding = embed_model.encode(
        [req.query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # vector search
    scores, indices = index.search(query_embedding, req.k)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        candidates.append({
            "id": int(doc_ids[idx]),
            "content": doc_texts[idx],
            "metadata": docs[idx]["metadata"],
            "score": float(score)
        })

    reranked = False

    # ------------------------
    # RERANK STEP
    # ------------------------

    if req.rerank and len(candidates) > 0:

        pairs = [(req.query, c["content"]) for c in candidates]

        rerank_scores = rerank_model.predict(pairs)

        normalized_scores = normalize(rerank_scores)

        for i in range(len(candidates)):
            candidates[i]["score"] = float(normalized_scores[i])

        candidates.sort(key=lambda x: x["score"], reverse=True)

        candidates = candidates[:req.rerankK]

        reranked = True

    latency = int((time.time() - start_time) * 1000)

    return {

        "results": candidates,

        "reranked": reranked,

        "metrics": {
            "latency": latency,
            "totalDocs": len(docs)
        }
    }


# ------------------------
# Render PORT FIX
# ------------------------

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(app, host="0.0.0.0", port=port)
