import os
import psycopg
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/postgres")

shelf = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    shelf["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    shelf["conn"] = psycopg.connect(DATABASE_URL)
    yield
    shelf["conn"].close()

app = FastAPI(lifespan=lifespan)

@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=20)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="q cannot be blank")
    vec = shelf["model"].encode([q], normalize_embeddings=True)[0]
    with shelf["conn"].cursor() as cur:
        cur.execute(
            "SELECT text, 1 - (embedding <=> %s) AS score "
            "FROM sentences ORDER BY embedding <=> %s LIMIT %s",
            (str(vec.tolist()), str(vec.tolist()), k),
        )
        rows = cur.fetchall()
    return rows