import psycopg
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
conn = psycopg.connect("postgresql://postgres:devpass@localhost:5432/postgres")
cur = conn.cursor()

query = input("Search: ")
q = model.encode([query], normalize_embeddings=True)[0]

cur.execute(
    "SELECT text, embedding <=> %s AS distance FROM sentences ORDER BY distance LIMIT 5",
    (str(q.tolist()),)
)

for text, distance in cur.fetchall():
    print(f"{distance:.3f}  {text}")

conn.close()
