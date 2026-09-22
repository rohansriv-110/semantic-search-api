import os
import numpy as np
import psycopg

f = open("data/sentences.txt", encoding="utf-8")
sentences = []
for line in f:
    if line.strip():
        sentences.append(line.strip())
f.close()

embeddings = np.load("data/embeddings.npy")

conn = psycopg.connect(os.getenv("DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/postgres"))
cur = conn.cursor()

for text, vec in zip(sentences, embeddings):
    cur.execute(
        "INSERT INTO sentences (text, embedding) VALUES (%s, %s)",
        (text, str(vec.tolist()))
    )

conn.commit()
conn.close()
print("Inserted", len(sentences), "rows")
