import numpy as np
from sentence_transformers import SentenceTransformer

f = open("data/sentences.txt", encoding="utf-8")
sentences = []
for line in f:
    if line.strip():
        sentences.append(line.strip())
f.close()

embeddings = np.load("data/embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")
query = input("Search: ")
q = model.encode([query], normalize_embeddings=True)[0]
scores=embeddings@q
top = np.argsort(-scores)[:5]
for i in top:
    print(f"{scores[i]:.3f}  {sentences[i]}")
    
