import numpy as np                                                                                
from sentence_transformers import SentenceTransformer

f=open("data/sentences.txt", encoding="utf-8")
sentences=[]
for line in f:
    if line.strip():
        sentences.append(line.strip())
f.close()

model=SentenceTransformer("all-MiniLM-L6-v2")    
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)
np.save("data/embeddings.npy", embeddings)
