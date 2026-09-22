import json
import urllib.parse
import urllib.request
from metrics import precision_at_k, ndcg_at_k

API = "https://semantic-search-api-1lew.onrender.com/search"
K = 10

labels = json.load(open("eval/labels.json"))
p_scores, n_scores = [], []

for query, relevant in labels.items():
    url = API + "?" + urllib.parse.urlencode({"q": query, "k": K})
    rows = json.load(urllib.request.urlopen(url))
    retrieved = [row[0] for row in rows]          # which slot holds the id?
    p = precision_at_k(retrieved, relevant, K)
    n = ndcg_at_k(retrieved, relevant, K)
    p_scores.append(p)
    n_scores.append(n)
    print(f"{query:25} P@10={p:.2f}  nDCG@10={n:.2f}")

print(f"\nMEAN  P@10={sum(p_scores)/len(p_scores):.2f}  nDCG@10={sum(n_scores)/len(n_scores):.2f}")