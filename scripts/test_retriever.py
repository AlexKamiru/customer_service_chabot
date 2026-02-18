# test_retriever.py
from app.retriever import retrieve

query = "How do I track my order?"
results = retrieve(query)

for r in results:
    print(f"Source: {r['source_file']} | Score: {r['score']:.4f}")
    print(f"Text: {r['text']}\n")