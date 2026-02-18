from app.retriever import retrieve
from app.llm import generate_answer

query = "How do I track my order?"

#1: Retrieve top 2 relevant chunks
results= retrieve(query, top_k=2)

#2: Generate answer
answer= generate_answer(results,query)

print("\nFinal Answer:\n")
print(answer)