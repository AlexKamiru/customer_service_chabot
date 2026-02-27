from app.logger import log_query
from app.retriever import retrieve
from app.llm import generate_answer

# Sample user query
question = "How can I track my order?"

# Step 1: Retrieve chunks
chunks = retrieve(question)

# Step 2: Generate answer
response = generate_answer(chunks, question)

# Step 3: Log the interaction
log_query(
    user_question=question,
    retrieved_chunks=chunks,
    llm_response=response.answer
)

print("Logger test completed. Check chatbot.log file.")