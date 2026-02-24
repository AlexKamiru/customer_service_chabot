import ollama

MODEL_NAME= "phi3:mini"  #the local model

from app.prompts import RAG_PROMPT_TEMPLATE

def generate_answer(context_chunks, question):
    """
    Generate a grounded answer with structured citations. 
    """

    #1 Assign numeric IDs to chunks
    numbered_chunks = []

    for i, chunk in enumerate(context_chunks, start=1):
        numbered_chunks.append({
            "id": i,
            **chunk
        })


    #2 Build context with explicit chunk IDs
    context_text = "\n\n".join(
        [
          f"[{chunk['id']}] Source: {chunk['source_file']} (chunk {chunk['chunk_id']})\n{chunk['text']}"
            for chunk in numbered_chunks
        ]
    )    
    

    #3 Build prompt
    prompt = RAG_PROMPT_TEMPLATE.format(
        context = context_text,
        question = question
     )
    
    #4 Generate response
    response = ollama.chat(
        model = MODEL_NAME,
        messages= [{"role":"user", "content":prompt}]
    )

    answer_text =response["message"]["content"]

    #5 Build deterministic sources section
    sources_section = "\n\nSources:\n"
    for chunk in numbered_chunks:
        sources_section += f"[{chunk['id']}] {chunk['source_file']} (chunk {chunk['chunk_id']})\n"

    return answer_text.strip() + sources_section