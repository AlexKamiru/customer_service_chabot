import ollama

MODEL_NAME= "phi3:mini"  #the local model

from app.prompts import RAG_PROMPT_TEMPLATE

def generate_answer(context_chunks, question):
    """
    Generate a grounded answer using retrieved context and the local ollama model.

    Args:
        context_chunks(list of dicts): [{text,source_file,chunk_id,score},...]
        question (str): user query

    Returns:
        str: LLM-generated answer    
    """

    #1. Combine retrieved chunks
    context_text = "\n\n".join(
        [f"source: {c['source_file']}\n{c['text']}" for c in context_chunks]
    )

    #2. Create prompt
    prompt = RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)
    
    #3. Generate answer via Ollama
    response = ollama.chat(
        model= MODEL_NAME,
        messages=[{"role":"user" , "content": prompt}]
    )

    #4. Return the content
    return response["message"]["content"]