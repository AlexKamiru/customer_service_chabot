import streamlit as st
import requests

API_URL = "https://chabot-api-m0na.onrender.com"

st.set_page_config(
    page_title = "Customer Service Chatbot",
    page_icon = "🤖",
    layout="centered"
)

st.title("🤖 Customer Service Chatbot")
st.caption("Ask me anything about orders, shipping, returns, and more.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("sources"):
                for source in message["sources"]:
                    st.caption(f"• {source['source_file']} (chunk {source['chunk_id']})")   


# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt},
                    timeout=30
                )
                data = response.json()
                answer = data.get("answer","Sorry, I could not get a response.")
                sources = data.get("sources",[])
            except Exception as e:
                answer = f"Could not reach the API: {str(e)}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("📎 Sources"):
                for source in sources:
                    st.caption(f"• {source['source_file']} (chunk {source['chunk_id']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })                            
               
