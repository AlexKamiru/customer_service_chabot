# 🤖 Customer Service Chatbot

> A production-ready RAG (Retrieval-Augmented Generation) chatbot built from scratch and deployed on free-tier cloud infrastructure — no GPU, no paid hosting required.

Built to demonstrate end-to-end ML engineering skills: from document indexing and semantic search, to LLM integration, REST API design, and full-stack deployment.

**Live Demo:** [customerservicechabot-fwhobmzfapptekgssnzmg83.streamlit.app](https://customerservicechabot-fwhobmzfapptekgssnzmg83.streamlit.app)  
**API Docs:** [chabot-api-m0na.onrender.com/docs](https://chabot-api-m0na.onrender.com/docs)

---
##  Real Business Value

Customer service is one of the most expensive operations in any business. Companies spend millions annually on support agents answering the same repetitive questions about orders, returns, and policies.

This chatbot directly addresses that problem:

**Cost Reduction**
A single support agent handles ~50 tickets/day at a cost of $30,000+/year. This chatbot handles unlimited queries simultaneously at near-zero cost, freeing human agents to focus on complex issues that actually need human judgment.

**Consistency**
Human agents give inconsistent answers depending on experience and mood. This chatbot always answers from the same verified source documents, ensuring every customer gets the same accurate information.

**24/7 Availability**
Unlike human agents, the chatbot operates around the clock with no overtime costs — critical for businesses serving customers across different time zones.

**Instant Onboarding**
Adding a new product line or policy? Just drop a `.txt` file into the `data/` folder and restart — no retraining, no fine-tuning, no ML expertise required.

**Transparency and Trust**
Every answer includes citations showing exactly which document and section it came from. Businesses can audit responses and customers can verify information — something generic chatbots cannot offer.

> In short: this system gives small and medium businesses access to enterprise-grade customer service automation without the enterprise price tag.
---
---

##  What This Project Does

This chatbot answers customer service questions grounded in your own documents. Instead of relying on a generic LLM that may hallucinate, it first retrieves the most relevant content from a knowledge base, then generates a cited, accurate answer.

Ask it *"How do I track my order?"* and it will find the exact policy, generate a clear answer, and tell you which document it came from.

---

##  Architecture

```
User Question
     │
     ▼
Streamlit UI  ──────────►  FastAPI Backend
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Voyage AI               FAISS Index
           (Embed Query)         (Retrieve Top-K Chunks)
                    │                     │
                    └──────────┬──────────┘
                               ▼
                          Groq LLM
                    (llama-3.3-70b-versatile)
                     (Generate Answer)
                               │
                               ▼
                     Answer + Citations
```

---

##  Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit | Chat UI |
| Backend | FastAPI | REST API |
| Embeddings | Voyage AI (`voyage-3-lite`) | Semantic search |
| Vector Store | FAISS | Similarity search |
| LLM | Groq (`llama-3.3-70b-versatile`) | Answer generation |
| Backend Hosting | Render (free tier) | API deployment |
| Frontend Hosting | Streamlit Community Cloud | UI deployment |

**Total hosting cost: $0**

---

##  Project Structure

```
customer_service_chatbot/
├── app/
│   ├── config.py          # Configuration and paths
│   ├── embeddings.py      # Vector store builder
│   ├── retriever.py       # FAISS semantic search
│   ├── llm.py             # Groq LLM integration
│   ├── main.py            # FastAPI app and endpoints
│   ├── schemas.py         # Pydantic models
│   ├── prompts.py         # RAG prompt template
│   └── logger.py          # Query logging
├── frontend/
│   ├── streamlit_app.py   # Streamlit chat UI
│   └── requirements.txt   # UI dependencies
├── data/
│   ├── faqs.txt           # FAQ documents
│   └── policies.txt       # Policy documents
├── requirements.txt       # Backend dependencies
└── Dockerfile             # Container configuration
```

---

##  Getting Started

### Prerequisites
- Python 3.10+
- [Voyage AI API key](https://platform.voyageai.com) — free
- [Groq API key](https://console.groq.com) — free

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/AlexKamiru/customer_service_chatbot.git
cd customer_service_chatbot
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create a .env file in the project root
VOYAGE_API_KEY=your_voyage_api_key
GROQ_API_KEY=your_groq_api_key
```

5. **Run the API**
```bash
uvicorn app.main:app --reload
```

6. **Run the UI** (in a separate terminal)
```bash
pip install streamlit requests
streamlit run frontend/streamlit_app.py
```

---

##  API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check |
| POST | `/chat` | Ask a question |
| GET | `/docs` | Swagger UI |

### Example Request
```bash
curl -X POST "https://chabot-api-m0na.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I track my order?"}'
```

### Example Response
```json
{
  "answer": "You can track your order using the tracking number sent to your email or SMS after shipping [2].",
  "sources": [
    {"source_file": "faqs.txt", "chunk_id": 7},
    {"source_file": "faqs.txt", "chunk_id": 6}
  ]
}
```

---

##  How It Works

### 1. Document Indexing
On startup, the app loads `.txt` files from the `data/` folder, splits them into paragraphs, generates embeddings using Voyage AI, and stores them in a FAISS index.

### 2. Query Processing
When a user asks a question, it is embedded using the same Voyage AI model and compared against all document chunks using L2 distance in FAISS. The top-K most similar chunks are returned.

### 3. Answer Generation
The retrieved chunks are passed to Groq's Llama 3.3 70B model along with the user question. The model generates a grounded answer with inline citations referencing the source chunks.

---

##  Key Engineering Decisions

- **Ephemeral filesystem handling** — Render's free tier resets the filesystem on each deploy, so the vector store is rebuilt at startup automatically rather than relying on a cached index.
- **Lazy index loading** — The FAISS index is loaded only after it's built, preventing dimension mismatch errors between old and new embeddings.
- **Decoupled frontend and backend** — Deployed independently with separate dependency files, keeping each service lean and within free-tier memory limits.
- **Lightweight stack** — Replacing `sentence-transformers` + PyTorch (2GB+) with the Voyage AI API brought memory usage from 512MB+ down to under 100MB, making free-tier deployment possible.

---

##  Future Improvements

- [ ] Add support for PDF and DOCX documents
- [ ] Implement conversation memory for multi-turn chat
- [ ] Persist vector store to cloud storage (e.g. AWS S3) to avoid rebuilding on every restart
- [ ] Add evaluation metrics (faithfulness, relevance, hallucination rate)
- [ ] Support multiple knowledge bases / tenants
- [ ] Add user authentication

---

##  Author

**Alex Kamiru**  
[GitHub](https://github.com/AlexKamiru)

---

## ⭐ If you found this useful, give it a star on GitHub!
