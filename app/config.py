import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4"
EMBEDDING_MODEL = "text-embedding-3-small"

# Retrieval
TOP_K = 3

# Generation settings
TEMPERATURE = 0
MAX_TOKENS = 500

# Paths
DATA_PATH = "data"
VECTOR_STORE_PATH = "vector_store"
FAISS_INDEX_FILE = "vector_store/index.faiss"
METADATA_FILE = "vector_store/metadata.pkl"