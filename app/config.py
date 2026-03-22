import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Retrieval
TOP_K = 3

# Generation settings
TEMPERATURE = 0
MAX_TOKENS = 500

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_STORE_PATH = BASE_DIR / "vector_store"
FAISS_INDEX_FILE = VECTOR_STORE_PATH / "index.faiss"
METADATA_FILE = VECTOR_STORE_PATH / "metadata.pkl"