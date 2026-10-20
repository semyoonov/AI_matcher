import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Set offline flag to avoid huggingface hub update checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load .env file if available
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Settings(BaseModel):
    # LLM Provider Configuration ("groq" or "mistral")
    LLM_PROVIDER: str = Field(default=os.getenv("LLM_PROVIDER", "groq"))
    
    # API Keys
    GROQ_API_KEY: str = Field(default=os.getenv("GROQ_API_KEY", ""))
    MISTRAL_API_KEY: str = Field(default=os.getenv("MISTRAL_API_KEY", ""))
    
    # Models
    # Groq: llama-3.3-70b-versatile, llama-3.1-8b-instant
    GROQ_MODEL: str = Field(default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    # Mistral: mistral-small-latest, open-mistral-7b
    MISTRAL_MODEL: str = Field(default=os.getenv("MISTRAL_MODEL", "mistral-small-latest"))
    
    # Local Embedding Model (fast Russian/multilingual sentence transformer)
    EMBEDDING_MODEL_NAME: str = Field(default="cointegrated/rubert-tiny2")
    
    # Cache and Storage
    DATA_PATH: Path = Field(default=BASE_DIR / "resumes_domains_rows.sql")
    CACHE_DIR: Path = Field(default=BASE_DIR / "cache")
    CACHE_CANDIDATES_FILE: Path = Field(default=BASE_DIR / "cache" / "enriched_candidates.json")
    GRAPH_CACHE_FILE: Path = Field(default=BASE_DIR / "cache" / "domain_graph.json")
    
    # Search & Retrieval Parameters (as specified in presentation slides 7, 8, 9)
    TOP_K_RETRIEVAL: int = Field(default=50, description="Top-K candidates passed to stage 2 reranker")
    GRAPH_EXPANSION_HOPS: int = Field(default=1, description="Graph traversal depth for domain query expansion")
    
    # Scoring Weights for Final Match Formula
    WEIGHT_ROLE: float = Field(default=0.25)
    WEIGHT_SENIORITY: float = Field(default=0.20)
    WEIGHT_STACK: float = Field(default=0.25)
    WEIGHT_DOMAIN: float = Field(default=0.30)
    
    def get_active_provider(self) -> str:
        """Determines active LLM provider based on explicit choice or available keys."""
        prov = self.LLM_PROVIDER.lower()
        if prov == "groq" and self.GROQ_API_KEY:
            return "groq"
        if prov == "mistral" and self.MISTRAL_API_KEY:
            return "mistral"
        if self.GROQ_API_KEY:
            return "groq"
        if self.MISTRAL_API_KEY:
            return "mistral"
        return "mock"  # Fallback to smart deterministic engine if keys not set yet

settings = Settings()
