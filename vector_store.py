import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from models import EnrichedCandidate, DomainCard
from config import settings

logger = logging.getLogger("VectorStore")

class VectorStore:
    """
    Manages vector embeddings for:
    1. Raw domain phrases (baseline for evaluation)
    2. Enriched domain cards (enriched context representation)
    Fast cosine similarity retrieval.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model = None
        self._candidates: List[EnrichedCandidate] = []
        self._card_embeddings: Optional[np.ndarray] = None
        self._card_to_candidate_map: List[Tuple[int, DomainCard]] = []
        
        # Raw baseline embeddings
        self._raw_domain_embeddings: Optional[np.ndarray] = None
        self._raw_to_candidate_map: List[Tuple[int, str]] = []

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def warmup(self):
        """Warm up embedding model and compute a dummy vector so inference is instant on first request."""
        try:
            logger.info("Warming up embedding model...")
            _ = self.encode(["тестовый прогрев эмбеддингов"])
            logger.info("Embedding model warmed up.")
        except Exception as e:
            logger.warning(f"Embedding warmup warning: {e}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes texts into normalized numpy embeddings."""
        if not texts:
            return np.empty((0, 312))
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def index_candidates(self, candidates: List[EnrichedCandidate]):
        """
        Indexes enriched candidates' domain cards into vector space.
        Also indexes raw domain phrases for comparative evaluation.
        """
        self._candidates = candidates
        self._card_to_candidate_map = []
        self._raw_to_candidate_map = []

        card_texts = []
        raw_texts = []

        for cand in candidates:
            # Enriched cards
            for card in cand.domain_cards:
                text = card.enriched_text or card.build_enriched_text()
                card_texts.append(text)
                self._card_to_candidate_map.append((cand.id, card))

            # Raw phrases for baseline comparison
            for d in cand.cleaned_domains:
                raw_texts.append(d)
                self._raw_to_candidate_map.append((cand.id, d))

        card_emb_file = settings.CACHE_DIR / "card_embeddings.npy"
        raw_emb_file = settings.CACHE_DIR / "raw_embeddings.npy"

        if card_emb_file.exists() and raw_emb_file.exists():
            logger.info("Loading cached embeddings from disk...")
            try:
                self._card_embeddings = np.load(card_emb_file)
                self._raw_domain_embeddings = np.load(raw_emb_file)
                if len(self._card_embeddings) == len(card_texts) and len(self._raw_domain_embeddings) == len(raw_texts):
                    logger.info(f"Loaded {len(self._card_embeddings)} cards and {len(self._raw_domain_embeddings)} raw embeddings instantly.")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Recomputing.")

        logger.info(f"Encoding {len(card_texts)} enriched domain cards...")
        if card_texts:
            self._card_embeddings = self.encode(card_texts)
            np.save(card_emb_file, self._card_embeddings)
        else:
            self._card_embeddings = np.empty((0, 312))

        logger.info(f"Encoding {len(raw_texts)} raw domain phrases for baseline...")
        if raw_texts:
            self._raw_domain_embeddings = self.encode(raw_texts)
            np.save(raw_emb_file, self._raw_domain_embeddings)
        else:
            self._raw_domain_embeddings = np.empty((0, 312))

    def search_raw_baseline(self, query: str, top_k: int = 20) -> List[Tuple[int, str, float]]:
        """
        Baseline search (Method 1 from Slide 10):
        Matches raw query string against raw domain phrases using cosine similarity.
        """
        if self._raw_domain_embeddings is None or len(self._raw_domain_embeddings) == 0:
            return []

        q_vec = self.encode([query])[0]
        # Dot product because vectors are normalized
        sims = np.dot(self._raw_domain_embeddings, q_vec)

        candidate_best_scores: Dict[int, Tuple[str, float]] = {}
        for idx, sim in enumerate(sims):
            cid, raw_domain = self._raw_to_candidate_map[idx]
            val = float(sim)
            if cid not in candidate_best_scores or val > candidate_best_scores[cid][1]:
                candidate_best_scores[cid] = (raw_domain, val)

        sorted_cands = sorted(candidate_best_scores.items(), key=lambda x: x[1][1], reverse=True)
        return [(cid, dom_score[0], dom_score[1]) for cid, dom_score in sorted_cands[:top_k]]

    def search_enriched(self, query: str, top_k: int = 50) -> List[Tuple[int, DomainCard, float]]:
        """
        Enriched Card search (Method 2 from Slide 10):
        Matches query against rich 4-field Domain Cards using cosine similarity.
        """
        if self._card_embeddings is None or len(self._card_embeddings) == 0:
            return []

        q_vec = self.encode([query])[0]
        sims = np.dot(self._card_embeddings, q_vec)

        candidate_best_cards: Dict[int, Tuple[DomainCard, float]] = {}
        for idx, sim in enumerate(sims):
            cid, card = self._card_to_candidate_map[idx]
            val = float(sim)
            if cid not in candidate_best_cards or val > candidate_best_cards[cid][1]:
                candidate_best_cards[cid] = (card, val)

        sorted_cands = sorted(candidate_best_cards.items(), key=lambda x: x[1][1], reverse=True)
        return [(cid, card_score[0], card_score[1]) for cid, card_score in sorted_cands[:top_k]]

vector_store = VectorStore()
