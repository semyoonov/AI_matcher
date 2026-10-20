import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from config import settings
from models import (
    RawCandidate, 
    EnrichedCandidate, 
    DomainCard, 
    RecruiterQuery, 
    CandidateScoreResult
)
from data_loader import load_raw_candidates
from domain_cleaner import clean_candidate_domains, parse_recruiter_query
from domain_enricher import domain_enricher
from domain_graph import domain_graph
from vector_store import vector_store
from reranker import reranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DomainScoringPipeline")

class DomainScoringPipeline:
    """
    End-to-End Pipeline implementing the architecture from presentation:
    'Архитектура скоринга: граф связей доменов и реранкер'
    """
    def __init__(self):
        self.candidates: Dict[int, EnrichedCandidate] = {}
        self.is_indexed = False

    def initialize(self, force_recompute: bool = False):
        """
        Loads, cleans, enriches, builds knowledge graph, and vectorizes candidates.
        Uses persistent disk cache for instant startup.
        """
        if self.is_indexed and not force_recompute:
            return

        os.makedirs(settings.CACHE_DIR, exist_ok=True)
        cache_path = settings.CACHE_CANDIDATES_FILE

        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading enriched candidates from cache: {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    cards = [DomainCard(**c) for c in item.get("domain_cards", [])]
                    for card in cards:
                        if not card.enriched_text:
                            card.build_enriched_text()
                    cand = EnrichedCandidate(
                        id=item["id"],
                        cleaned_domains=item["cleaned_domains"],
                        primary_industries=item["primary_industries"],
                        domain_cards=cards
                    )
                    self.candidates[cand.id] = cand
                    # Populate graph with enriched connections
                    for card in cards:
                        domain_graph.auto_expand_with_card(card.domain, card.industry)

                vector_store.index_candidates(list(self.candidates.values()))
                vector_store.warmup()
                self.is_indexed = True
                logger.info(f"Successfully loaded and indexed {len(self.candidates)} candidates from cache.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # Process raw mock data
        logger.info("Ingesting and enriching mock dataset from scratch...")
        raw_candidates = load_raw_candidates()
        enriched_list = []

        for raw_c in raw_candidates:
            enriched = domain_enricher.enrich_candidate(raw_c)
            self.candidates[enriched.id] = enriched
            enriched_list.append(enriched)
            # Update knowledge graph
            for card in enriched.domain_cards:
                domain_graph.auto_expand_with_card(card.domain, card.industry)

        # Index vectors
        vector_store.index_candidates(enriched_list)

        # Save to disk cache
        try:
            cache_data = [
                {
                    "id": c.id,
                    "cleaned_domains": c.cleaned_domains,
                    "primary_industries": c.primary_industries,
                    "domain_cards": [card.model_dump() for card in c.domain_cards]
                }
                for c in enriched_list
            ]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved enriched candidate index to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving candidate cache: {e}")

        self.is_indexed = True

    def search_and_score(
        self,
        query_text: str,
        top_k: int = 10,
        apply_hard_industry_filter: bool = False,
        base_role_score: float = 1.0,
        base_seniority_score: float = 1.0,
        base_stack_score: float = 1.0
    ) -> Tuple[RecruiterQuery, List[CandidateScoreResult]]:
        """
        Executes complete search and scoring pipeline:
        1. Parse recruiter query
        2. Filter by industry (Slide 7 Step 1)
        3. Nearest domains in graph & vector space (Slide 7 Steps 2 & 3)
        4. Re-rank top candidates with Cross-Encoder LLM (Slide 7 Step 4 & Slide 8)
        5. Composite multi-factor scoring
        """
        if not self.is_indexed:
            self.initialize()

        # Step 1: Parse query
        query = parse_recruiter_query(query_text)
        logger.info(f"Parsed query: industry={query.target_industry}, domains={query.target_business_domains}")

        # Step 2: Industry pre-filtering & retrieval
        # Target domains query string for vector search
        target_str = " ".join(query.target_business_domains) or query_text
        if query.target_nuances:
            target_str += " " + query.target_nuances

        # Retrieve candidates via enriched vector search
        candidate_matches = vector_store.search_enriched(target_str, top_k=settings.TOP_K_RETRIEVAL)

        # Step 3: Combine with Graph Proximity
        staged_candidates: List[Dict[str, Any]] = []
        for cid, best_card, vec_sim in candidate_matches:
            cand = self.candidates.get(cid)
            if not cand:
                continue

            # Hard filter check if requested
            industry_ok = True
            if apply_hard_industry_filter and query.target_industry:
                target_ind = query.target_industry.lower()
                industry_ok = any(target_ind in ind.lower() for ind in cand.primary_industries)
                if not industry_ok:
                    continue

            # Graph topological proximity score
            graph_score = domain_graph.compute_graph_proximity(
                query.target_business_domains or [query.target_industry or ""],
                cand.cleaned_domains
            )

            # Combined Stage 1 retrieval score: vector similarity + graph proximity
            stage1_score = 0.60 * vec_sim + 0.40 * graph_score

            staged_candidates.append({
                "cid": cid,
                "cand": cand,
                "best_card": best_card,
                "stage1_score": stage1_score,
                "industry_ok": industry_ok
            })

        # Sort by Stage 1 score and select top candidates for reranking
        staged_candidates.sort(key=lambda x: x["stage1_score"], reverse=True)
        top_to_rerank = staged_candidates[:min(top_k + 4, 12, len(staged_candidates))]

        # Step 4: LLM Reranking (Cross-Encoder detailed domain scoring)
        # Evaluates candidates in batch to respect LLM rate limits (Slide 8)
        batch_evals = reranker.evaluate_batch(query, top_to_rerank)
        results: List[CandidateScoreResult] = []

        for item in top_to_rerank:
            cid = item["cid"]
            cand = item["cand"]
            best_card = item["best_card"]
            stage1_score = item["stage1_score"]

            # Retrieve score from batch evaluation
            if cid in batch_evals:
                rerank_score, verdict, reasoning = batch_evals[cid]
            else:
                # No silent 0.5 default! Use true Stage 1 semantic & graph score
                rerank_score = stage1_score
                if stage1_score >= 0.80:
                    verdict = "ТОЧНОЕ СОВПАДЕНИЕ"
                    reasoning = f"Высокая топологическая близость по графу связей и семантическому контексту ({stage1_score*100:.0f}%)."
                elif stage1_score >= 0.55:
                    verdict = "БЛИЗКИЙ ДОМЕН"
                    reasoning = f"Подтвержден опыт в смежных узлах онтологии доменов ({stage1_score*100:.0f}%)."
                elif stage1_score >= 0.30:
                    verdict = "СМЕЖНЫЙ ДОМЕН"
                    reasoning = f"Частичное пересечение по смежным направлениям ({stage1_score*100:.0f}%)."
                else:
                    verdict = "НЕТ СОВПАДЕНИЯ"
                    reasoning = "Опыт кандидата в предметной области не закрывает ключевые требования запроса."

            # Domain score: weighted combination favoring reranker
            domain_score = 0.85 * rerank_score + 0.15 * stage1_score

            # Composite overall candidate score (role + seniority + stack + domain)
            composite_score = (
                settings.WEIGHT_ROLE * base_role_score +
                settings.WEIGHT_SENIORITY * base_seniority_score +
                settings.WEIGHT_STACK * base_stack_score +
                settings.WEIGHT_DOMAIN * domain_score
            )

            results.append(CandidateScoreResult(
                candidate_id=cid,
                industry_matched=item["industry_ok"],
                stage1_retrieval_score=round(stage1_score, 4),
                stage2_rerank_score=round(rerank_score, 4),
                domain_score=round(domain_score, 4),
                matched_domains=cand.cleaned_domains,
                best_matching_card=best_card,
                verdict_label=verdict,
                reranker_reasoning=reasoning,
                composite_score=round(composite_score, 4)
            ))

        # Sort final results by domain score / composite score
        results.sort(key=lambda r: r.domain_score, reverse=True)
        return query, results[:top_k]

pipeline = DomainScoringPipeline()
