import logging
from typing import List, Dict, Any, Tuple, Optional
from models import DomainCard, RecruiterQuery
from llm_client import llm_client

logger = logging.getLogger("Reranker")

RERANKER_SYSTEM_PROMPT = """Ты эксперт по оценке соответствия кандидатов (Cross-Encoder LLM Reranker).
Твоя задача — детально сопоставить поисковый запрос рекрутера с обогащёнными доменными карточками кандидата и выставить точный балл соответствия предметной области.

Шкала скоринга:
- 0.90 .. 1.00: [ТОЧНОЕ СОВПАДЕНИЕ] — кандидат имеет прямой подтвержденный опыт в требуемом узком домене (например: прямо указана интеграция банковского ядра, АБС, core banking).
- 0.70 .. 0.89: [БЛИЗКИЙ ДОМЕН] — кандидат имеет опыт в глубоко связанных подсистемах той же сферы (например: банковские интеграции, корпоративный онлайн-банкинг, кредитный конвейер банка, факторинг).
- 0.35 .. 0.69: [СМЕЖНЫЙ ДОМЕН] — опыт в рамках общей макро-отрасли (например: общий финтех, платёжные операции, биллинг, эквайринг, но без глубоких банковских интеграций ядра).
- 0.00 .. 0.34: [НЕТ СОВПАДЕНИЯ] — опыт из совершенно чужих областей (e-commerce, геймдев, реклама, медицина, промышленный IoT).

Ответь строго в формате JSON:
{
  "score": 0.95,
  "verdict": "ТОЧНОЕ СОВПАДЕНИЕ",
  "reasoning": "Краткое и емкое обоснование на русском языке, объясняющее, почему кандидат подходит или нет под специфику домена."
}
"""

BATCH_RERANKER_SYSTEM_PROMPT = """Ты эксперт по оценке соответствия кандидатов (Cross-Encoder LLM Reranker).
Твоя задача — детально сопоставить поисковый запрос рекрутера со списком кандидатов и их обогащёнными доменными карточками, и выставить точный балл соответствия предметной области каждому кандидату.

Шкала скоринга:
- 0.90 .. 1.00: [ТОЧНОЕ СОВПАДЕНИЕ] — кандидат имеет прямой подтвержденный опыт в требуемом узком домене (например: прямо указана интеграция банковского ядра, АБС, core banking).
- 0.70 .. 0.89: [БЛИЗКИЙ ДОМЕН] — кандидат имеет опыт в глубоко связанных подсистемах той же сферы (например: банковские интеграции, корпоративный онлайн-банкинг, кредитный конвейер банка, факторинг).
- 0.35 .. 0.69: [СМЕЖНЫЙ ДОМЕН] — опыт в рамках общей макро-отрасли (например: общий финтех, платёжные операции, биллинг, эквайринг, но без глубоких банковских интеграций ядра).
- 0.00 .. 0.34: [НЕТ СОВПАДЕНИЯ] — опыт из совершенно чужих областей (e-commerce, геймдев, реклама, медицина, промышленный IoT).

Ответь строго в формате JSON:
{
  "evaluations": [
    {
      "candidate_id": 79,
      "score": 0.96,
      "verdict": "ТОЧНОЕ СОВПАДЕНИЕ",
      "reasoning": "Краткое и емкое обоснование на русском языке."
    }
  ]
}
"""

class LLMReranker:
    """
    Implements Stage 4 of the pipeline (Slides 7 & 8):
    Precise cross-attention evaluation between recruiter prompt and candidate's enriched domain context.
    Supports single and batch evaluation to respect LLM rate limits.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def evaluate_batch(
        self,
        query: RecruiterQuery,
        candidate_items: List[Dict[str, Any]]
    ) -> Dict[int, Tuple[float, str, str]]:
        """
        Evaluates a batch of candidates in a single LLM call to respect rate limits (Slide 8).
        Returns a mapping of candidate_id -> (score, verdict, reasoning).
        """
        results: Dict[int, Tuple[float, str, str]] = {}
        to_query_items = []

        # 1. Check cache first
        for item in candidate_items:
            cid = item["cid"]
            cand = item["cand"]
            cache_key = f"{query.raw_text}__cand_{cid}__" + "_".join(cand.cleaned_domains)
            if cache_key in self._cache:
                c_res = self._cache[cache_key]
                results[cid] = (c_res["score"], c_res["verdict"], c_res["reasoning"])
            else:
                to_query_items.append(item)

        if not to_query_items:
            return results

        # 2. Build single batch prompt for remaining candidates
        cands_prompt_parts = []
        for item in to_query_items:
            cid = item["cid"]
            cand = item["cand"]
            cards_summary = "\n".join([f"    * {card.enriched_text}" for card in cand.domain_cards])
            cands_prompt_parts.append(
                f"--- Кандидат #{cid} ---\n"
                f"Домены: {', '.join(cand.cleaned_domains)}\n"
                f"Обогащенные карточки:\n{cards_summary}"
            )

        user_prompt = (
            f"Запрос рекрутера:\n{query.raw_text}\n\n"
            f"Целевая отрасль: {query.target_industry or 'любая'}\n"
            f"Целевые бизнес-домены: {', '.join(query.target_business_domains)}\n"
            f"Нюансы опыта: {query.target_nuances or 'не указаны'}\n\n"
            f"Список кандидатов для оценки:\n" + "\n\n".join(cands_prompt_parts)
        )

        try:
            data = llm_client.generate_json(BATCH_RERANKER_SYSTEM_PROMPT, user_prompt)
            evals = data.get("evaluations", [])
            for ev in evals:
                cid_raw = ev.get("candidate_id")
                if cid_raw is not None:
                    cid = int(cid_raw)
                    sc = float(ev.get("score", 0.5))
                    vd = str(ev.get("verdict", "СМЕЖНЫЙ ДОМЕН"))
                    rs = str(ev.get("reasoning", ""))
                    results[cid] = (sc, vd, rs)
                    item_matched = next((it for it in to_query_items if it["cid"] == cid), None)
                    if item_matched:
                        cache_key = f"{query.raw_text}__cand_{cid}__" + "_".join(item_matched["cand"].cleaned_domains)
                        self._cache[cache_key] = {"score": sc, "verdict": vd, "reasoning": rs}
        except Exception as e:
            logger.warning(f"Batch reranking failed: {e}. Falling back to individual/rule evaluation.")

        # 3. For any candidate missing from batch response, evaluate instantly with SemanticRuleEngine
        if len(results) < len(to_query_items):
            from llm_client import SemanticRuleEngine
            rule_engine = SemanticRuleEngine()
            for item in to_query_items:
                cid = item["cid"]
                if cid not in results:
                    cards_summary = " ".join([c.enriched_text for c in item["cand"].domain_cards])
                    single_prompt = f"Запрос рекрутера: {query.raw_text}\nОпыт кандидата: {cards_summary}"
                    res = rule_engine._handle_rerank(single_prompt)
                    sc = res.get("score", 0.05)
                    vd = res.get("verdict", "НЕТ СОВПАДЕНИЯ")
                    rs = res.get("reasoning", "Прямых пересечений по специфике домена не обнаружено.")
                    results[cid] = (sc, vd, rs)
                    cache_key = f"{query.raw_text}__cand_{cid}__" + "_".join(item["cand"].cleaned_domains)
                    self._cache[cache_key] = {"score": sc, "verdict": vd, "reasoning": rs}

        return results

    def evaluate_candidate(
        self, 
        query: RecruiterQuery, 
        candidate_id: int, 
        domain_cards: List[DomainCard],
        cleaned_domains: List[str]
    ) -> Tuple[float, str, str]:
        """
        Evaluates single candidate match against query.
        Returns: (score [0.0..1.0], verdict_label, reasoning)
        """
        cache_key = f"{query.raw_text}__cand_{candidate_id}__" + "_".join(cleaned_domains)
        if cache_key in self._cache:
            res = self._cache[cache_key]
            return res["score"], res["verdict"], res["reasoning"]

        # Compile candidate context
        cards_summary = "\n".join([f"- {card.enriched_text}" for card in domain_cards])
        user_prompt = (
            f"Запрос рекрутера:\n{query.raw_text}\n\n"
            f"Целевая отрасль: {query.target_industry or 'любая'}\n"
            f"Целевые бизнес-домены: {', '.join(query.target_business_domains)}\n"
            f"Нюансы опыта: {query.target_nuances or 'не указаны'}\n\n"
            f"Опыт кандидата #{candidate_id}:\n"
            f"Очищенные домены: {', '.join(cleaned_domains)}\n"
            f"Обогащенные карточки доменов:\n{cards_summary}"
        )

        try:
            data = llm_client.generate_json(RERANKER_SYSTEM_PROMPT, user_prompt)
            score = float(data.get("score", 0.0))
            verdict = data.get("verdict", "НЕТ СОВПАДЕНИЯ")
            reasoning = data.get("reasoning", "")
        except Exception as e:
            logger.warning(f"Reranker failed for candidate #{candidate_id}: {e}. Using rule fallback.")
            from llm_client import SemanticRuleEngine
            engine_res = SemanticRuleEngine()._handle_rerank(user_prompt)
            score = float(engine_res.get("score", 0.05))
            verdict = str(engine_res.get("verdict", "НЕТ СОВПАДЕНИЯ"))
            reasoning = str(engine_res.get("reasoning", "Прямых пересечений по специфике домена не обнаружено."))

        self._cache[cache_key] = {"score": score, "verdict": verdict, "reasoning": reasoning}
        return score, verdict, reasoning

reranker = LLMReranker()
