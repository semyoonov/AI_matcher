import re
import logging
from typing import List, Dict, Any
from models import RecruiterQuery
from llm_client import llm_client

logger = logging.getLogger("DomainCleaner")

CLEANING_SYSTEM_PROMPT = """Ты эксперт по анализу резюме и бизнес-доменов (предметных областей).
Твоя задача — очистить список сущностей, извлеченных из резюме кандидата:
1. Исправь опечатки (например: "виде аналитика" -> "видеоаналитика", "программматическая" -> "программатик").
2. Удали НЕ-доменные факты:
   - общие софт-скиллы ("работа в команде", "управление персоналом", если это не HR-tech)
   - чисто технические generic-термины без бизнес-специфики ("api сервисы", "сетевые компоненты", "веб-приложения", "тестирование программного обеспечения").
3. Удали явные дубликаты и синонимы (например: "страхование" и "страховой сектор" -> "страхование", "промышленный iot" и "промышленный интернет вещей" -> "промышленный IoT").
4. Сохрани бизнес-домены (финтех, интеграция банковского ядра, кредитный скоринг, биллинг, e-commerce, mes-система и т.д.).

Ответь строго в формате JSON:
{
  "cleaned_domains": ["домен 1", "домен 2", ...]
}
"""

RECRUITER_QUERY_PROMPT = """Ты AI-ассистент рекрутера. Проанализируй текст поискового запроса на вакансию и разбей его на структурированные сущности:
1. role: роль / позиция (например: "автотестировщик", "backend разработчик")
2. seniority: грейд (junior, middle, senior, lead)
3. stack: список технологий/инструментов (например: ["Kafka", "Python", "Docker"])
4. target_industry: глобальная отрасль (например: "финтех", "ритейл", "медицина")
5. target_business_domains: конкретные целевые бизнес-домены (например: ["интеграция банковского ядра", "кредитный скоринг"])
6. target_nuances: детализация и специфика бизнес-опыта (например: "опыт с интеграцией банковского ядра, АБС, межбанковские протоколы")

Ответь строго в формате JSON:
{
  "role": "...",
  "seniority": "...",
  "stack": ["..."],
  "target_industry": "...",
  "target_business_domains": ["..."],
  "target_nuances": "..."
}
"""

# Common non-domain phrases for fast pre-filtering
NON_DOMAIN_KEYWORDS = {
    "работа в команде", "api сервисы", "сетевые компоненты", "веб-приложения", 
    "мультимедийные компоненты", "разработка программного обеспечения", 
    "тестирование программного обеспечения", "тестирование и обеспечение качества по",
    "обеспечение качества программного обеспечения", "sql-аналитика", "валидация данных",
    "a/b-эксперименты", "проведение a/b-экспериментов"
}

def clean_candidate_domains(raw_domains: List[str]) -> List[str]:
    """
    Cleans raw candidate domains: fixes typos, removes non-domain facts, deduplicates.
    """
    if not raw_domains:
        return []

    # Fast deterministic pre-cleaning
    deduped = []
    seen = set()
    for d in raw_domains:
        norm = d.strip().lower()
        if norm in NON_DOMAIN_KEYWORDS:
            continue
        # Typo correction heuristics
        if norm == "виде аналитика":
            norm = "видеоаналитика"
        if norm == "страховой сектор":
            norm = "страхование"
        if norm == "банковский сектор":
            norm = "банкинг"
        if norm == "промышленный интернет вещей":
            norm = "промышленный iot"
            
        if norm not in seen:
            seen.add(norm)
            deduped.append(norm)

    # If LLM key is configured, invoke LLM for deep cleaning
    if getattr(llm_client, "provider", None) in ("groq", "mistral") and len(deduped) > 3:
        try:
            user_prompt = f"Исходный список доменов кандидата:\n{deduped}"
            data = llm_client.generate_json(CLEANING_SYSTEM_PROMPT, user_prompt)
            if "cleaned_domains" in data and isinstance(data["cleaned_domains"], list):
                return data["cleaned_domains"]
        except Exception as e:
            logger.warning(f"LLM cleaning fallback to heuristic: {e}")

    return deduped


def parse_recruiter_query(query_text: str) -> RecruiterQuery:
    """
    Extracts role, seniority, stack, target industry and business domains from recruiter query.
    """
    try:
        data = llm_client.generate_json(RECRUITER_QUERY_PROMPT, query_text)
        return RecruiterQuery(
            raw_text=query_text,
            extracted_role=data.get("role"),
            extracted_seniority=data.get("seniority"),
            extracted_stack=data.get("stack", []),
            target_industry=data.get("target_industry"),
            target_business_domains=data.get("target_business_domains", []),
            target_nuances=data.get("target_nuances")
        )
    except Exception as e:
        logger.error(f"Error parsing recruiter query: {e}")
        from llm_client import SemanticRuleEngine
        parsed = SemanticRuleEngine()._handle_query_parse(query_text)
        return RecruiterQuery(
            raw_text=query_text,
            extracted_role=parsed.get("role"),
            extracted_seniority=parsed.get("seniority"),
            extracted_stack=parsed.get("stack", []),
            target_industry=parsed.get("target_industry"),
            target_business_domains=parsed.get("target_business_domains", []),
            target_nuances=parsed.get("target_nuances")
        )
