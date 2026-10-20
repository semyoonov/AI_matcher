from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DomainCard(BaseModel):
    """
    Structured domain card as defined in Slide 5 of the presentation:
    - Домен
    - Отрасль (достроено через граф/LLM)
    - Тип организации
    - Бизнес-процесс
    - В чём заключалась работа
    """
    domain: str = Field(description="Normalized domain name, e.g., 'интеграция банковского ядра'")
    industry: str = Field(description="Global industry, e.g., 'финтех'")
    org_type: str = Field(description="Organization type, e.g., 'банк'")
    business_process: str = Field(description="Business process, e.g., 'обмен данными между системами, процессинг'")
    work_scope: str = Field(description="Scope of work, e.g., 'интеграция ядра с внешними и внутренними сервисами'")
    standards_and_tools: str = Field(default="", description="Промышленные стандарты, протоколы и профильные системы (ISO 20022, FIX, АБС, SCADA, PCI DSS и др.)")
    enriched_text: str = Field(default="", description="Compiled rich text for dense semantic embedding")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of enriched_text")

    def build_enriched_text(self) -> str:
        parts = [
            f"Отрасль: {self.industry}.",
            f"Бизнес-домен: {self.domain}.",
            f"Организация: {self.org_type}.",
            f"Бизнес-процессы: {self.business_process}.",
            f"Специфика работы: {self.work_scope}."
        ]
        if self.standards_and_tools:
            parts.append(f"Промышленные стандарты и профильные системы: {self.standards_and_tools}.")
        self.enriched_text = " ".join(parts)
        return self.enriched_text


class RawCandidate(BaseModel):
    id: int
    raw_domains: List[str]


class EnrichedCandidate(BaseModel):
    id: int
    cleaned_domains: List[str] = Field(default_factory=list)
    primary_industries: List[str] = Field(default_factory=list)
    domain_cards: List[DomainCard] = Field(default_factory=list)
    
    def get_summary_text(self) -> str:
        """Returns consolidated domain summary for quick inspection."""
        industries_str = ", ".join(self.primary_industries) if self.primary_industries else "не указано"
        domains_str = ", ".join(self.cleaned_domains)
        return f"[Кандидат #{self.id}] Отрасли: {industries_str} | Домены: {domains_str}"


class RecruiterQuery(BaseModel):
    raw_text: str
    extracted_role: Optional[str] = None
    extracted_seniority: Optional[str] = None
    extracted_stack: List[str] = Field(default_factory=list)
    target_industry: Optional[str] = Field(default=None, description="Global domain/industry filter, e.g., 'финтех'")
    target_business_domains: List[str] = Field(default_factory=list, description="Specific business domains, e.g., 'интеграция банковского ядра'")
    target_nuances: Optional[str] = Field(default=None, description="Specific domain context/details")


class CandidateScoreResult(BaseModel):
    candidate_id: int
    industry_matched: bool = True
    stage1_retrieval_score: float = 0.0
    stage2_rerank_score: float = 0.0
    domain_score: float = 0.0
    matched_domains: List[str] = Field(default_factory=list)
    best_matching_card: Optional[DomainCard] = None
    verdict_label: str = "НЕТ СОВПАДЕНИЯ"  # ТОЧНОЕ, БЛИЗКОЕ, СМЕЖНОЕ, ЧУЖОЕ
    reranker_reasoning: str = ""
    composite_score: float = 0.0  # Combined score with role, seniority, stack
