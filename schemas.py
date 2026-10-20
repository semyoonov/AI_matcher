from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DomainCardSchema(BaseModel):
    domain: str = Field(description="Normalized domain name")
    industry: str = Field(description="Macro industry")
    org_type: str = Field(description="Organization type")
    business_process: str = Field(description="Business process")
    work_scope: str = Field(description="Specific work scope")
    standards_and_tools: Optional[str] = Field(default="", description="Промышленные стандарты, протоколы и профильные системы")
    enriched_text: str = Field(description="Full context representation")

class ParsedQuerySchema(BaseModel):
    role: str = Field(default="не указана")
    seniority: str = Field(default="любой")
    stack: List[str] = Field(default_factory=list)
    industry: str = Field(default="все отрасли")
    domains: List[str] = Field(default_factory=list)
    nuances: str = Field(default="")

class CandidateResultSchema(BaseModel):
    rank: int
    candidate_id: int
    domain_score: float = Field(description="Domain match percentage (0.0 to 100.0)")
    composite_score: float = Field(description="Overall composite score (0.0 to 100.0)")
    verdict: str = Field(description="Verdict label: ТОЧНОЕ СОВПАДЕНИЕ, БЛИЗКИЙ ДОМЕН, etc.")
    reasoning: str = Field(description="Cross-encoder LLM reasoning")
    matched_domains: List[str] = Field(default_factory=list)
    best_card: Optional[DomainCardSchema] = None

class MatchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Recruiter search query text")
    top_k: int = Field(default=3, ge=1, le=50, description="Number of top candidates to return")
    industry_filter: bool = Field(default=False, description="Apply strict industry filter")
    role_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    seniority_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    stack_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    domain_weight: float = Field(default=0.30, ge=0.0, le=1.0)

class MatchResponse(BaseModel):
    parsed_query: ParsedQuerySchema
    count: int
    results: List[CandidateResultSchema]

class PresetItem(BaseModel):
    id: str
    title: str
    query: str

class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    indexed_candidates: int
    graph_nodes: int
    graph_edges: int
