import os
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from pipeline import pipeline
from config import settings
from domain_graph import domain_graph
from schemas import (
    MatchRequest, 
    MatchResponse, 
    CandidateResultSchema, 
    DomainCardSchema, 
    ParsedQuerySchema, 
    PresetItem, 
    HealthResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FastAPIServer")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager replacing deprecated startup/shutdown events."""
    logger.info("Initializing Domain Scoring Pipeline on startup...")
    pipeline.initialize()
    logger.info(f"Pipeline ready: {len(pipeline.candidates)} candidates indexed.")
    yield
    logger.info("Shutting down Domain Scoring Pipeline...")

app = FastAPI(
    title="AI Candidate Domain Matcher API",
    description="Пайплайн скоринга бизнес-доменов кандидата и вакансии (Граф связей + LLM Реранкер)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response

@app.get("/api/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint providing runtime statistics."""
    graph_dict = domain_graph.to_dict()
    return HealthResponse(
        status="ok",
        llm_provider=settings.get_active_provider(),
        indexed_candidates=len(pipeline.candidates),
        graph_nodes=len(graph_dict["nodes"]),
        graph_edges=len(graph_dict["edges"])
    )

@app.get("/api/presets", response_model=List[PresetItem], tags=["Matching"])
def get_presets():
    """Returns curated preset queries representing key scenarios from the assignment."""
    return [
        PresetItem(
            id="case_1",
            title="Senior QA + Kafka + Ядро банка",
            query="Нужен автотестировщик senior-уровня с обязательным отличным знанием Kafka и обязательным опытом работы в финтехе. У кандидата желателен опыт с интеграцией банковского ядра"
        ),
        PresetItem(
            id="case_2",
            title="Кредитный конвейер и скоринг",
            query="Разработчик с глубоким опытом в анализе кредитных заявок, кредитном скоринге и кредитных конвейерах"
        ),
        PresetItem(
            id="case_3",
            title="MES-системы и Промышленный IoT",
            query="Инженер автоматизации с практическим опытом работы в MES-системах и промышленном интернете вещей (IoT)"
        ),
        PresetItem(
            id="case_4",
            title="E-commerce корзина и промокоды",
            query="Middle backend разработчик для интернет-магазина с опытом в e-commerce, корзине покупателя и динамическом ценообразовании"
        )
    ]

QUERY_RESPONSE_CACHE: Dict[str, MatchResponse] = {}

@app.post("/api/match", response_model=MatchResponse, tags=["Matching"])
def match_candidates(req: MatchRequest):
    """
    Executes the full two-stage candidate scoring pipeline:
    1. Query entity analysis (Role, Seniority, Tech Stack, Industry, Domain)
    2. Vector & Graph pre-filtering (Top-50 candidates)
    3. Cross-Encoder LLM reranking (0.0 - 1.0 score with reasoning)
    4. Multi-factor composite scoring
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Текст запроса не может быть пустым")

    cache_key = f"{req.query.strip().lower()}__{req.top_k}__{req.industry_filter}"
    if cache_key in QUERY_RESPONSE_CACHE:
        return QUERY_RESPONSE_CACHE[cache_key]

    # Dynamic weights
    settings.WEIGHT_ROLE = req.role_weight
    settings.WEIGHT_SENIORITY = req.seniority_weight
    settings.WEIGHT_STACK = req.stack_weight
    settings.WEIGHT_DOMAIN = req.domain_weight

    query_obj, results = pipeline.search_and_score(
        query_text=req.query,
        top_k=req.top_k,
        apply_hard_industry_filter=req.industry_filter
    )

    formatted_results: List[CandidateResultSchema] = []
    for rank, r in enumerate(results, 1):
        best_card_schema = None
        if r.best_matching_card:
            best_card_schema = DomainCardSchema(
                domain=r.best_matching_card.domain,
                industry=r.best_matching_card.industry,
                org_type=r.best_matching_card.org_type,
                business_process=r.best_matching_card.business_process,
                work_scope=r.best_matching_card.work_scope,
                enriched_text=r.best_matching_card.enriched_text
            )

        formatted_results.append(CandidateResultSchema(
            rank=rank,
            candidate_id=r.candidate_id,
            domain_score=round(r.domain_score * 100, 1),
            composite_score=round(r.composite_score * 100, 1),
            verdict=r.verdict_label,
            reasoning=r.reranker_reasoning,
            matched_domains=r.matched_domains,
            best_card=best_card_schema
        ))

    parsed_schema = ParsedQuerySchema(
        role=query_obj.extracted_role or "не указана",
        seniority=query_obj.extracted_seniority or "любой",
        stack=query_obj.extracted_stack,
        industry=query_obj.target_industry or "все отрасли",
        domains=query_obj.target_business_domains,
        nuances=query_obj.target_nuances or ""
    )

    response = MatchResponse(
        parsed_query=parsed_schema,
        count=len(formatted_results),
        results=formatted_results
    )
    QUERY_RESPONSE_CACHE[cache_key] = response
    return response

# Static files mounting
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "Frontend static file is being generated..."})
    return FileResponse(str(index_file))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
