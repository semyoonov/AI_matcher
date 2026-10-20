import pytest
from pipeline import pipeline
from models import RecruiterQuery

def test_pipeline_initialization():
    pipeline.initialize()
    assert pipeline.is_indexed is True
    assert len(pipeline.candidates) == 117

def test_pipeline_case_1_banking_core():
    query = "Нужен автотестировщик senior-уровня с обязательным отличным знанием Kafka и обязательным опытом работы в финтехе. У кандидата желателен опыт с интеграцией банковского ядра"
    query_obj, results = pipeline.search_and_score(query, top_k=5)
    
    assert query_obj.target_industry == "финтех"
    assert "интеграция банковского ядра" in query_obj.target_business_domains
    assert len(results) == 5

    # Top-1 candidate must be #79
    top_cand = results[0]
    assert top_cand.candidate_id == 79
    assert top_cand.verdict_label == "ТОЧНОЕ СОВПАДЕНИЕ"
    assert top_cand.domain_score >= 0.90
    assert "интеграция банковского ядра" in top_cand.matched_domains

def test_pipeline_case_3_mes_iot():
    query = "Инженер автоматизации с практическим опытом работы в MES-системах и промышленном интернете вещей (IoT)"
    query_obj, results = pipeline.search_and_score(query, top_k=5)

    assert query_obj.target_industry == "промышленность"
    assert len(results) == 5

    # Top-1 candidate must be #20
    top_cand = results[0]
    assert top_cand.candidate_id == 20
    assert top_cand.verdict_label == "ТОЧНОЕ СОВПАДЕНИЕ"
    assert top_cand.domain_score >= 0.90
