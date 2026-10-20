import pytest
from fastapi.testclient import TestClient
from server import app

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["indexed_candidates"] == 117
        assert data["graph_nodes"] > 0
        assert data["graph_edges"] > 0

def test_get_presets():
    with TestClient(app) as client:
        response = client.get("/api/presets")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 4
        assert any("банковск" in p["query"].lower() for p in data)

def test_match_case_1_core_banking():
    with TestClient(app) as client:
        payload = {
            "query": "Нужен автотестировщик senior-уровня с обязательным отличным знанием Kafka и обязательным опытом работы в финтехе. У кандидата желателен опыт с интеграцией банковского ядра",
            "top_k": 5,
            "industry_filter": False
        }
        response = client.post("/api/match", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5
        assert data["parsed_query"]["role"] == "автотестировщик"
        assert "Kafka" in data["parsed_query"]["stack"]

        # Candidate #79 must be #1
        top_candidate = data["results"][0]
        assert top_candidate["candidate_id"] == 79
        assert top_candidate["verdict"] == "ТОЧНОЕ СОВПАДЕНИЕ"
        assert top_candidate["domain_score"] >= 90.0

def test_match_empty_query():
    with TestClient(app) as client:
        response = client.post("/api/match", json={"query": "   "})
        assert response.status_code == 422 or response.status_code == 400
