import pytest
from domain_graph import domain_graph, RelationType

def test_graph_initialization():
    assert domain_graph.graph.number_of_nodes() > 10
    assert domain_graph.graph.number_of_edges() > 10

def test_graph_synonyms():
    # 'промышленный интернет вещей' is synonym to 'промышленный iot'
    neighbors = domain_graph.get_domain_neighbors("промышленный iot", max_hops=1)
    assert "промышленный интернет вещей" in neighbors
    assert neighbors["промышленный интернет вещей"] >= 0.95

def test_graph_hierarchy():
    # 'интеграция банковского ядра' -> 'банкинг' -> 'финтех'
    neighbors = domain_graph.get_domain_neighbors("интеграция банковского ядра", max_hops=2)
    assert "банкинг" in neighbors
    assert "финтех" in neighbors

def test_graph_auto_expansion():
    initial_nodes = domain_graph.graph.number_of_nodes()
    domain_graph.auto_expand_with_card("новые платежные протоколы", "финтех")
    assert domain_graph.graph.has_node("новые платежные протоколы")
    assert domain_graph.graph.has_edge("новые платежные протоколы", "финтех")
