import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
import networkx as nx

logger = logging.getLogger("DomainGraph")

class RelationType:
    SYNONYM = "synonym"          # Weight: 1.0 (полный синоним или тождество)
    SUBDOMAIN_OF = "subdomain_of"# Weight: 0.85 (вложенность: иерархический домен -> макро-отрасль)
    RELATED = "related"          # Weight: 0.50 (смежная предметная область)

class DomainGraph:
    """
    Self-expanding Domain Knowledge Graph as presented in Slides 3, 6, and 11.
    Maintains hierarchical subdomains, synonyms, and related domains.
    Automatically grows as new resumes and queries are processed.
    """
    def __init__(self):
        self.graph = nx.Graph()
        self._initialize_core_taxonomy()

    def _initialize_core_taxonomy(self):
        """Initializes core seeds of macro-industries and known domain relations."""
        # Macro industries
        macro_nodes = [
            "финтех", "банкинг", "e-commerce", "ритейл", "медицина", 
            "промышленность", "телеком", "страхование", "госсектор", "логистика"
        ]
        for node in macro_nodes:
            self.graph.add_node(node, type="industry", label=node)

        # Pre-seed essential hierarchy from the assignment and slides
        # Fintech & Banking hierarchy
        self.add_relation("банкинг", "финтех", RelationType.SUBDOMAIN_OF, weight=0.90)
        self.add_relation("интеграция банковского ядра", "банкинг", RelationType.SUBDOMAIN_OF, weight=0.88)
        self.add_relation("банковские интеграции", "банкинг", RelationType.SUBDOMAIN_OF, weight=0.85)
        self.add_relation("банковские технологии", "банкинг", RelationType.SUBDOMAIN_OF, weight=0.85)
        self.add_relation("корпоративный онлайн-банкинг", "банкинг", RelationType.SUBDOMAIN_OF, weight=0.85)
        self.add_relation("интернет-банкинг", "корпоративный онлайн-банкинг", RelationType.SYNONYM, weight=0.95)
        self.add_relation("кредитный конвейер", "банкинг", RelationType.SUBDOMAIN_OF, weight=0.85)
        self.add_relation("анализ кредитных заявок", "кредитный конвейер", RelationType.SUBDOMAIN_OF, weight=0.90)
        self.add_relation("кредитное оформление", "кредитный конвейер", RelationType.SYNONYM, weight=0.92)
        self.add_relation("биллинг", "финтех", RelationType.SUBDOMAIN_OF, weight=0.75)
        self.add_relation("эквайринг", "финтех", RelationType.SUBDOMAIN_OF, weight=0.80)
        self.add_relation("платёжные системы", "финтех", RelationType.SUBDOMAIN_OF, weight=0.80)
        self.add_relation("платёжные операции", "платёжные системы", RelationType.SUBDOMAIN_OF, weight=0.90)

        # Cross-relations (related)
        self.add_relation("интеграция банковского ядра", "банковские интеграции", RelationType.RELATED, weight=0.80)
        self.add_relation("банковские интеграции", "платёжные системы", RelationType.RELATED, weight=0.60)
        self.add_relation("биллинг", "платёжные операции", RelationType.RELATED, weight=0.65)
        self.add_relation("страхование", "финтех", RelationType.RELATED, weight=0.45)

        # Industry IoT / MES
        self.add_relation("промышленный iot", "промышленность", RelationType.SUBDOMAIN_OF, weight=0.85)
        self.add_relation("промышленный интернет вещей", "промышленный iot", RelationType.SYNONYM, weight=1.0)
        self.add_relation("mes система", "промышленность", RelationType.SUBDOMAIN_OF, weight=0.85)
        self.add_relation("mes система", "промышленный iot", RelationType.RELATED, weight=0.60)

        # Entity Linking: Standards, Protocols & Vendor Software
        self.add_relation("абс", "интеграция банковского ядра", RelationType.SYNONYM, weight=0.95)
        self.add_relation("cbs", "интеграция банковского ядра", RelationType.SYNONYM, weight=0.95)
        self.add_relation("цфт-банк", "интеграция банковского ядра", RelationType.SUBDOMAIN_OF, weight=0.90)
        self.add_relation("диасофт", "интеграция банковского ядра", RelationType.SUBDOMAIN_OF, weight=0.90)
        self.add_relation("iso 20022", "интеграция банковского ядра", RelationType.RELATED, weight=0.85)
        self.add_relation("swift", "банковские интеграции", RelationType.RELATED, weight=0.85)
        self.add_relation("бки", "кредитный конвейер", RelationType.RELATED, weight=0.90)
        self.add_relation("нбки", "бки", RelationType.SYNONYM, weight=0.95)
        self.add_relation("pci dss", "платёжные системы", RelationType.RELATED, weight=0.85)
        self.add_relation("scada", "промышленный iot", RelationType.RELATED, weight=0.85)
        self.add_relation("opc ua", "промышленный iot", RelationType.RELATED, weight=0.88)
        self.add_relation("mqtt", "промышленный iot", RelationType.RELATED, weight=0.85)
        self.add_relation("емиас", "медицина", RelationType.SUBDOMAIN_OF, weight=0.90)
        self.add_relation("hl7", "медицина", RelationType.RELATED, weight=0.85)
        self.add_relation("смэв", "госсектор", RelationType.SUBDOMAIN_OF, weight=0.90)
        self.add_relation("есиа", "госсектор", RelationType.SUBDOMAIN_OF, weight=0.90)

        # CRM & Enterprise Integration hierarchy
        self.add_relation("crm", "it / бизнес", RelationType.SUBDOMAIN_OF, weight=0.85)
        self.add_relation("crm система", "crm", RelationType.SYNONYM, weight=0.98)
        self.add_relation("crm интеграция", "crm", RelationType.SUBDOMAIN_OF, weight=0.95)
        self.add_relation("интеграция с crm", "crm интеграция", RelationType.SYNONYM, weight=1.0)
        self.add_relation("интеграция с crm-системами", "crm интеграция", RelationType.SYNONYM, weight=1.0)
        self.add_relation("банковская crm", "crm", RelationType.SUBDOMAIN_OF, weight=0.90)
        self.add_relation("корпоративный crm", "crm", RelationType.SUBDOMAIN_OF, weight=0.90)
        self.add_relation("банковский клиентский crm", "crm", RelationType.SUBDOMAIN_OF, weight=0.90)
        self.add_relation("банковские интеграционные шлюзы", "банковские интеграции", RelationType.SUBDOMAIN_OF, weight=0.95)
        self.add_relation("платёжный шлюз", "платёжные системы", RelationType.SUBDOMAIN_OF, weight=0.95)

    def normalize(self, name: str) -> str:
        return name.strip().lower()

    def add_node_if_missing(self, domain: str, node_type: str = "domain"):
        norm = self.normalize(domain)
        if norm not in self.graph:
            self.graph.add_node(norm, type=node_type, label=domain)

    def add_relation(self, source: str, target: str, rel_type: str, weight: float = 0.7):
        s_norm = self.normalize(source)
        t_norm = self.normalize(target)
        self.add_node_if_missing(s_norm)
        self.add_node_if_missing(t_norm)
        self.graph.add_edge(s_norm, t_norm, rel_type=rel_type, weight=weight)

    def auto_expand_with_card(self, domain: str, industry: str):
        """
        Self-expansion logic (Slide 6):
        When a new domain card is encountered, connects it to its parent industry
        and detects potential synonyms/hierarchies.
        """
        d_norm = self.normalize(domain)
        i_norm = self.normalize(industry)
        
        self.add_node_if_missing(d_norm, node_type="domain")
        self.add_node_if_missing(i_norm, node_type="industry")
        
        if not self.graph.has_edge(d_norm, i_norm):
            self.add_relation(d_norm, i_norm, RelationType.SUBDOMAIN_OF, weight=0.75)

    def get_domain_neighbors(self, domain: str, max_hops: int = 1) -> Dict[str, float]:
        """
        Finds adjacent and parent/child domains in the graph with proximity scores.
        Returns dict: {domain_name: proximity_score} where 1.0 is exact match.
        """
        norm = self.normalize(domain)
        results = {norm: 1.0}
        
        if norm not in self.graph:
            return results

        # BFS up to max_hops
        visited = {norm: 1.0}
        queue = [(norm, 1.0, 0)]

        while queue:
            current, current_weight, hop = queue.pop(0)
            if hop >= max_hops:
                continue

            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph.get_edge_data(current, neighbor)
                edge_weight = edge_data.get("weight", 0.7)
                combined_weight = current_weight * edge_weight

                if neighbor not in visited or visited[neighbor] < combined_weight:
                    visited[neighbor] = combined_weight
                    queue.append((neighbor, combined_weight, hop + 1))

        return visited

    def compute_graph_proximity(self, query_domains: List[str], candidate_domains: List[str]) -> float:
        """
        Computes maximum graph-based topological semantic proximity between
        query domains and candidate domains.
        """
        if not query_domains or not candidate_domains:
            return 0.0

        max_proximity = 0.0
        for q_dom in query_domains:
            expanded = self.get_domain_neighbors(q_dom, max_hops=2)
            for c_dom in candidate_domains:
                c_norm = self.normalize(c_dom)
                if c_norm in expanded:
                    score = expanded[c_norm]
                    if score > max_proximity:
                        max_proximity = score

        return max_proximity

    def to_dict(self) -> Dict[str, Any]:
        """Serializes graph to dict for visualization."""
        nodes = [{"id": n, "label": self.graph.nodes[n].get("label", n), "type": self.graph.nodes[n].get("type", "domain")} for n in self.graph.nodes()]
        edges = [
            {
                "source": u, 
                "target": v, 
                "rel_type": data.get("rel_type", "related"), 
                "weight": data.get("weight", 0.5)
            } 
            for u, v, data in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

domain_graph = DomainGraph()
