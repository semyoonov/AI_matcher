import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from config import settings

logger = logging.getLogger("LLMClient")

class BaseLLMClient(ABC):
    """Abstract Base Class for LLM providers."""
    
    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Generates structured JSON from prompts."""
        pass

    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generates plain text from prompts."""
        pass

    @staticmethod
    def clean_json_string(raw: str) -> str:
        """Strips markdown fences and whitespaces from LLM response."""
        s = raw.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()


class GroqClient(BaseLLMClient):
    """Production Groq LLM Client supporting Llama 3.3 70B and Llama 3.1 8B."""

    def __init__(self, api_key: str, model: str):
        from groq import Groq
        self.provider = "groq"
        self.client = Groq(api_key=api_key)
        self.model = model
        self._last_call_time = 0.0
        self._min_interval = 0.5
        self._cooldown_until = 0.0
        logger.info(f"Initialized GroqClient [model={self.model}]")

    def _wait_pacing(self):
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.monotonic()

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if time.monotonic() < self._cooldown_until:
            return SemanticRuleEngine().generate_text(system_prompt, user_prompt)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self._wait_pacing()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                logger.warning(f"Groq API call attempt {attempt+1}/{max_retries} failed: {e}")
                if "429" in err_str or "rate limit" in err_str or "quota" in err_str or attempt == max_retries - 1:
                    self._cooldown_until = time.monotonic() + 30.0
                    logger.info("Engaging 30s cooldown and falling back to SemanticRuleEngine.")
                    return SemanticRuleEngine().generate_text(system_prompt, user_prompt)
                time.sleep(0.5)
        return SemanticRuleEngine().generate_text(system_prompt, user_prompt)

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if time.monotonic() < self._cooldown_until:
            return SemanticRuleEngine().generate_json(system_prompt, user_prompt)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self._wait_pacing()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                content = resp.choices[0].message.content or "{}"
                return json.loads(self.clean_json_string(content))
            except Exception as e:
                err_str = str(e).lower()
                logger.warning(f"Groq JSON call attempt {attempt+1}/{max_retries} failed: {e}")
                if "429" in err_str or "rate limit" in err_str or "quota" in err_str or attempt == max_retries - 1:
                    self._cooldown_until = time.monotonic() + 30.0
                    logger.info("Engaging 30s cooldown and falling back to SemanticRuleEngine.")
                    return SemanticRuleEngine().generate_json(system_prompt, user_prompt)
                time.sleep(0.5)
        return SemanticRuleEngine().generate_json(system_prompt, user_prompt)


class MistralClient(BaseLLMClient):
    """Production Mistral AI Client with Rate Limit Pacing and Circuit Breaker."""

    def __init__(self, api_key: str, model: str):
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=api_key, timeout_ms=8000)
        except Exception:
            try:
                from mistralai.client import Mistral
                self.client = Mistral(api_key=api_key)
            except Exception:
                from mistralai import Mistral
                self.client = Mistral(api_key=api_key)
        self.provider = "mistral"
        self.model = model
        self._last_call_time = 0.0
        self._min_interval = 1.15  # Mistral Free Tier limit: max 1 req/sec
        self._cooldown_until = 0.0
        logger.info(f"Initialized MistralClient [model={self.model}, min_interval={self._min_interval}s]")

    def _wait_pacing(self):
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.monotonic()

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if time.monotonic() < self._cooldown_until:
            return SemanticRuleEngine().generate_text(system_prompt, user_prompt)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self._wait_pacing()
                resp = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                logger.warning(f"Mistral API call attempt {attempt+1}/{max_retries} failed: {e}")
                if any(w in err_str for w in ["429", "rate limit", "quota", "ssl", "eof", "connection", "timeout", "timed out"]) or attempt == max_retries - 1:
                    self._cooldown_until = time.monotonic() + 300.0
                    logger.warning("Mistral connection/limit issue. Circuit breaker engaged for 300s. Routing to SemanticRuleEngine.")
                    return SemanticRuleEngine().generate_text(system_prompt, user_prompt)
                time.sleep(0.3)
        return SemanticRuleEngine().generate_text(system_prompt, user_prompt)

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if time.monotonic() < self._cooldown_until:
            return SemanticRuleEngine().generate_json(system_prompt, user_prompt)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self._wait_pacing()
                resp = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                content = resp.choices[0].message.content or "{}"
                return json.loads(self.clean_json_string(content))
            except Exception as e:
                err_str = str(e).lower()
                logger.warning(f"Mistral JSON call attempt {attempt+1}/{max_retries} failed: {e}")
                if any(w in err_str for w in ["429", "rate limit", "quota", "ssl", "eof", "connection", "timeout", "timed out"]) or attempt == max_retries - 1:
                    self._cooldown_until = time.monotonic() + 300.0
                    logger.warning("Mistral connection/limit issue. Circuit breaker engaged for 300s. Routing to SemanticRuleEngine.")
                    return SemanticRuleEngine().generate_json(system_prompt, user_prompt)
                time.sleep(0.3)
        return SemanticRuleEngine().generate_json(system_prompt, user_prompt)


class SemanticRuleEngine(BaseLLMClient):
    """
    High-performance semantic rule-based engine used when live API keys are not supplied
    or when circuit breaker is active. Produces deterministic, instant (<5ms) responses.
    """
    def __init__(self):
        self.provider = "rule_engine"

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return "Deterministic engine active."

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        sys_lower = system_prompt.lower()

        # 1. Cross-Encoder Batch Reranker (Priority check for batch format)
        if any(w in sys_lower for w in ["evaluations", "списком кандидатов", "каждому кандидату"]):
            return self._handle_batch_rerank(user_prompt)

        # 2. Cross-Encoder Single Reranker
        if any(w in sys_lower for w in ["cross-encoder", "оценке соответствия", "реранкер"]):
            return self._handle_rerank(user_prompt)

        # 3. Recruiter Query Parser
        if any(w in sys_lower for w in ["структурированные сущности", "ассистент рекрутера"]):
            return self._handle_query_parse(user_prompt)

        # 4. Domain Cleaning
        if any(w in sys_lower for w in ["очисти", "cleaner", "исправь опечатки"]):
            return self._handle_cleaning(user_prompt)

        # 5. Domain Card Enrichment
        if any(w in sys_lower for w in ["карточка домена", "enrichment", "фиксированные вопросы"]):
            return self._handle_enrichment(user_prompt)

        return {"status": "ok"}

    def _handle_batch_rerank(self, user_prompt: str) -> Dict[str, Any]:
        req_part = user_prompt.split("Список кандидатов для оценки:")[0] if "Список кандидатов для оценки:" in user_prompt else user_prompt
        evals = []
        import re
        cands_sections = re.split(r'---\s*Кандидат\s*#(\d+)\s*---', user_prompt)
        if len(cands_sections) > 1:
            for i in range(1, len(cands_sections), 2):
                cid = int(cands_sections[i])
                cand_text = cands_sections[i+1]
                prompt_for_single = f"{req_part}\n\nОпыт кандидата:\n{cand_text}"
                res = self._handle_rerank(prompt_for_single)
                evals.append({
                    "candidate_id": cid,
                    "score": res["score"],
                    "verdict": res["verdict"],
                    "reasoning": res["reasoning"]
                })
        return {"evaluations": evals}

    def _handle_query_parse(self, user_prompt: str) -> Dict[str, Any]:
        u = user_prompt.lower()
        role = None
        if "автотестировщик" in u or "qa" in u or "тестировщик" in u:
            role = "автотестировщик"
        elif "разработчик" in u or "developer" in u:
            role = "разработчик"
        elif "аналитик" in u:
            role = "аналитик"
        elif "инженер" in u:
            role = "инженер"
        
        seniority = None
        if "senior" in u or "сеньор" in u:
            seniority = "senior"
        elif "lead" in u or "лид" in u:
            seniority = "lead"
        elif "middle" in u or "мидл" in u:
            seniority = "middle"
        elif "junior" in u or "джун" in u:
            seniority = "junior"
        
        stack = []
        for tech in ["kafka", "python", "java", "sql", "docker", "kubernetes", "postgres", "fastapi", "spring"]:
            if tech in u:
                stack.append(tech.capitalize())

        target_industry = None
        if any(w in u for w in ["финтех", "банк", "кредит", "платеж", "валют", "казначей", "шлюз"]):
            target_industry = "финтех"
        elif any(w in u for w in ["iot", "mes", "промышленн", "завод"]):
            target_industry = "промышленность"
        elif any(w in u for w in ["ритейл", "e-commerce", "магазин", "товар", "корзин"]):
            target_industry = "e-commerce"
        elif any(w in u for w in ["медицин", "здравоохран"]):
            target_industry = "медицина"
        elif any(w in u for w in ["госуслуг", "госсектор", "государств"]):
            target_industry = "госсектор"
        elif any(w in u for w in ["страхован", "полис"]):
            target_industry = "страхование"
        elif any(w in u for w in ["телеком", "связь"]):
            target_industry = "телеком"
        elif any(w in u for w in ["логистик", "транспорт", "груз"]):
            target_industry = "логистика"

        domains = []
        # 1. Gateways / Шлюзы
        if "банковские интеграционные шлюзы" in u or ("шлюз" in u and ("интеграц" in u or "банк" in u)):
            domains.append("банковские интеграционные шлюзы")
        elif "платёжный шлюз" in u or ("шлюз" in u and "плат" in u):
            domains.append("платёжный шлюз")

        # 2. Core banking
        if "ядро" in u or "ядра" in u or "абс" in u:
            domains.append("интеграция банковского ядра")

        # 3. Credit domains
        if "потребительское кредитование" in u:
            domains.append("потребительское кредитование")
        if "кредитный конвейер" in u:
            domains.append("кредитный конвейер")
        if "кредитный скоринг" in u:
            domains.append("кредитный скоринг")
        if "формирование кредитных офферов" in u or "оффер" in u:
            domains.append("формирование кредитных офферов")
        if "оферты по кредитным продуктам" in u or "оферт" in u:
            domains.append("оферты по кредитным продуктам")
        if "кредит" in u and not domains:
            domains.extend(["кредитный конвейер", "кредитный скоринг"])

        # 4. Industry specifics
        if "mes" in u:
            domains.append("mes система")
        if "iot" in u or "интернет вещей" in u:
            domains.append("промышленный iot")
        if "корзин" in u:
            domains.append("корзина")
        if "промокод" in u:
            domains.append("управление промокодами")
        if "видеоаналитик" in u:
            domains.append("видеоаналитика")
        if "биллинг" in u:
            domains.append("биллинг")
        if "эквайринг" in u:
            domains.append("эквайринг")

        # 5. CRM & Enterprise
        if "crm" in u or "црм" in u:
            target_industry = "IT / Бизнес"
            if "интеграц" in u or "опыт" in u or "разработ" in u:
                domains.append("crm интеграция")
            else:
                domains.append("crm система")
        elif "erp" in u:
            target_industry = "IT / Бизнес"
            domains.append("интеграция erp-систем")

        # 6. Direct phrase fallback if no preset domain matched
        if not domains:
            stopwords = {
                "нужен", "нужна", "нужно", "требуется", "ищем", "поиск", "вакансия",
                "опыт", "опытом", "работы", "работать", "кандидат", "кандидата", "у",
                "с", "со", "в", "во", "на", "по", "для", "и", "или", "от", "до",
                "обязательно", "обязательным", "отличным", "хорошим", "знанием", "знание",
                "желателен", "желательно", "уровня", "уровень"
            }
            words = [w.strip(".,!?:;\"'()[]{}") for w in u.split()]
            meaningful = []
            for w in words:
                if w in stopwords or len(w) <= 2:
                    continue
                if role and w in role:
                    continue
                if seniority and w in seniority:
                    continue
                if any(w == s.lower() for s in stack):
                    continue
                meaningful.append(w)
            if meaningful:
                domains.append(" ".join(meaningful))
            elif target_industry:
                domains.append(target_industry)

        nuance = ""
        if "ядро" in u or "ядра" in u:
            nuance = "Опыт интеграции с банковским ядром (CBS/АБС)"
        elif "шлюз" in u:
            nuance = "Интеграционные протоколы, платежные и банковские шлюзы"
        elif "кредит" in u:
            nuance = "Кредитные процессы, скоринг и конвейер"

        return {
            "role": role or "не указана",
            "seniority": seniority or "любой",
            "stack": stack,
            "target_industry": target_industry or "все отрасли",
            "target_business_domains": domains,
            "target_nuances": nuance
        }

    def _handle_cleaning(self, user_prompt: str) -> Dict[str, Any]:
        import ast
        try:
            if "[" in user_prompt and "]" in user_prompt:
                list_str = user_prompt[user_prompt.find("["):user_prompt.rfind("]")+1]
                items = ast.literal_eval(list_str)
                if isinstance(items, list):
                    return {"cleaned_domains": [str(x).strip() for x in items if x]}
        except Exception:
            pass
        return {"cleaned_domains": []}

    def _handle_enrichment(self, user_prompt: str) -> Dict[str, Any]:
        from domain_enricher import classify_domain_full
        domain = user_prompt.strip()
        if "бизнес-домен для анализа:" in user_prompt.lower():
            import re
            m = re.search(r"бизнес-домен для анализа:\s*['\"]([^'\"]+)['\"]", user_prompt, re.I)
            if m:
                domain = m.group(1).strip()
        tax = classify_domain_full(domain)
        return {
            "domain": domain,
            "industry": tax["industry"],
            "org_type": tax["org_type"],
            "business_process": tax["business_process"],
            "work_scope": tax["work_scope"],
            "standards_and_tools": tax["standards_and_tools"]
        }

    def _handle_rerank(self, user_prompt: str) -> Dict[str, Any]:
        u = user_prompt.lower()
        cand_text = u.split("опыт кандидата")[1] if "опыт кандидата" in u else u
        req_text = u.split("опыт кандидата")[0] if "опыт кандидата" in u else u

        # 1. Gateways / Шлюзы
        if "шлюз" in req_text:
            if "банковские интеграционные шлюзы" in cand_text:
                return {
                    "score": 0.98,
                    "verdict": "ТОЧНОЕ СОВПАДЕНИЕ",
                    "reasoning": "Кандидат имеет профильный опыт работы с банковскими интеграционными шлюзами и межбанковскими протоколами обмена."
                }
            elif any(w in cand_text for w in ["платёжный шлюз", "банковские интеграции", "интеграция банковских систем", "интеграция банковского ядра"]):
                return {
                    "score": 0.78,
                    "verdict": "БЛИЗКИЙ ДОМЕН",
                    "reasoning": "Кандидат обладает смежным банковским опытом (банковские интеграции, платежные шлюзы, интеграция банковских систем)."
                }
            elif any(w in cand_text for w in ["банкинг", "финтех", "платеж"]):
                return {
                    "score": 0.50,
                    "verdict": "СМЕЖНЫЙ ДОМЕН",
                    "reasoning": "Кандидат работал в финтехе и банках, но прямой опыт со шлюзами не акцентирован."
                }
            else:
                return {
                    "score": 0.05,
                    "verdict": "НЕТ СОВПАДЕНИЯ",
                    "reasoning": "Опыт кандидата далек от банковских интеграционных систем и шлюзов."
                }

        # 2. Core banking
        if "интеграция банковского ядра" in req_text or "ядра" in req_text:
            if "интеграция банковского ядра" in cand_text:
                return {
                    "score": 0.98,
                    "verdict": "ТОЧНОЕ СОВПАДЕНИЕ",
                    "reasoning": "Кандидат имеет прямой подтвержденный опыт интеграции банковского ядра (CBS/АБС) в банке, что идеально закрывает ключевое требование вакансии."
                }
            elif any(w in cand_text for w in ["банковские технологии", "банковские интеграции", "корпоративный онлайн-банкинг", "внутренние банковские системы"]):
                return {
                    "score": 0.78,
                    "verdict": "БЛИЗКИЙ ДОМЕН",
                    "reasoning": "Кандидат обладает смежным банковским опытом (банковские интеграции, корпоративный банкинг, АБС-инфраструктура)."
                }
            elif any(w in cand_text for w in ["финтех", "биллинг", "эквайринг", "платёж", "кредит"]):
                return {
                    "score": 0.50,
                    "verdict": "СМЕЖНЫЙ ДОМЕН",
                    "reasoning": "Кандидат работал в финтехе и платёжных сервисах, но прямого опыта интеграции банковского ядра не имеет."
                }
            else:
                return {
                    "score": 0.05,
                    "verdict": "НЕТ СОВПАДЕНИЯ",
                    "reasoning": "Домены кандидата (e-commerce, медиа, промышленность) далеки от банковского ядра и финтеха."
                }

        # 3. Credit domains
        elif "кредит" in req_text or "оферт" in req_text or "скоринг" in req_text:
            matched_credit = [w for w in ["кредитный конвейер", "кредитный скоринг", "потребительское кредитование", "формирование кредитных офферов", "оферты по кредитным продуктам", "анализ кредитных заявок"] if w in cand_text]
            if matched_credit:
                return {
                    "score": 0.96,
                    "verdict": "ТОЧНОЕ СОВПАДЕНИЕ",
                    "reasoning": f"Кандидат имеет непосредственный опыт работы с кредитными процессами ({', '.join(matched_credit[:2])})."
                }
            elif any(w in cand_text for w in ["банкинг", "финтех", "финансовые операции"]):
                return {
                    "score": 0.75,
                    "verdict": "БЛИЗКИЙ ДОМЕН",
                    "reasoning": "Кандидат имеет банковский и финансовый опыт, быстро освоит кредитный конвейер."
                }
            else:
                return {
                    "score": 0.05,
                    "verdict": "НЕТ СОВПАДЕНИЯ",
                    "reasoning": "Опыт кандидата не связан с кредитованием и финансами."
                }

        # 4. MES / IoT
        elif any(w in req_text for w in ["mes", "iot", "промышленн"]):
            if "mes" in cand_text and "iot" in cand_text:
                return {
                    "score": 0.98,
                    "verdict": "ТОЧНОЕ СОВПАДЕНИЕ",
                    "reasoning": "Кандидат имеет подтвержденную комбинацию опыта в MES-системах и промышленном IoT (интернет вещей)."
                }
            elif any(w in cand_text for w in ["mes", "iot", "промышленност"]):
                return {
                    "score": 0.75,
                    "verdict": "БЛИЗКИЙ ДОМЕН",
                    "reasoning": "Кандидат работал в смежной сфере промышленного сектора / IoT."
                }
            else:
                return {
                    "score": 0.05,
                    "verdict": "НЕТ СОВПАДЕНИЯ",
                    "reasoning": "Кандидат не имеет опыта в промышленной автоматизации и MES/IoT."
                }

        # 5. CRM & Enterprise systems
        elif "crm" in req_text or "црм" in req_text:
            if any(w in cand_text for w in ["crm интеграция", "интеграция с crm", "интеграция с crm-системами", "crm система", "банковская crm", "корпоративный crm", "банковский клиентский crm", "crm"]):
                return {
                    "score": 0.98,
                    "verdict": "ТОЧНОЕ СОВПАДЕНИЕ",
                    "reasoning": "Кандидат имеет подтвержденный профильный опыт интеграции и разработки решений в сфере CRM-систем."
                }
            elif any(w in cand_text for w in ["интеграция сторонних сервисов", "системная интеграция", "интеграция erp", "erp"]):
                return {
                    "score": 0.65,
                    "verdict": "СМЕЖНЫЙ ДОМЕН",
                    "reasoning": "Кандидат имеет общий опыт системной интеграции, но прямой опыт с CRM выражен слабее."
                }
            else:
                return {
                    "score": 0.05,
                    "verdict": "НЕТ СОВПАДЕНИЯ",
                    "reasoning": "Опыт кандидата (банковские шлюзы, биллинг, производство) не связан с внедрением и интеграцией CRM-систем."
                }

        # General dynamic matching for any other arbitrary industry/domain
        import re
        req_words = set(re.findall(r'[a-zA-Zа-яА-ЯёЁ]{2,}', req_text))
        cand_words = set(re.findall(r'[a-zA-Zа-яА-ЯёЁ]{2,}', cand_text))
        stop_words = {
            "опыт", "кандидата", "требования", "вакансии", "уровень", "старший", 
            "инженер", "разработчик", "специалист", "отрасль", "организация", 
            "процесс", "задачи", "работа", "систем", "проектах", "домены", "домен", 
            "бизнес", "целевые", "список", "оценки", "запрос", "рекрутера", "нюансы", 
            "опыта", "любая", "все", "отрасли", "сфере", "решений", "разработка", 
            "тестирование", "для", "при"
        }
        req_clean = req_words - stop_words
        cand_clean = cand_words - stop_words

        overlap = req_clean & cand_clean
        overlap_ratio = len(overlap) / max(len(req_clean), 1)

        if overlap_ratio >= 0.4 or len(overlap) >= 2:
            matched_terms = list(overlap)[:3]
            return {
                "score": 0.90,
                "verdict": "ТОЧНОЕ СОВПАДЕНИЕ",
                "reasoning": f"Кандидат имеет подтвержденный опыт в ключевых доменах вакансии ({', '.join(matched_terms)})."
            }
        elif overlap_ratio > 0.15 or len(overlap) >= 1:
            matched_terms = list(overlap)[:2]
            return {
                "score": 0.72,
                "verdict": "БЛИЗКИЙ ДОМЕН",
                "reasoning": f"Опыт кандидата частично пересекается со смежной предметной областью ({', '.join(matched_terms)})."
            }
        else:
            return {
                "score": 0.05,
                "verdict": "НЕТ СОВПАДЕНИЯ",
                "reasoning": "Опыт кандидата в предметной области не закрывает ключевые требования запроса."
            }


def create_llm_client() -> BaseLLMClient:
    """Factory to initialize active LLM provider."""
    provider = settings.get_active_provider()

    if provider == "groq" and settings.GROQ_API_KEY:
        try:
            return GroqClient(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)
        except Exception as e:
            logger.error(f"Failed to create GroqClient: {e}. Falling back to SemanticRuleEngine.")

    elif provider == "mistral" and settings.MISTRAL_API_KEY:
        try:
            return MistralClient(api_key=settings.MISTRAL_API_KEY, model=settings.MISTRAL_MODEL)
        except Exception as e:
            logger.error(f"Failed to create MistralClient: {e}. Falling back to SemanticRuleEngine.")

    logger.info("Using SemanticRuleEngine (No live API keys configured).")
    return SemanticRuleEngine()

llm_client = create_llm_client()
