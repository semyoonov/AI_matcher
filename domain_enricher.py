import logging
from typing import List, Dict, Optional, Any
from models import RawCandidate, EnrichedCandidate, DomainCard
from domain_cleaner import clean_candidate_domains
from llm_client import llm_client

logger = logging.getLogger("DomainEnricher")

DOMAIN_CARD_SYSTEM_PROMPT = """Ты AI-эксперт по анализу предметных областей в IT и бизнесе.
Твоя задача — проанализировать бизнес-домен кандидата и сопутствующий контекст его резюме и построить структурированную КАРТОЧКУ ДОМЕНА (Domain Card) с глубоким контекстом:
1. industry: макро-отрасль, к которой относится домен (например: "финтех / банкинг", "e-commerce и ритейл", "медицина и здравоохранение", "промышленность и IoT", "телеком и медиа", "госсектор и налоги", "логистика и транспорт", "страхование", "AdTech и Web3").
2. org_type: тип организации, где этот опыт применяется (например: "банк", "страховая компания", "маркетплейс", "производственное предприятие", "телеком-оператор", "государственный орган").
3. business_process: ключевой бизнес-процесс, описывающий этот опыт (например: "обмен данными между системами", "клиринг", "скоринг кредитов", "биллинг", "телеметрия оборудования").
4. work_scope: конкретная специфика задач в рамках этого домена (например: "интеграция банковского ядра (АБС) с внешними сервисами", "процессинг транзакций", "маршрутизация платежей").
5. standards_and_tools: промышленные стандарты, вендорские системы и протоколы предметной области (Entity Linking, например: "АБС ЦФТ-Банк, ISO 20022, SWIFT, PCI DSS", "БКИ, FICO score", "MQTT, OPC UA, SCADA, ISA-95", "HL7, DICOM", "СМЭВ, ЕСИА").

Ответь строго в формате JSON:
{
  "domain": "название домена",
  "industry": "макро-отрасль",
  "org_type": "тип организации",
  "business_process": "бизнес-процесс",
  "work_scope": "в чём заключалась работа",
  "standards_and_tools": "промышленные стандарты, вендоры и протоколы"
}
"""

def classify_domain_full(d: str, context: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Deterministic domain taxonomy engine covering 14 key IT industries.
    Guarantees deep domain differentiation even when live LLM APIs hit rate limits.
    """
    u = d.strip().lower().replace('ё', 'е')
    ctx = " ".join(context or []).lower().replace('ё', 'е')
    
    # 1. Fintech, Banking, Payments & Investments
    if any(k in u for k in ["банк", "кредит", "платеж", "финтех", "биллинг", "эквайринг", "казначей", 
                           "ностро", "валют", "депозит", "инвестиц", "факторинг", "займ", "микрофинанс", 
                           "swift", "fintech", "kyc", "денежн", "финанс", "трейдинг", "транзакцион",
                           "шлюз", "ядро", "абс", "cbs", "ностро-счет", "перевод", "пенсионн", "календарь"]):
        bp = "управление расчетными и кредитными операциями"
        ws = f"проектирование и сопровождение финансовых систем в сфере {d}"
        st = "PCI DSS, ISO 20022"
        org = "банк / финтех-компания"
        if any(k in u for k in ["шлюз", "интеграц", "ядро", "абс"]):
            bp = "интеграция банковского ядра, маршрутизация межбанковских платежей и шлюзов"
            st = "АБС (ЦФТ, Диасофт), SWIFT, ISO 20022"
            ws = "разработка интеграционных коннекторов и протоколов обмена банковских систем"
        elif any(k in u for k in ["кредит", "скоринг", "конвейер", "оффер", "оферт"]):
            bp = "кредитный конвейер, автоматизация скоринга и одобрения заявок"
            st = "БКИ, FICO Score, скоринговые модели"
            ws = "автоматизация процессов кредитования и кредитного анализа"
        elif any(k in u for k in ["казначей", "валют", "ностро"]):
            bp = "казначейские операции, валютный контроль и реконсиляция счетов"
            st = "Nostro/Vostro, валютный контроль ЦБ РФ"
            ws = "автоматизация казначейских расчетов и учета"
        elif any(k in u for k in ["платеж", "эквайринг", "биллинг", "перевод"]):
            bp = "клиринг, биллинг, транзакционный процессинг и эквайринг"
            st = "PCI DSS, 3-D Secure, Fast Payment System (СБП)"
            ws = "обработка платежных транзакций и интеграция платежных шлюзов"
        return {
            "industry": "финтех / банкинг",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 2. Insurance (Иншуртех и страхование)
    if any(k in u for k in ["страхов", "полис", "страховател", "премий", "премия"]):
        bp = "расчет страховых премий, оформление полисов и урегулирование убытков"
        ws = f"автоматизация страховых сервисов и личных кабинетов ({d})"
        st = "АИС ОСАГО, страховой андеррайтинг"
        org = "страховая компания / иншуртех"
        return {
            "industry": "страхование",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 3. CRM, Customer Management & Enterprise B2B Systems
    if any(k in u for k in ["crm", "управление клиент", "клиентск", "продаж", "воронка", "персонал", "hr", "подбор", "взаимоотношен", "кадров"]):
        bp = "управление жизненным циклом клиента (CLM) и автоматизация воронки продаж"
        ws = f"разработка, интеграция и внедрение CRM-модулей в сфере {d}"
        st = "CRM API, Bitrix24, Creatio, Salesforce, 1C"
        org = "корпоративный B2B-сектор / вендор CRM"
        if any(k in u for k in ["hr", "персонал", "подбор", "кадров"]):
            bp = "автоматизация рекрутинга, онбординга и кадрового учета (HR-tech)"
            st = "HRMS, ATS, 1C:ЗУП"
            ws = "автоматизация процессов найма и управления кадровым потенциалом"
        return {
            "industry": "CRM и корпоративные сервисы",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 4. E-Commerce, Retail & FMCG
    if any(k in u for k in ["ритейл", "e-commerce", "магазин", "товар", "корзин", "промокод", "заказ", 
                           "доставк", "ценообразов", "покупател", "food", "дистрибуц", "одежд", "бытовой", 
                           "ретейл", "мебель", "книжн", "маркетплейс", "horeca", "бытов", "алкогол", "мерчант",
                           "коммерц", "торговл", "продукт", "кухон", "продавец", "продавц", "билет"]):
        bp = "управление каталогом товаров, оформление заказов и корзина покупателя"
        ws = f"разработка компонентов электронной коммерции в области {d}"
        st = "ERP, 1C, WMS, REST/GraphQL e-commerce API"
        org = "интернет-магазин / маркетплейс / розничная сеть"
        if any(k in u for k in ["промокод", "ценообразов", "скидк", "бонус", "лояльност"]):
            bp = "динамическое ценообразование, программы лояльности и применение промокодов"
            st = "Loyalty Engine, Pricing Matrix, Promo Service"
            ws = "разработка механизмов дисконтирования и маркетинговых акций"
        elif any(k in u for k in ["склад", "дистрибуц", "поставк"]):
            bp = "складской учет, распределительная логистика и цепочки поставок"
            st = "WMS, ERP-системы дистрибьютора, 1C:Склад"
            ws = "автоматизация складского учета и отгрузки товаров"
        return {
            "industry": "e-commerce и ритейл",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 5. Logistics, Fleet & Transportation
    if any(k in u for k in ["логистик", "транспорт", "груз", "перевоз", "автопарк", "флот", "каршеринг", 
                           "вагон", "железнодорож", "авиац", "самолет", "метро", "судостроен", "круиз", 
                           "air cargo", "автомобильн", "поставк", "телеметри", "складск", "automotive"]):
        bp = "маршрутизация грузопотоков, диспетчеризация автопарка и мониторинг транспорта"
        ws = f"разработка транспортно-логистических платформ и телеметрии ({d})"
        st = "TMS, GPS/ГЛОНАСС, CAN-bus протоколы"
        org = "транспортно-логистическая компания / автотранспорт"
        return {
            "industry": "логистика и транспорт",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 6. Industrial, Manufacturing, Hardware & IoT
    if any(k in u for k in ["mes", "iot", "промышленн", "производств", "завод", "машиностроен", "прибор", 
                           "металлург", "энергетик", "нефтегаз", "станок", "сварка", "печатных плат", 
                           "электротехник", "электросетев", "молочн", "электроник", "ремонт"]):
        bp = "диспетчеризация производства, сбор телеметрии и контроль технологических процессов"
        ws = f"разработка и интеграция АСУ ТП и промышленного ПО в сфере {d}"
        st = "ISA-95, OPC UA, MQTT, SCADA, Modbus"
        org = "промышленное / производственное предприятие"
        return {
            "industry": "промышленность и IoT",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 7. Telecom, Media, Communication & Streaming
    if any(k in u for k in ["телеком", "связь", "провайдер", "сим-карт", "видео", "стриминг", "кинотеатр", 
                           "мультимедиа", "хостинг", "мессенджер", "чат", "broadcast", "digital media", "контент",
                           "абонент", "интернет-услуг", "знакомств", "dating"]):
        bp = "доставка медиаконтента, абонентское обслуживание и биллинг телекома"
        ws = f"разработка медиасервисов, видеостриминга и телеком-платформ ({d})"
        st = "HLS, WebRTC, RTSP, BSS/OSS"
        org = "телеком-оператор / медиахолдинг"
        return {
            "industry": "телеком и медиа",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 8. Tourism, HoReCa & Hospitality
    if any(k in u for k in ["туризм", "travel", "ресторан", "гостиниц", "отел", "бронирован"]):
        bp = "онлайн-бронирование услуг, автоматизация ресторанов и туристических сервисов"
        ws = f"разработка платформ бронирования и ресторанного менеджмента ({d})"
        st = "GDS, OTA API, ресторанные POS (iiko, r-keeper)"
        org = "туристический оператор / ресторанная сеть"
        return {
            "industry": "туризм и HoReCa",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 9. HealthTech & Medicine
    if any(k in u for k in ["медицин", "здравоохран", "пациент", "стоматолог", "врач", "клиник", "мис", "эмк", "life science", "санитарн"]):
        bp = "ведение электронных медицинских карт (ЭМК) и клинических записей"
        ws = f"разработка медицинских информационных систем и обработка данных пациентов ({d})"
        st = "HL7, DICOM, ЕГИСЗ"
        org = "медицинское учреждение / телемед-сервис"
        return {
            "industry": "медицина и здравоохранение",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 10. GovTech, Tax & Public Services
    if any(k in u for k in ["госуслуг", "госпортал", "госсектор", "налог", "ндс", "тендер", "декларац", "публичн", "государств", "госинформ", "b2g"]):
        bp = "оказание электронных госуслуг, фискальный учет и межведомственный обмен"
        ws = f"разработка и интеграция сервисов электронного правительства в сфере {d}"
        st = "СМЭВ 3, ЕСИА, ГОСТ КриптоПро"
        org = "государственное ведомство / орган власти"
        return {
            "industry": "госсектор и налоги",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 11. EdTech, Gaming & Sports
    if any(k in u for k in ["игр", "геймификац", "образован", "обучен", "урок", "доск", "спорт", "беттинг", "гемблинг", "igaming", "социальн", "достижен"]):
        bp = "геймификация пользовательского опыта, адаптивное обучение и интерактивные коммуникации"
        ws = f"создание игровых и образовательных интерактивных сервисов ({d})"
        st = "WebSockets, LMS, Unity/Unreal Engine, Real-time messaging"
        org = "EdTech-платформа / GameDev-студия / спорт-технологии"
        return {
            "industry": "EdTech и гейминг",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 12. Web3, Blockchain & AdTech
    if any(k in u for k in ["gamefi", "nft", "блокчейн", "крипто", "adtech", "dsp", "ssp", "rtb", "реклам", "маркетинг", "ido", "токен"]):
        bp = "проведение рекламных RTB-аукционов / эмиссия и смарт-контракты цифровых активов"
        ws = f"разработка programmatic-платформ и Web3-инфраструктуры ({d})"
        st = "OpenRTB, ERC-20, Smart Contracts, Web3.js"
        org = "AdTech-платформа / Web3-проект"
        return {
            "industry": "AdTech и Web3",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 13. Real Estate, Construction & Housing
    if any(k in u for k in ["недвижим", "строитель", "стройматериал", "жкх", "пожарн", "аренд", "риелтор"]):
        bp = "учет строительных объектов, каталогизация недвижимости и коммунальное обслуживание"
        ws = f"разработка систем управления объектами недвижимости и строительства ({d})"
        st = "BIM, GIS-картография, реестры недвижимости"
        org = "девелопер / агентство недвижимости / ЖКХ"
        return {
            "industry": "недвижимость и строительство",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # 14. Cybersecurity & Integration
    if any(k in u for k in ["безопасн", "уязвимост", "вредоносн", "документооборот", "подпис", "интеграц", "сервис", "защит"]):
        bp = "контроль защищенности, криптографическая подпись и системная интеграция сервисов"
        ws = f"обеспечение информационной безопасности и межсистемной интеграции ({d})"
        st = "PKI, КриптоПро ГОСТ, OAuth2, API Gateway"
        org = "enterprise-интегратор / компания кибербезопасности"
        return {
            "industry": "кибербезопасность и интеграция",
            "org_type": org,
            "business_process": bp,
            "work_scope": ws,
            "standards_and_tools": st
        }

    # Contextual inference if domain itself is ambiguous
    if any(k in ctx for k in ["банк", "финтех", "кредит"]):
        return {
            "industry": "финтех / банкинг",
            "org_type": "банк / финансовая организация",
            "business_process": "банковские бизнес-процессы и операции",
            "work_scope": f"сопровождение профильных систем в домене {d}",
            "standards_and_tools": "PCI DSS, банковские регламенты"
        }

    # Default fallback for genuine generic IT / tools
    return {
        "industry": "системное ПО и инструменты",
        "org_type": "IT-компания / разработка софта",
        "business_process": f"автоматизация и оптимизация {d}",
        "work_scope": f"проектирование и разработка решений в области {d}",
        "standards_and_tools": "CI/CD, DevOps, Microservices"
    }


class DomainEnricher:
    """
    Обогащение доменов через LLM (Слайд 5 презентации).
    Превращает сырые доменные фразы в структурированные карточки контекста:
    отрасль, организация, процессы, специфика задач и профильные стандарты (Entity Linking).
    """
    def __init__(self):
        self.card_cache: Dict[str, DomainCard] = {}

    def enrich_domain(self, domain_name: str, candidate_context: List[str] = None) -> DomainCard:
        """Обогащает бизнес-домен кандидата через LLM или онтологический движок."""
        norm_name = domain_name.strip().lower()
        if norm_name in self.card_cache:
            return self.card_cache[norm_name]

        context_str = ", ".join(candidate_context) if candidate_context else "контекст не указан"
        user_prompt = (
            f"Бизнес-домен для анализа: '{domain_name}'\n"
            f"Сопутствующие домены и опыт кандидата: {context_str}"
        )

        card: Optional[DomainCard] = None

        # 1. Attempt LLM generation if available
        if getattr(llm_client, "provider", None) in ("groq", "mistral"):
            try:
                data = llm_client.generate_json(DOMAIN_CARD_SYSTEM_PROMPT, user_prompt)
                ind = data.get("industry")
                if ind and ind not in ("IT / Бизнес", "не указана"):
                    card = DomainCard(
                        domain=data.get("domain") or domain_name,
                        industry=ind,
                        org_type=data.get("org_type") or "коммерческая организация",
                        business_process=data.get("business_process") or f"автоматизация процессов {domain_name}",
                        work_scope=data.get("work_scope") or f"разработка и сопровождение в домене {domain_name}",
                        standards_and_tools=data.get("standards_and_tools") or ""
                    )
            except Exception as e:
                logger.warning(f"Ошибка LLM-обогащения для '{domain_name}': {e}. Использование таксономического движка.")

        # 2. Rich deterministic fallback if LLM gave generic or failed
        if not card:
            tax = classify_domain_full(domain_name, candidate_context)
            card = DomainCard(
                domain=domain_name,
                industry=tax["industry"],
                org_type=tax["org_type"],
                business_process=tax["business_process"],
                work_scope=tax["work_scope"],
                standards_and_tools=tax["standards_and_tools"]
            )

        card.build_enriched_text()
        self.card_cache[norm_name] = card
        return card

    def enrich_candidate(self, candidate: RawCandidate) -> EnrichedCandidate:
        """Обогащает все очищенные домены кандидата через LLM в DomainCards."""
        cleaned = clean_candidate_domains(candidate.raw_domains)
        cards: List[DomainCard] = []
        industries = set()

        for d in cleaned:
            card = self.enrich_domain(d, cleaned)
            cards.append(card)
            if card.industry:
                industries.add(card.industry)

        return EnrichedCandidate(
            id=candidate.id,
            cleaned_domains=cleaned,
            primary_industries=list(industries),
            domain_cards=cards
        )

domain_enricher = DomainEnricher()
