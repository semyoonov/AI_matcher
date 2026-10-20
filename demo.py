import sys
import logging
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from pipeline import pipeline
from models import RecruiterQuery, CandidateScoreResult
from config import settings

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING)

console = Console()

SAMPLE_QUERIES = [
    {
        "title": "Кейс 1 (из ТЗ): Senior QA + Kafka + финтех + интеграция банковского ядра",
        "query": "Нужен автотестировщик senior-уровня с обязательным отличным знанием Kafka и обязательным опытом работы в финтехе. У кандидата желателен опыт с интеграцией банковского ядра"
    },
    {
        "title": "Кейс 2: Разработчик кредитных конвейеров и кредитного скоринга",
        "query": "Разработчик с глубоким опытом в анализе кредитных заявок, кредитном скоринге и кредитных конвейерах"
    },
    {
        "title": "Кейс 3: Автоматизация производства, MES-системы и промышленный IoT",
        "query": "Инженер автоматизации с практическим опытом работы в MES-системах и промышленном интернете вещей (IoT)"
    },
    {
        "title": "Кейс 4: E-commerce разработчик корзины и каталога товаров",
        "query": "Middle backend разработчик для интернет-магазина с опытом в e-commerce, корзине покупателя и динамическом ценообразовании"
    }
]

def format_verdict(verdict: str) -> Text:
    if "ТОЧНОЕ" in verdict:
        return Text(f"✔ {verdict}", style="bold green")
    elif "БЛИЗКИЙ" in verdict:
        return Text(f"✦ {verdict}", style="bold cyan")
    elif "СМЕЖНЫЙ" in verdict:
        return Text(f"~ {verdict}", style="yellow")
    else:
        return Text(f"✖ {verdict}", style="dim red")

def run_query_demo(query_text: str, top_k: int = 5):
    console.print(Panel(f"[bold cyan]Запрос рекрутера:[/bold cyan]\n[white]{query_text}[/white]", border_style="cyan"))

    with console.status("[bold green]Анализ запроса и скоринг кандидатов...[/bold green]"):
        query, results = pipeline.search_and_score(query_text, top_k=top_k)

    # 1. Parsed Query Summary
    info_table = Table(title="[bold yellow]Извлеченные требования из запроса (Слайд 4)[/bold yellow]", show_header=True, header_style="bold magenta")
    info_table.add_column("Параметр", style="cyan", width=25)
    info_table.add_column("Значение", style="white")

    info_table.add_row("Роль / Позиция", query.extracted_role or "не указана")
    info_table.add_row("Грейд / Seniority", query.extracted_seniority or "любой")
    info_table.add_row("Технический стек", ", ".join(query.extracted_stack) if query.extracted_stack else "не указан")
    info_table.add_row("Целевая макро-отрасль", query.target_industry or "все")
    info_table.add_row("Целевые бизнес-домены", ", ".join(query.target_business_domains) or "не выделены")
    if query.target_nuances:
        info_table.add_row("Нюансы опыта", query.target_nuances)

    console.print(info_table)
    console.print()

    # 2. Results Table
    results_table = Table(
        title=f"[bold green]Топ-{len(results)} кандидатов по скорингу бизнес-доменов (Слайды 7-8)[/bold green]",
        show_header=True,
        header_style="bold blue"
    )
    results_table.add_column("Ранг", style="bold", justify="center", width=6)
    results_table.add_column("ID", justify="center", width=6)
    results_table.add_column("Балл домена", justify="center", width=12)
    results_table.add_column("Итоговый скор", justify="center", width=14)
    results_table.add_column("Вердикт реранкера", justify="center", width=20)
    results_table.add_column("Ключевые домены резюме", style="white", width=35)
    results_table.add_column("Обоснование LLM Реранкера (Slide 8)", style="dim white", width=45)

    for rank, res in enumerate(results, 1):
        domain_score_str = f"{res.domain_score:.3f}"
        composite_score_str = f"{res.composite_score:.3f}"
        domains_preview = ", ".join(res.matched_domains[:3])
        if len(res.matched_domains) > 3:
            domains_preview += f" (+{len(res.matched_domains)-3})"

        results_table.add_row(
            f"#{rank}",
            str(res.candidate_id),
            domain_score_str,
            composite_score_str,
            format_verdict(res.verdict_label),
            domains_preview,
            res.reranker_reasoning
        )

    console.print(results_table)

    # 3. Best Match Card details
    best = results[0]
    if best.best_matching_card:
        card = best.best_matching_card
        card_panel = Panel(
            f"[bold green]Домен:[/bold green] {card.domain}\n"
            f"[bold green]Отрасль (через граф/LLM):[/bold green] {card.industry}\n"
            f"[bold green]Тип организации:[/bold green] {card.org_type}\n"
            f"[bold green]Бизнес-процесс:[/bold green] {card.business_process}\n"
            f"[bold green]В чём заключалась работа:[/bold green] {card.work_scope}\n\n"
            f"[bold cyan]Обогащенный текст для векторного поиска:[/bold cyan]\n"
            f"[italic]{card.enriched_text}[/italic]",
            title=f"[bold]Обогащенная Карточка Домена Топ-1 Кандидата #{best.candidate_id} (Слайд 5)[/bold]",
            border_style="green"
        )
        console.print()
        console.print(card_panel)

def main():
    console.print(Panel.fit(
        "[bold yellow]AI Matcher: Пайплайн скоринга бизнес-доменов кандидата и вакансии[/bold yellow]\n"
        "[white]Архитектура: Граф связей доменов (NetworkX) + 4-вопросное обогащение + Реранкер (Groq/Mistral)[/white]",
        border_style="yellow"
    ))

    provider_name = settings.get_active_provider().upper()
    console.print(f"[dim]Активный LLM провайдер: [bold]{provider_name}[/bold][/dim]\n")

    pipeline.initialize()

    while True:
        console.print("[bold cyan]Выберите действие:[/bold cyan]")
        for i, sample in enumerate(SAMPLE_QUERIES, 1):
            console.print(f"  [bold]{i}.[/bold] {sample['title']}")
        console.print("  [bold]5.[/bold] Ввести собственный поисковый запрос рекрутера")
        console.print("  [bold]6.[/bold] Выход")

        choice = input("\nВаш выбор (1-6): ").strip()
        if choice in ("1", "2", "3", "4"):
            idx = int(choice) - 1
            run_query_demo(SAMPLE_QUERIES[idx]["query"])
            input("\nНажмите Enter, чтобы продолжить...")
            console.clear()
        elif choice == "5":
            user_q = input("\nВведите поисковый запрос рекрутера:\n> ").strip()
            if user_q:
                run_query_demo(user_q)
                input("\nНажмите Enter, чтобы продолжить...")
                console.clear()
        elif choice in ("6", "q", "exit"):
            console.print("[bold green]До свидания![/bold green]")
            break
        else:
            console.print("[red]Неверный ввод, повторите попытку.[/red]\n")

if __name__ == "__main__":
    main()
