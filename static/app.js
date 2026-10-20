document.addEventListener("DOMContentLoaded", () => {
  const queryInput = document.getElementById("queryInput");
  if (queryInput) {
    queryInput.value = "";
  }
  const submitBtn = document.getElementById("submitBtn");
  const presetChips = document.getElementById("presetChips");
  const topKSelect = document.getElementById("topKSelect");
  if (topKSelect) {
    topKSelect.value = "3";
  }
  const candidatesContainer = document.getElementById("candidatesContainer");
  const resultsCount = document.getElementById("resultsCount");

  // Parsed query fields
  const parsedBox = document.getElementById("parsedBox");
  const parsedRole = document.getElementById("parsedRole");
  const parsedSeniority = document.getElementById("parsedSeniority");
  const parsedStack = document.getElementById("parsedStack");
  const parsedIndustry = document.getElementById("parsedIndustry");
  const parsedDomains = document.getElementById("parsedDomains");

  // Clean Light Theme (Dark theme completely removed)
  document.documentElement.removeAttribute("data-theme");
  try {
    localStorage.removeItem("theme");
  } catch (e) {}

  // Load Presets
  async function loadPresets() {
    try {
      const res = await fetch("/api/presets");
      if (!res.ok) return;
      const presets = await res.json();
      presetChips.innerHTML = "";

      presets.forEach((p, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "preset-btn";
        btn.innerHTML = p.title;
        btn.addEventListener("click", () => {
          queryInput.value = p.query;
          executeMatch();
        });
        presetChips.appendChild(btn);
      });
    } catch (e) {
      console.warn("Could not load presets:", e);
    }
  }

  // Execute Match
  async function executeMatch() {
    const query = queryInput.value.trim();
    if (!query) {
      alert("Пожалуйста, введите текст запроса рекрутера.");
      return;
    }

    // Set Loading State
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>Поиск...</span>`;
    candidatesContainer.innerHTML = `
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p style="color: #cbd5e1; font-weight: 600; font-size: 1.05rem;">
          Анализ запроса через LLM &bull; Поиск в графе связей &bull; Cross-Encoder Реранкинг...
        </p>
      </div>
    `;

    try {
      const response = await fetch("/api/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          top_k: parseInt(topKSelect.value, 10),
          industry_filter: false
        })
      });

      if (!response.ok) {
        let errorMsg = "Ошибка при выполнении запроса";
        try {
          const errData = await response.json();
          if (typeof errData.detail === "string") {
            errorMsg = errData.detail;
          } else if (Array.isArray(errData.detail)) {
            errorMsg = errData.detail.map(d => d.msg || JSON.stringify(d)).join(", ");
          } else if (errData.detail) {
            errorMsg = JSON.stringify(errData.detail);
          }
        } catch (e) {}
        throw new Error(errorMsg);
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      candidatesContainer.innerHTML = `
        <div class="empty-state" style="border-color: rgba(239, 68, 68, 0.4);">
          <p style="color: #f87171; font-weight: 600;">Ошибка: ${escapeHtml(err.message)}</p>
        </div>
      `;
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span>Рассчитать соответствие</span>`;
    }
  }

  // Render Candidates & Query Breakdown
  function renderResults(data) {
    const { parsed_query, count, results } = data;

    // 1. Update Parsed Query Breakdown
    parsedBox.classList.add("active");
    if (parsedRole) parsedRole.textContent = parsed_query.role || "не указана";
    if (parsedSeniority) parsedSeniority.textContent = parsed_query.seniority || "любой";
    parsedStack.textContent = (parsed_query.stack && parsed_query.stack.length > 0)
      ? parsed_query.stack.join(", ") 
      : "не указан";
    parsedIndustry.textContent = parsed_query.industry || "все отрасли";
    parsedDomains.textContent = (parsed_query.domains && parsed_query.domains.length > 0)
      ? parsed_query.domains.join(", ") 
      : "не выделены";

    resultsCount.textContent = count;

    if (!results || results.length === 0) {
      candidatesContainer.innerHTML = `
        <div class="empty-state">
          <p style="color: var(--text-muted);">Кандидаты не найдены по заданным критериям фильтрации.</p>
        </div>
      `;
      return;
    }

    // 2. Render Candidate Cards
    candidatesContainer.innerHTML = "";

    results.forEach((c) => {
      const isTop1 = c.rank === 1;
      const cardEl = document.createElement("article");
      cardEl.className = `candidate-card ${isTop1 ? "top-1" : ""}`;

      let verdictClass = "verdict-none";
      if (c.verdict.includes("ТОЧНОЕ")) verdictClass = "verdict-exact";
      else if (c.verdict.includes("БЛИЗКИЙ")) verdictClass = "verdict-close";
      else if (c.verdict.includes("СМЕЖНЫЙ")) verdictClass = "verdict-related";

      // Render domains pills
      const domainsHtml = (c.matched_domains || [])
        .map(d => `<span class="domain-pill">${escapeHtml(d)}</span>`)
        .join("");

      // Render best card expandable block if present
      let cardDetailsHtml = "";
      if (c.best_card) {
        cardDetailsHtml = `
          <button type="button" class="domain-card-toggle" data-target="details-${c.candidate_id}">
            <span>Карточка домена кандидата № ${c.candidate_id}</span>
            <span class="toggle-arrow">▼</span>
          </button>
          <div id="details-${c.candidate_id}" class="domain-card-details">
            <div class="details-grid">
              <div>
                <div class="detail-prop-label">Кандидат в базе:</div>
                <div class="detail-prop-val">№ ${c.candidate_id}</div>
              </div>
              <div>
                <div class="detail-prop-label">Домен:</div>
                <div class="detail-prop-val">${escapeHtml(c.best_card.domain)}</div>
              </div>
              <div>
                <div class="detail-prop-label">Отрасль (достроено через граф/LLM):</div>
                <div class="detail-prop-val">${escapeHtml(c.best_card.industry)}</div>
              </div>
              <div>
                <div class="detail-prop-label">Тип организации:</div>
                <div class="detail-prop-val">${escapeHtml(c.best_card.org_type)}</div>
              </div>
              <div>
                <div class="detail-prop-label">Бизнес-процессы:</div>
                <div class="detail-prop-val">${escapeHtml(c.best_card.business_process)}</div>
              </div>
            </div>
            <div style="margin-top: 0.6rem;">
              <div class="detail-prop-label">В чём заключалась работа (Work Scope):</div>
              <div class="detail-prop-val">${escapeHtml(c.best_card.work_scope)}</div>
            </div>
            ${c.best_card.standards_and_tools ? `
            <div style="margin-top: 0.6rem; padding: 0.45rem 0.65rem; background: #eff6ff; border-radius: 6px; border-left: 3px solid #3b82f6;">
              <div class="detail-prop-label" style="color: #1e40af; font-weight: 600;">Промышленные стандарты и профильные системы (Entity Linking):</div>
              <div class="detail-prop-val" style="color: #1e3a8a; font-weight: 500;">${escapeHtml(c.best_card.standards_and_tools)}</div>
            </div>` : ''}
            <div class="enriched-card-summary">
              <strong>Текст для семантического поиска:</strong> "${escapeHtml(c.best_card.enriched_text)}"
            </div>
          </div>
        `;
      }

      cardEl.innerHTML = `
        <div class="card-top-row">
          <div class="candidate-id-wrap">
            <span class="candidate-db-badge">Кандидат № ${c.candidate_id}</span>
            <span class="verdict-badge ${verdictClass}">${escapeHtml(c.verdict)}</span>
          </div>

          <div class="scores-container">
            <div class="score-metric">
              <div class="score-metric-label">Балл домена</div>
              <div class="score-metric-value ${isTop1 ? 'highlight' : ''}">${c.domain_score}%</div>
            </div>
            <div class="score-metric">
              <div class="score-metric-label">Общий скор</div>
              <div class="score-metric-value">${c.composite_score}%</div>
            </div>
          </div>
        </div>

        <div class="reasoning-box">
          <strong>Обоснование реранкера:</strong> ${escapeHtml(c.reasoning)}
        </div>

        <div class="domain-chips-row">
          ${domainsHtml}
        </div>

        ${cardDetailsHtml}
      `;

      candidatesContainer.appendChild(cardEl);
    });

    // Setup toggle listeners
    document.querySelectorAll(".domain-card-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-target");
        const detailsEl = document.getElementById(targetId);
        const arrow = btn.querySelector(".toggle-arrow");
        if (detailsEl) {
          detailsEl.classList.toggle("open");
          if (arrow) {
            arrow.textContent = detailsEl.classList.contains("open") ? "▲" : "▼";
          }
        }
      });
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Event Listeners
  submitBtn.addEventListener("click", executeMatch);
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      executeMatch();
    }
  });

  // Init
  loadPresets();
});
