from __future__ import annotations

MEMORY_INTROSPECTION_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Memory Vault AI - Introspection</title>
  <style>
    :root {
      --bg-start: #f4f7ef;
      --bg-end: #e9f5f6;
      --card: #ffffff;
      --ink: #132630;
      --muted: #4d6572;
      --line: #d6e2e6;
      --brand: #0b7f8a;
      --brand-2: #f27a54;
      --good: #0a9b62;
      --bad: #c0392b;
      --shadow: 0 16px 36px rgba(17, 39, 48, 0.12);
      --radius: 16px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Century Gothic", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 18%, rgba(242, 122, 84, 0.2), transparent 36%),
        radial-gradient(circle at 88% 8%, rgba(11, 127, 138, 0.22), transparent 42%),
        linear-gradient(145deg, var(--bg-start), var(--bg-end));
      padding: 24px;
    }

    .shell {
      max-width: 1160px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
      animation: page-rise 620ms cubic-bezier(.2,.8,.2,1);
    }

    .hero {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: calc(var(--radius) + 6px);
      box-shadow: var(--shadow);
      padding: 22px 24px;
      position: relative;
      overflow: hidden;
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: auto -30px -40px auto;
      width: 220px;
      height: 220px;
      background: radial-gradient(circle, rgba(11, 127, 138, 0.16), transparent 70%);
      pointer-events: none;
    }

    .title {
      margin: 0;
      letter-spacing: 0.02em;
      font-size: clamp(1.45rem, 2.3vw, 2rem);
    }

    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 72ch;
      line-height: 1.45;
    }

    .grid {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 16px;
    }

    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
      animation: card-slide 520ms ease both;
    }

    .panel:nth-child(2) { animation-delay: 60ms; }
    .panel:nth-child(3) { animation-delay: 120ms; }

    h2 {
      margin: 4px 0 12px;
      font-size: 1.04rem;
      letter-spacing: 0.01em;
    }

    .stack {
      display: grid;
      gap: 10px;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    input,
    select,
    textarea,
    button {
      font: inherit;
    }

    input,
    select,
    textarea {
      width: 100%;
      border: 1px solid #b7c8cf;
      border-radius: 10px;
      padding: 10px 11px;
      color: var(--ink);
      background: #fcfefe;
      transition: border-color 140ms ease, box-shadow 140ms ease;
    }

    input:focus,
    select:focus,
    textarea:focus {
      outline: none;
      border-color: var(--brand);
      box-shadow: 0 0 0 4px rgba(11, 127, 138, 0.12);
    }

    textarea {
      min-height: 92px;
      resize: vertical;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }

    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      cursor: pointer;
      transition: transform 90ms ease, filter 140ms ease;
    }

    button:active {
      transform: translateY(1px);
    }

    .primary {
      color: #fff;
      background: linear-gradient(120deg, var(--brand), #05939f);
    }

    .secondary {
      color: #fff;
      background: linear-gradient(120deg, var(--brand-2), #ea5b52);
    }

    .ghost {
      background: #edf4f6;
      color: #214451;
    }

    .status {
      min-height: 22px;
      margin: 8px 0 0;
      font-size: 0.9rem;
      color: var(--muted);
    }

    .status.good { color: var(--good); }
    .status.bad { color: var(--bad); }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 12px;
      padding: 0;
      list-style: none;
    }

    .chip {
      border: 1px solid var(--line);
      background: #f5fbfc;
      color: #265663;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.82rem;
    }

    .list {
      display: grid;
      gap: 10px;
    }

    .memory-card {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fff;
      display: grid;
      gap: 7px;
    }

    .memory-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }

    .mono {
      font-family: "Courier Prime", "Consolas", monospace;
      font-size: 0.82rem;
      color: #2d4f5c;
    }

    .meta {
      color: var(--muted);
      font-size: 0.84rem;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .content {
      margin: 0;
      line-height: 1.45;
      color: #213843;
      white-space: pre-wrap;
    }

    .empty {
      padding: 16px;
      border: 1px dashed #b8ccd4;
      border-radius: 10px;
      color: var(--muted);
      background: #f8fcfd;
    }

    .recall-box {
      margin-top: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fcfefe;
    }

    .recall-box pre {
      margin: 0;
      white-space: pre-wrap;
      font-family: "Courier Prime", "Consolas", monospace;
      font-size: 0.83rem;
    }

    @keyframes page-rise {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes card-slide {
      from { opacity: 0; transform: translateY(14px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 980px) {
      body { padding: 14px; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1 class="title">Memory Introspection Console</h1>
      <p class="subtitle">
        Inspect, search, and curate a user memory timeline in one place. This page calls the live
        API endpoints, so it reflects exactly what the assistant can retrieve.
      </p>
    </section>

    <section class="grid">
      <aside class="panel">
        <h2>Connection</h2>
        <div class="stack">
          <label>
            User ID
            <input id="userId" type="text" value="default" placeholder="user_123" />
          </label>
          <label>
            API key (optional)
            <input id="apiKey" type="password" placeholder="Paste ML_API_KEY value" />
          </label>
          <div class="row">
            <label>
              Page
              <input id="page" type="number" min="1" value="1" />
            </label>
            <label>
              Page size
              <input id="pageSize" type="number" min="1" value="20" />
            </label>
          </div>
          <div class="row">
            <label>
              Memory type
              <select id="memoryTypeFilter">
                <option value="">All</option>
                <option value="episodic">Episodic</option>
                <option value="semantic">Semantic</option>
                <option value="procedural">Procedural</option>
                <option value="working">Working</option>
              </select>
            </label>
            <label>
              Include compressed
              <select id="includeCompressed">
                <option value="false">No</option>
                <option value="true">Yes</option>
              </select>
            </label>
          </div>
        </div>

        <h2 style="margin-top:16px;">Save Memory</h2>
        <div class="stack">
          <label>
            Session ID
            <input id="sessionId" type="text" value="sess_ui" />
          </label>
          <label>
            Type hint
            <select id="saveType">
              <option value="">Auto</option>
              <option value="episodic">Episodic</option>
              <option value="semantic">Semantic</option>
              <option value="procedural">Procedural</option>
              <option value="working">Working</option>
            </select>
          </label>
          <label>
            Content
            <textarea id="saveText" placeholder="Add memory text..."></textarea>
          </label>
          <div class="actions">
            <button id="saveBtn" class="primary">Save</button>
            <button id="refreshBtn" class="ghost">Refresh list</button>
          </div>
        </div>

        <h2 style="margin-top:16px;">Recall Probe</h2>
        <div class="stack">
          <label>
            Query
            <input id="recallQuery" type="text" placeholder="What does this user prefer?" />
          </label>
          <div class="row">
            <label>
              Top K
              <input id="recallTopK" type="number" min="1" value="5" />
            </label>
            <label>
              Token budget
              <input id="recallBudget" type="number" min="1" value="2000" />
            </label>
          </div>
          <button id="recallBtn" class="secondary">Run recall</button>
        </div>

        <div id="status" class="status"></div>
      </aside>

      <section class="panel">
        <div class="toolbar">
          <button id="prevPageBtn" class="ghost">Previous</button>
          <button id="nextPageBtn" class="ghost">Next</button>
        </div>

        <ul id="chips" class="chips"></ul>

        <div id="list" class="list"></div>

        <section id="recallPanel" class="recall-box" style="display:none;">
          <strong>Recall prompt block</strong>
          <pre id="promptBlock"></pre>
        </section>
      </section>
    </section>
  </main>

  <script>
    const userIdInput = document.getElementById("userId");
    const apiKeyInput = document.getElementById("apiKey");
    const pageInput = document.getElementById("page");
    const pageSizeInput = document.getElementById("pageSize");
    const memoryTypeFilter = document.getElementById("memoryTypeFilter");
    const includeCompressed = document.getElementById("includeCompressed");
    const sessionIdInput = document.getElementById("sessionId");
    const saveTypeInput = document.getElementById("saveType");
    const saveTextInput = document.getElementById("saveText");
    const recallQueryInput = document.getElementById("recallQuery");
    const recallTopKInput = document.getElementById("recallTopK");
    const recallBudgetInput = document.getElementById("recallBudget");
    const statusNode = document.getElementById("status");
    const listNode = document.getElementById("list");
    const chipsNode = document.getElementById("chips");
    const recallPanel = document.getElementById("recallPanel");
    const promptBlockNode = document.getElementById("promptBlock");

    function setStatus(message, level) {
      statusNode.textContent = message;
      statusNode.className = "status" + (level ? " " + level : "");
    }

    function headers(base) {
      const merged = Object.assign({}, base || {});
      const token = apiKeyInput.value.trim();
      if (token) {
        merged["Authorization"] = "Bearer " + token;
      }
      return merged;
    }

    async function requestJson(url, options) {
      const response = await fetch(url, options || {});
      const text = await response.text();
      let data = {};
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = { detail: text };
        }
      }
      if (!response.ok) {
        const detail = data.detail || ("HTTP " + response.status);
        throw new Error(detail);
      }
      return data;
    }

    function escaped(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function memoryCard(chunk) {
      const relevance = chunk.relevance_score == null
        ? "-"
        : Number(chunk.relevance_score).toFixed(2);
      const compressed = chunk.compressed ? "yes" : "no";
      return ""
        + "<article class=\"memory-card\">"
        +   "<div class=\"memory-head\">"
        +     "<span class=\"mono\">" + escaped(chunk.id) + "</span>"
        +     "<button class=\"ghost\" data-delete=\"" + escaped(chunk.id) + "\">Delete</button>"
        +   "</div>"
        +   "<div class=\"meta\">"
        +     "<span>type: " + escaped(chunk.memory_type) + "</span>"
        +     "<span>importance: " + Number(chunk.importance).toFixed(2) + "</span>"
        +     "<span>tokens: " + escaped(chunk.token_count) + "</span>"
        +     "<span>relevance: " + escaped(relevance) + "</span>"
        +     "<span>compressed: " + compressed + "</span>"
        +   "</div>"
        +   "<p class=\"content\">" + escaped(chunk.content) + "</p>"
        + "</article>";
    }

    function renderStats(result) {
      const counts = { episodic: 0, semantic: 0, procedural: 0, working: 0 };
      result.items.forEach((item) => {
        if (Object.prototype.hasOwnProperty.call(counts, item.memory_type)) {
          counts[item.memory_type] += 1;
        }
      });

      chipsNode.innerHTML = [
        "<li class=\"chip\">total: " + result.total + "</li>",
        "<li class=\"chip\">page: " + result.page + "</li>",
        "<li class=\"chip\">page size: " + result.page_size + "</li>",
        "<li class=\"chip\">episodic: " + counts.episodic + "</li>",
        "<li class=\"chip\">semantic: " + counts.semantic + "</li>",
        "<li class=\"chip\">procedural: " + counts.procedural + "</li>",
        "<li class=\"chip\">working: " + counts.working + "</li>",
      ].join("");
    }

    async function refreshList() {
      const userId = userIdInput.value.trim();
      if (!userId) {
        setStatus("User ID is required.", "bad");
        return;
      }

      const params = new URLSearchParams({
        user_id: userId,
        page: String(pageInput.value || 1),
        page_size: String(pageSizeInput.value || 20),
        include_compressed: includeCompressed.value,
      });
      if (memoryTypeFilter.value) {
        params.set("memory_type", memoryTypeFilter.value);
      }

      try {
        setStatus("Loading memories...");
        const result = await requestJson("/v1/memory?" + params.toString(), {
          headers: headers(),
        });

        renderStats(result);

        if (!result.items || result.items.length === 0) {
          listNode.innerHTML = "<div class=\"empty\">No memories matched this filter.</div>";
        } else {
          listNode.innerHTML = result.items.map(memoryCard).join("");
        }

        setStatus("Memory list updated.", "good");
      } catch (error) {
        listNode.innerHTML = "";
        chipsNode.innerHTML = "";
        setStatus(error.message || "Could not load memories.", "bad");
      }
    }

    async function saveMemory() {
      const userId = userIdInput.value.trim();
      const text = saveTextInput.value.trim();
      const sessionId = sessionIdInput.value.trim();
      if (!userId || !text || !sessionId) {
        setStatus("User ID, Session ID, and Content are required.", "bad");
        return;
      }

      const payload = {
        user_id: userId,
        session_id: sessionId,
        text: text,
      };
      if (saveTypeInput.value) {
        payload.memory_type_hint = saveTypeInput.value;
      }

      try {
        setStatus("Saving memory...");
        await requestJson("/v1/memory", {
          method: "POST",
          headers: headers({ "Content-Type": "application/json" }),
          body: JSON.stringify(payload),
        });
        saveTextInput.value = "";
        setStatus("Memory saved.", "good");
        await refreshList();
      } catch (error) {
        setStatus(error.message || "Memory save failed.", "bad");
      }
    }

    async function runRecall() {
      const userId = userIdInput.value.trim();
      const query = recallQueryInput.value.trim();
      if (!userId || !query) {
        setStatus("User ID and recall query are required.", "bad");
        return;
      }

      const params = new URLSearchParams({
        user_id: userId,
        query: query,
        top_k: String(recallTopKInput.value || 5),
        token_budget: String(recallBudgetInput.value || 2000),
      });

      try {
        setStatus("Running recall...");
        const result = await requestJson("/v1/memory/recall?" + params.toString(), {
          headers: headers(),
        });
        recallPanel.style.display = "block";
        promptBlockNode.textContent = result.prompt_block || "<memory>\n</memory>";
        setStatus("Recall completed.", "good");
      } catch (error) {
        setStatus(error.message || "Recall failed.", "bad");
      }
    }

    async function deleteMemory(memoryId) {
      const userId = userIdInput.value.trim();
      if (!userId) {
        setStatus("User ID is required for delete.", "bad");
        return;
      }

      try {
        setStatus("Deleting memory " + memoryId + "...");
        const params = new URLSearchParams({ user_id: userId });
        await requestJson("/v1/memory/" + encodeURIComponent(memoryId) + "?" + params.toString(), {
          method: "DELETE",
          headers: headers(),
        });
        setStatus("Memory deleted.", "good");
        await refreshList();
      } catch (error) {
        setStatus(error.message || "Delete failed.", "bad");
      }
    }

    document.getElementById("refreshBtn").addEventListener("click", refreshList);
    document.getElementById("saveBtn").addEventListener("click", saveMemory);
    document.getElementById("recallBtn").addEventListener("click", runRecall);
    document.getElementById("prevPageBtn").addEventListener("click", () => {
      const current = Number(pageInput.value || 1);
      pageInput.value = String(Math.max(1, current - 1));
      refreshList();
    });
    document.getElementById("nextPageBtn").addEventListener("click", () => {
      const current = Number(pageInput.value || 1);
      pageInput.value = String(current + 1);
      refreshList();
    });

    listNode.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      const memoryId = target.getAttribute("data-delete");
      if (!memoryId) {
        return;
      }
      deleteMemory(memoryId);
    });

    refreshList();
  </script>
</body>
</html>
"""


__all__ = ["MEMORY_INTROSPECTION_HTML"]
