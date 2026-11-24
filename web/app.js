/* DocuMind chat client */

const state = {
  documents: [],
  history: [],
  sources: [],
  busy: false,
};

const $ = (sel) => document.querySelector(sel);
const messagesEl = $("#messages");
const inputEl = $("#input");
const sendBtn = $("#sendBtn");
const fileInput = $("#fileInput");
const dropzone = $("#dropzone");
const docListEl = $("#docList");
const docCountEl = $("#docCount");
const sourcePanelEl = $("#sourcePanel");
const uploadStatusEl = $("#uploadStatus");
const scopeToggle = $("#scopeToggle");
const llmLabel = $("#llmLabel");
const llmStatus = $("#llmStatus");
const chatSubtitle = $("#chatSubtitle");

/* ------------------------------------------------------------------ utils */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(text) {
  const blocks = [];
  let html = escapeHtml(text);

  // fenced code
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
    return `@@BLOCK${blocks.length - 1}@@`;
  });

  // inline code
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");

  // bold / italic
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  // headings
  html = html.replace(/^###\s+(.+)$/gm, "<h4>$1</h4>");
  html = html.replace(/^##\s+(.+)$/gm, "<h3>$1</h3>");

  // line-oriented blocks: lists, blockquotes, paragraphs
  const lines = html.split("\n");
  const out = [];
  let listType = null;

  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      closeList();
      continue;
    }
    if (/^@@BLOCK\d+@@$/.test(trimmed)) {
      closeList();
      out.push(trimmed);
      continue;
    }
    if (/^<\/?(ul|ol|pre|h3|h4|table)/.test(trimmed)) {
      closeList();
      out.push(trimmed);
      continue;
    }

    const bullet = trimmed.match(/^[-*+]\s+(.+)$/);
    const numbered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    const quote = trimmed.match(/^&gt;\s*(.*)$/);

    if (bullet) {
      if (listType !== "ul") {
        closeList();
        out.push("<ul>");
        listType = "ul";
      }
      out.push(`<li>${bullet[1]}</li>`);
      continue;
    }
    if (numbered) {
      if (listType !== "ol") {
        closeList();
        out.push("<ol>");
        listType = "ol";
      }
      out.push(`<li>${numbered[1]}</li>`);
      continue;
    }
    if (quote) {
      closeList();
      out.push(`<blockquote>${quote[1]}</blockquote>`);
      continue;
    }

    closeList();
    out.push(`<p>${trimmed}</p>`);
  }
  closeList();

  return out.join("\n").replace(/@@BLOCK(\d+)@@/g, (_, i) => blocks[Number(i)]);
}

function normalizeAnswer(text) {
  // Models sometimes emit a whole bullet list on one line ("- a [1] - b [1]").
  // Split those into real lines so the list renders correctly. Only split at
  // separators that follow a citation or bold marker, so date ranges such as
  // "2025 - 2026" are left alone.
  return text
    .split("\n")
    .map((line) => {
      const trimmed = line.trim();
      if (!/^[-*+]\s+/.test(trimmed)) return line;
      const parts = trimmed.split(/(\[[^\]]{1,4}\]|\*\*[^*]{1,80}\*\*)\s*[-–]\s+/);
      if (parts.length < 3) return line;
      let rebuilt = "";
      let bullets = 0;
      for (const part of parts) {
        if (/^(\[[^\]]{1,4}\]|\*\*[^*]{1,80}\*\*)$/.test(part)) {
          rebuilt += part + " ";
          continue;
        }
        rebuilt += `- ${part.trim()} `;
        bullets += 1;
      }
      return bullets > 1 ? rebuilt.trim() : line;
    })
    .join("\n");
}

function addCitations(text, sources) {
  const byIndex = new Map(sources.map((s) => [s.index, s]));
  const html = renderMarkdown(normalizeAnswer(text));
  return html.replace(/\[(\d{1,2})\]/g, (match, num) => {
    const source = byIndex.get(Number(num));
    if (!source) return "";
    const location = source.citation.split(" — ")[1] || source.citation;
    const title = escapeHtml(`${source.filename} — ${location}`);
    return `<span class="cite" data-src="${source.index}" title="${title}">${num}</span>`;
  });
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* --------------------------------------------------------------- messages */
function addMessage(role, html, { streaming = false } = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "user" ? "You" : "◈";
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  const content = document.createElement("div");
  content.className = "content";
  content.innerHTML = html;
  bubble.appendChild(content);
  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return { wrapper, bubble, content };
}

function addTyping() {
  const { content, bubble } = addMessage("bot", "");
  content.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  return { content, bubble };
}

/* --------------------------------------------------------------- sources */
function renderSources(sources) {
  state.sources = sources || [];
  if (!state.sources.length) {
    sourcePanelEl.innerHTML = '<p class="muted small">Sources appear here with the exact passage used to answer.</p>';
    return;
  }
  sourcePanelEl.innerHTML = state.sources
    .map(
      (s) => `
      <div class="source-card" data-index="${s.index}">
        <div class="source-head">
          <span class="source-title">${escapeHtml(s.filename)}</span>
          <span class="source-loc">${escapeHtml(s.citation.split(" — ")[1] || "")}</span>
        </div>
        <div class="source-text">${escapeHtml(s.snippet)}</div>
      </div>`
    )
    .join("");
}

/* ------------------------------------------------------------- documents */
async function refreshDocuments() {
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    state.documents = data.documents || [];
    renderDocuments();
    updateSubtitle();
  } catch (err) {
    console.error(err);
  }
}

function renderDocuments() {
  docCountEl.textContent = state.documents.length;
  if (!state.documents.length) {
    docListEl.innerHTML = '<li class="muted small">No documents yet.</li>';
    return;
  }
  docListEl.innerHTML = state.documents
    .map(
      (doc) => `
      <li class="doc-item">
        <span class="doc-icon">📄</span>
        <div class="doc-meta">
          <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
          <div class="doc-sub">${doc.pages} section${doc.pages === 1 ? "" : "s"} · ${doc.chunks} chunks · ${(doc.size_bytes / 1024).toFixed(0)} KB</div>
        </div>
        <button class="doc-remove" data-doc="${doc.doc_id}" title="Remove">✕</button>
      </li>`
    )
    .join("");
}

function updateSubtitle() {
  if (!state.documents.length) {
    chatSubtitle.textContent = "Upload a document to start asking questions.";
    return;
  }
  const names = state.documents.map((d) => d.filename);
  const preview = names.slice(0, 2).join(", ") + (names.length > 2 ? ` +${names.length - 2} more` : "");
  chatSubtitle.textContent = `${state.documents.length} document${state.documents.length === 1 ? "" : "s"} ready — ${preview}`;
}

async function uploadFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const files = Array.from(fileList);
  uploadStatusEl.innerHTML = `<span>Indexing ${files.length} file${files.length === 1 ? "" : "s"}…</span>`;

  const form = new FormData();
  files.forEach((file) => form.append("files", file));

  try {
    const res = await fetch("/api/documents", { method: "POST", body: form });
    const data = await res.json();
    const lines = (data.results || []).map((r) =>
      r.ok
        ? `<span class="ok">✓ ${escapeHtml(r.filename)} — ${r.document.chunks} chunks indexed</span>`
        : `<span class="err">✕ ${escapeHtml(r.filename)} — ${escapeHtml(r.error)}</span>`
    );
    uploadStatusEl.innerHTML = lines.join("<br/>") || "<span class='muted'>Nothing uploaded.</span>";
    await refreshDocuments();
  } catch (err) {
    uploadStatusEl.innerHTML = `<span class="err">Upload failed: ${escapeHtml(err.message)}</span>`;
  }
}

/* ------------------------------------------------------------------- chat */
function selectedDocIds() {
  if (scopeToggle.checked) return null;
  const ids = state.documents.map((d) => d.doc_id);
  return ids.length ? ids : null;
}

async function sendMessage(text) {
  if (state.busy) return;
  state.busy = true;
  sendBtn.disabled = true;

  addMessage("user", `<p>${escapeHtml(text)}</p>`);
  state.history.push({ role: "user", content: text });

  const typing = addTyping();
  let sources = [];

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: state.history.slice(-8, -1),
        doc_ids: selectedDocIds(),
      }),
    });

    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answerText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";

      for (const frame of frames) {
        const evMatch = frame.match(/^event:\s*(.+)$/m);
        const dataMatch = frame.match(/^data:\s*(.+)$/m);
        if (!evMatch || !dataMatch) continue;
        const event = evMatch[1].trim();
        let payload;
        try {
          payload = JSON.parse(dataMatch[1]);
        } catch {
          continue;
        }

        if (event === "sources") {
          sources = payload.sources || [];
          renderSources(sources);
          typing.content.innerHTML = "";
        } else if (event === "delta") {
          answerText += payload.text || "";
          typing.content.innerHTML = addCitations(answerText, sources);
          scrollToBottom();
        } else if (event === "done") {
          const finalAnswer = payload.answer || answerText;
          typing.content.innerHTML = addCitations(finalAnswer, sources);
          const meta = [];
          if (payload.mode) meta.push(`<span class="badge mode">${escapeHtml(payload.mode)}</span>`);
          if (payload.grounded === false) meta.push('<span class="badge ungrounded">not found in documents</span>');
          else if (payload.mode) meta.push('<span class="badge grounded">grounded</span>');
          if (payload.elapsed_ms != null) meta.push(`${payload.elapsed_ms} ms`);
          if (payload.warning) meta.push(`<span class="warning">${escapeHtml(payload.warning)}</span>`);

          const metaEl = document.createElement("div");
          metaEl.className = "msg-meta";
          metaEl.innerHTML = meta.filter((m) => m).join(" ");
          typing.bubble.appendChild(metaEl);

          state.history.push({ role: "assistant", content: finalAnswer });
        }
      }
    }
  } catch (err) {
    typing.content.innerHTML = `<p class="warning">Could not get an answer: ${escapeHtml(err.message)}</p>`;
  } finally {
    state.busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
    scrollToBottom();
  }
}

/* ------------------------------------------------------------------ events */
$("#chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text || state.busy) return;
  inputEl.value = "";
  inputEl.style.height = "auto";
  await sendMessage(text);
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chatForm").requestSubmit();
  }
});

fileInput.addEventListener("change", () => {
  uploadFiles(fileInput.files);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer && e.dataTransfer.files) uploadFiles(e.dataTransfer.files);
});

docListEl.addEventListener("click", async (event) => {
  const btn = event.target.closest(".doc-remove");
  if (!btn) return;
  const docId = btn.dataset.doc;
  await fetch(`/api/documents/${docId}`, { method: "DELETE" });
  await refreshDocuments();
});

$("#clearAll").addEventListener("click", async () => {
  if (!state.documents.length) return;
  await fetch("/api/reset", { method: "POST" });
  uploadStatusEl.innerHTML = "";
  await refreshDocuments();
});

$("#newChat").addEventListener("click", () => {
  state.history = [];
  messagesEl.innerHTML = "";
  location.reload();
});

$("#suggestions").addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (!chip) return;
  inputEl.value = chip.textContent;
  $("#chatForm").requestSubmit();
});

sourcePanelEl.addEventListener("click", (event) => {
  const card = event.target.closest(".source-card");
  if (!card) return;
  card.style.borderColor = "var(--accent)";
  setTimeout(() => (card.style.borderColor = ""), 1200);
});

/* ------------------------------------------------------------------- boot */
(async function boot() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.llm && data.llm.available) {
      llmStatus.className = "status-dot on";
      llmLabel.textContent = `engine: ${data.llm.provider}`;
    } else {
      llmStatus.className = "status-dot off";
      llmLabel.textContent = "engine: extractive mode (no LLM key)";
    }
  } catch {
    llmStatus.className = "status-dot off";
    llmLabel.textContent = "engine: unreachable";
  }
  await refreshDocuments();
  inputEl.focus();
})();
