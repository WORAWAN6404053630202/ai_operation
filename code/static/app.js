// ─── State ────────────────────────────────────────────────────────────────────
let sessionId = "";
let isSending = false;

const CLIENT_ID_KEY = "restbiz_client_id";
const BOT_AVATAR = "https://supercoconut.co/wp-content/uploads/2025/04/cropped-Untitled-design-18.png";

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const chatMessages     = document.getElementById("chatMessages");
const messageInput     = document.getElementById("messageInput");
const sendBtn          = document.getElementById("sendBtn");
const newChatBtn       = document.getElementById("newChatBtn");
const deleteSessionBtn = document.getElementById("deleteSessionBtn");
const sessionList      = document.getElementById("sessionList");
const topicCards       = document.getElementById("topicCards");
const welcomePanel     = document.getElementById("welcomePanel");
const welcomeTitle     = document.getElementById("welcomeTitle");
const welcomeSubtitle  = document.getElementById("welcomeSubtitle");

// ─── Helpers ──────────────────────────────────────────────────────────────────
function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function textToHtml(text) {
  return escapeHtml(text).replace(/\n/g, "<br>");
}

function formatTime(unixSeconds) {
  if (!unixSeconds) return "";
  return new Date(unixSeconds * 1000).toLocaleString("th-TH", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

function getClientId() {
  let id = localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id = (window.crypto?.randomUUID?.()) ||
         `client_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

function getHeaders() {
  return {
    "Content-Type": "application/json",
    "X-Client-Id": getClientId(),
  };
}

function setInputLocked(locked) {
  isSending = locked;
  messageInput.disabled = locked;
  sendBtn.disabled = locked;
}

function autoResize() {
  messageInput.style.height = "24px";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + "px";
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ─── Welcome panel ────────────────────────────────────────────────────────────
function showWelcome(title = "สวัสดีครับ 👋", subtitle = "") {
  welcomeTitle.textContent = title;
  welcomeSubtitle.textContent = subtitle;
  welcomePanel.style.display = "";
  chatMessages.style.display = "none";
}

function hideWelcome() {
  welcomePanel.style.display = "none";
  chatMessages.style.display = "";
}

// ─── Topic cards ──────────────────────────────────────────────────────────────
function renderTopicCards(topics = []) {
  topicCards.innerHTML = "";
  topicCards.style.display = topics.length > 0 ? "grid" : "none";

  topics.forEach((topic) => {
    const btn = document.createElement("button");
    btn.className = "topic-card";
    btn.type = "button";
    btn.innerHTML = `
      <div class="topic-title">${escapeHtml(topic.title || "")}</div>
      <div class="topic-desc">${escapeHtml(topic.description || "")}</div>
    `;
    btn.addEventListener("click", () => {
      if (isSending) return;
      messageInput.value = topic.title || "";
      autoResize();
      sendMessage();
    });
    topicCards.appendChild(btn);
  });
}

// ─── Session list ─────────────────────────────────────────────────────────────
function renderSessionList(sessions = []) {
  sessionList.innerHTML = "";
  sessions.forEach((item) => {
    const btn = document.createElement("button");
    btn.className = "session-item" + (item.session_id === sessionId ? " active" : "");
    btn.innerHTML = `
      <div class="session-item-title">${escapeHtml(item.preview || item.session_id)}</div>
      <div class="session-item-meta">${escapeHtml(formatTime(item.updated_at))}</div>
    `;
    btn.addEventListener("click", () => {
      if (item.session_id !== sessionId) loadSession(item.session_id);
    });
    sessionList.appendChild(btn);
  });
}

// ─── Message rendering ────────────────────────────────────────────────────────
function appendMessage(role, content) {
  hideWelcome();
  const isAssistant = role === "assistant";

  const row = document.createElement("div");
  row.className = `message-row ${isAssistant ? "assistant" : "user"}`;

  const avatarHtml = isAssistant
    ? `<img class="message-avatar" src="${BOT_AVATAR}" alt="bot" />`
    : `<div class="message-avatar user-avatar">U</div>`;

  const roleLabel = isAssistant ? "RESTBIZ" : "You";

  row.innerHTML = `
    <div class="message-card">
      ${avatarHtml}
      <div class="message-bubble-wrap">
        <div class="message-role">${roleLabel}</div>
        <div class="message-bubble">${textToHtml(content)}</div>
      </div>
    </div>
  `;

  chatMessages.appendChild(row);
  scrollToBottom();
}

// Creates an empty streaming bubble, returns { row, bubble }
function createStreamingBubble() {
  hideWelcome();
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="message-card">
      <img class="message-avatar" src="${BOT_AVATAR}" alt="bot" />
      <div class="message-bubble-wrap">
        <div class="message-role">RESTBIZ</div>
        <div class="message-bubble"><span class="typing-cursor">▋</span></div>
      </div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();
  const bubble = row.querySelector(".message-bubble");
  return { row, bubble };
}

// ─── API helpers ──────────────────────────────────────────────────────────────
async function apiGet(path) {
  const res = await fetch(path, { headers: { "X-Client-Id": getClientId() } });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

async function apiPost(path, payload = {}) {
  const res = await fetch(path, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

// ─── Session management ───────────────────────────────────────────────────────
async function refreshSessions() {
  try {
    const data = await apiGet("/api/v1/sessions");
    renderSessionList(data.sessions || []);
    return data.sessions || [];
  } catch (err) {
    console.error("refreshSessions error:", err);
    return [];
  }
}

async function createNewSession() {
  chatMessages.innerHTML = "";
  showWelcome("กรุณารอสักครู่.....", "ระบบกำลังเตรียมพร้อม");
  renderTopicCards([]);

  try {
    const data = await apiPost("/api/v1/greeting", { persona_id: "practical" });

    sessionId = data.session_id || "";

    // Update welcome title with greeting text, then switch to chat view
    const greeting = data.response || "สวัสดีครับ";
    const topics = data.topics || [];

    if (topics.length > 0) {
      // Show welcome panel with topic cards
      welcomeTitle.textContent = "ยินดีให้บริการครับ 😊";
      welcomeSubtitle.textContent = "เลือกหัวข้อที่ต้องการ หรือพิมพ์คำถามได้เลยครับ";
      renderTopicCards(topics);
      welcomePanel.style.display = "";
      chatMessages.style.display = "none";
    } else {
      appendMessage("assistant", greeting);
    }

    await refreshSessions();
  } catch (err) {
    console.error("createNewSession error:", err);
    showWelcome("เกิดข้อผิดพลาด", err.message);
  }
}

async function loadSession(targetId) {
  try {
    const data = await apiPost("/api/v1/session/load", { session_id: targetId });

    sessionId = data.session_id || "";
    chatMessages.innerHTML = "";

    const messages = data.messages || [];
    if (messages.length === 0) {
      showWelcome("ยินดีให้บริการครับ 😊", "พิมพ์คำถามได้เลยครับ");
    } else {
      hideWelcome();
      messages.forEach((msg) => {
        appendMessage(msg.role, msg.content || "");
      });
    }

    await refreshSessions();
  } catch (err) {
    console.error("loadSession error:", err);
    appendMessage("assistant", `โหลด session ไม่สำเร็จ: ${err.message}`);
  }
}

async function deleteCurrentSession() {
  if (!sessionId) return;

  try {
    await apiPost("/api/v1/session/delete", { session_id: sessionId });
    sessionId = "";

    const sessions = await refreshSessions();
    if (sessions.length > 0) {
      await loadSession(sessions[0].session_id);
    } else {
      await createNewSession();
    }
  } catch (err) {
    console.error("deleteCurrentSession error:", err);
    appendMessage("assistant", `ลบ session ไม่สำเร็จ: ${err.message}`);
  }
}

// ─── Send message (SSE streaming) ─────────────────────────────────────────────
async function sendMessage() {
  const text = (messageInput.value || "").trim();
  if (!text || isSending) return;

  // If no session yet, create one first
  if (!sessionId) {
    await createNewSession();
    if (!sessionId) return; // still failed
  }

  // Hide welcome, show chat
  hideWelcome();

  appendMessage("user", text);
  messageInput.value = "";
  autoResize();
  setInputLocked(true);

  const { bubble } = createStreamingBubble();
  let fullText = "";

  try {
    const res = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by \n\n
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete last part

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;

        let payload;
        try {
          payload = JSON.parse(line.slice(6));
        } catch {
          continue;
        }

        if (payload.type === "chunk") {
          fullText += payload.text || "";
          bubble.innerHTML = textToHtml(fullText) + '<span class="typing-cursor">▋</span>';
          scrollToBottom();

        } else if (payload.type === "done") {
          bubble.innerHTML = textToHtml(fullText);
          sessionId = payload.session_id || sessionId;
          scrollToBottom();
          await refreshSessions();

        } else if (payload.type === "error") {
          bubble.innerHTML = `<span style="color:#e53e3e">เกิดข้อผิดพลาด: ${escapeHtml(payload.message)}</span>`;
        }
      }
    }

    // If stream ended without a "done" event, remove cursor
    if (bubble.innerHTML.includes("typing-cursor")) {
      bubble.innerHTML = textToHtml(fullText);
    }

  } catch (err) {
    console.error("sendMessage error:", err);
    bubble.innerHTML = `<span style="color:#e53e3e">เกิดข้อผิดพลาด: ${escapeHtml(err.message)}</span>`;
  } finally {
    setInputLocked(false);
    messageInput.focus();
  }
}

// ─── Event listeners ──────────────────────────────────────────────────────────
newChatBtn.addEventListener("click", () => {
  if (!isSending) createNewSession();
});

deleteSessionBtn.addEventListener("click", () => {
  if (!isSending) deleteCurrentSession();
});

sendBtn.addEventListener("click", sendMessage);

messageInput.addEventListener("input", autoResize);

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  autoResize();

  // Hide chat, show loading welcome
  showWelcome("กรุณารอสักครู่.....", "ระบบกำลังเตรียมพร้อม");
  chatMessages.style.display = "none";

  const sessions = await refreshSessions();

  if (sessions.length > 0) {
    await loadSession(sessions[0].session_id);
  } else {
    await createNewSession();
  }
});
