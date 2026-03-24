let sessionId = "";
let currentPersona = "practical";

const CLIENT_ID_KEY = "restbiz_client_id";

const chatMessages = document.getElementById("chatMessages");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const healthBtn = document.getElementById("healthBtn");
const deleteSessionBtn = document.getElementById("deleteSessionBtn");
const sessionList = document.getElementById("sessionList");
const topicCards = document.getElementById("topicCards");
const welcomePanel = document.getElementById("welcomePanel");
const botSelect = document.getElementById("botSelect");

function escapeHtml(text) {
  return (text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatTimeLabel(unixSeconds) {
  if (!unixSeconds) return "";
  const date = new Date(unixSeconds * 1000);
  return date.toLocaleString("th-TH", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

function autoResizeTextarea() {
  messageInput.style.height = "24px";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 220) + "px";
}

function getClientId() {
  let value = localStorage.getItem(CLIENT_ID_KEY);
  if (value && value.trim()) return value;

  if (window.crypto && crypto.randomUUID) {
    value = crypto.randomUUID();
  } else {
    value = `client_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  }

  localStorage.setItem(CLIENT_ID_KEY, value);
  return value;
}

function getDefaultHeaders() {
  return {
    "Content-Type": "application/json",
    "X-Client-Id": getClientId(),
  };
}

function clearMessages() {
  chatMessages.innerHTML = "";
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideWelcome() {
  welcomePanel.style.display = "none";
}

function showWelcome() {
  welcomePanel.style.display = "";
}

function renderMessage(role, content) {
  hideWelcome();

  const row = document.createElement("div");
  row.className = `message-row ${role === "assistant" ? "assistant" : "user"}`;

  let avatarHtml = "";
  let roleLabel = "";

  if (role === "assistant") {
    avatarHtml = `<img class="message-avatar" src="https://supercoconut.co/wp-content/uploads/2025/04/cropped-Untitled-design-18.png" alt="bot" />`;
    roleLabel = "RESTBIZ";
  } else {
    avatarHtml = `<div class="message-avatar user-avatar">Y</div>`;
    roleLabel = "You";
  }

  row.innerHTML = `
    <div class="message-card">
      ${avatarHtml}
      <div class="message-bubble-wrap">
        <div class="message-role">${roleLabel}</div>
        <div class="message-bubble">${escapeHtml(content).replace(/\n/g, "<br>")}</div>
      </div>
    </div>
  `;

  chatMessages.appendChild(row);
  scrollToBottom();
}

function renderTopicCards(topics = []) {
  topicCards.innerHTML = "";
  
  // แสดงการ์ดเมื่อมีข้อมูล
  if (topics && topics.length > 0) {
    topicCards.style.display = "grid";
  } else {
    topicCards.style.display = "none";
  }

  topics.forEach((topic) => {
    const btn = document.createElement("button");
    btn.className = "topic-card";
    btn.type = "button";
    btn.innerHTML = `
      <div class="topic-title">${escapeHtml(topic.title || "-")}</div>
      <div class="topic-desc">${escapeHtml(topic.description || "")}</div>
    `;
    btn.addEventListener("click", () => {
      messageInput.value = topic.title || "";
      autoResizeTextarea();
      sendMessage();
    });
    topicCards.appendChild(btn);
  });
}

function renderSessionList(items = []) {
  sessionList.innerHTML = "";

  items.forEach((item) => {
    const btn = document.createElement("button");
    btn.className = "session-item";
    if (item.session_id === sessionId) {
      btn.classList.add("active");
    }

    btn.innerHTML = `
      <div class="session-item-title">${escapeHtml(item.preview || item.session_id)}</div>
      <div class="session-item-meta">${escapeHtml(item.persona_id || "")} • ${escapeHtml(formatTimeLabel(item.updated_at))}</div>
    `;

    btn.addEventListener("click", () => {
      loadSession(item.session_id);
    });

    sessionList.appendChild(btn);
  });
}

async function apiGet(url) {
  const response = await fetch(url, {
    headers: {
      "X-Client-Id": getClientId(),
    },
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

async function apiPost(url, payload = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: getDefaultHeaders(),
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

async function refreshSessions() {
  try {
    const data = await apiGet("/api/v1/sessions");
    const sessions = data.sessions || [];
    renderSessionList(sessions);
    return sessions;
  } catch (error) {
    console.error(error);
    return [];
  }
}

async function createNewSession() {
  try {
    clearMessages();
    showWelcome();

    currentPersona = botSelect.value || "practical";

    const data = await apiPost("/api/v1/greeting", {
      persona_id: currentPersona,
    });

    sessionId = data.session_id || "";
    currentPersona = data.persona_id || currentPersona;
    botSelect.value = currentPersona;

    renderTopicCards(data.topics || []);
    renderMessage("assistant", data.response || "สวัสดีครับ");
    await refreshSessions();
  } catch (error) {
    console.error(error);
    renderMessage("assistant", `เกิดข้อผิดพลาดในการเริ่มแชต: ${error.message}`);
  }
}

async function loadSession(targetSessionId) {
  try {
    const data = await apiPost("/api/v1/session/load", {
      session_id: targetSessionId,
    });

    sessionId = data.session_id || "";
    currentPersona = data.persona_id || "practical";
    botSelect.value = currentPersona;

    clearMessages();

    const messages = data.messages || [];
    if (messages.length === 0) {
      showWelcome();
    } else {
      hideWelcome();
      messages.forEach((msg) => {
        renderMessage(msg.role, msg.content || "");
      });
    }

    await refreshSessions();
  } catch (error) {
    console.error(error);
    renderMessage("assistant", `โหลด session ไม่สำเร็จ: ${error.message}`);
  }
}

async function deleteCurrentSession() {
  if (!sessionId) return;

  try {
    await apiPost("/api/v1/session/delete", {
      session_id: sessionId,
    });

    sessionId = "";
    clearMessages();
    showWelcome();

    const sessions = await refreshSessions();

    if (sessions.length > 0) {
      await loadSession(sessions[0].session_id);
    } else {
      await createNewSession();
    }
  } catch (error) {
    console.error(error);
    renderMessage("assistant", `ลบ session ไม่สำเร็จ: ${error.message}`);
  }
}

async function sendMessage() {
  const text = (messageInput.value || "").trim();
  if (!text) return;

  if (!sessionId) {
    await createNewSession();
  }

  renderMessage("user", text);
  messageInput.value = "";
  autoResizeTextarea();

  // สร้าง bubble ว่างไว้ก่อน แล้วค่อย append ทีละ chunk
  hideWelcome();
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="message-card">
      <img class="message-avatar" src="https://supercoconut.co/wp-content/uploads/2025/04/cropped-Untitled-design-18.png" alt="bot" />
      <div class="message-bubble-wrap">
        <div class="message-role">RESTBIZ</div>
        <div class="message-bubble" id="streamingBubble"><span class="typing-cursor">▋</span></div>
      </div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();

  const bubble = document.getElementById("streamingBubble");
  let fullText = "";

  try {
    const response = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: getDefaultHeaders(),
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // แยก SSE events ด้วย \n\n
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // ส่วนที่ยังไม่สมบูรณ์เก็บไว้ใน buffer

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;

        try {
          const payload = JSON.parse(line.slice(6));

          if (payload.type === "chunk") {
            fullText += payload.text;
            // แสดงข้อความพร้อม cursor กระพริบ
            bubble.innerHTML = escapeHtml(fullText).replace(/\n/g, "<br>") + '<span class="typing-cursor">▋</span>';
            scrollToBottom();

          } else if (payload.type === "done") {
            // จบแล้ว ลบ cursor ออก
            bubble.innerHTML = escapeHtml(fullText).replace(/\n/g, "<br>");
            sessionId = payload.session_id || sessionId;
            currentPersona = payload.persona_id || currentPersona;
            botSelect.value = currentPersona;
            await refreshSessions();

          } else if (payload.type === "error") {
            bubble.innerHTML = `<span style="color:#e53e3e">เกิดข้อผิดพลาด: ${escapeHtml(payload.message)}</span>`;
          }
        } catch (_) {
          // JSON parse error — ข้ามไป
        }
      }
    }

    // ถ้า stream จบโดยไม่ได้รับ done event ให้ลบ cursor ออก
    if (bubble.innerHTML.includes("typing-cursor")) {
      bubble.innerHTML = escapeHtml(fullText).replace(/\n/g, "<br>");
    }

  } catch (error) {
    console.error(error);
    bubble.innerHTML = `<span style="color:#e53e3e">เกิดข้อผิดพลาด: ${escapeHtml(error.message)}</span>`;
  }
}

async function checkHealth() {
  try {
    const data = await apiGet("/api/v1/healthcheck");
    renderMessage(
      "assistant",
      `Healthcheck OK\nservice: ${data.service}\nversion: ${data.version}\ncollection: ${data.collection_name}\nsession retention: ${data.session_retention_days} days`
    );
  } catch (error) {
    console.error(error);
    renderMessage("assistant", `healthcheck failed: ${error.message}`);
  }
}

newChatBtn.addEventListener("click", createNewSession);
sendBtn.addEventListener("click", sendMessage);
healthBtn.addEventListener("click", checkHealth);
deleteSessionBtn.addEventListener("click", deleteCurrentSession);

messageInput.addEventListener("input", autoResizeTextarea);

messageInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  if (e.shiftKey) return;

  e.preventDefault();
  sendMessage();
});

window.addEventListener("DOMContentLoaded", async () => {
  getClientId();
  autoResizeTextarea();

  const sessions = await refreshSessions();

  if (sessions.length > 0) {
    await loadSession(sessions[0].session_id);
  } else {
    await createNewSession();
  }
});