const STORAGE_KEY = "gdf_ai_state"

const chatEl = document.getElementById("chat")
const chatListEl = document.getElementById("chatList")
const input = document.getElementById("message")
const chatTitleEl = document.getElementById("chatTitle")
const chatSubtitleEl = document.getElementById("chatSubtitle")
const settingsModal = document.getElementById("settingsModal")

const DEFAULT_SETTINGS = {
    maxTokens: 80,
    temperature: 0,
    typingAnimation: true,
    fontSize: 15,
    enterToSend: true,
    lightTheme: false
}

const WELCOME_HTML = `
    <div class="welcome">
        <div class="welcome-badge">Neural Assistant</div>
        <h1>How can I help you today?</h1>
        <p class="welcome-sub">Ask anything — code, design, AI, or creative ideas</p>
        <div class="welcome-grid">
            <button onclick="quickPrompt('Напиши сайт на Flask')">
                <span class="card-icon">⚡</span>
                <span class="card-title">Flask backend</span>
                <span class="card-desc">REST API & routes</span>
            </button>
            <button onclick="quickPrompt('Объясни attention')">
                <span class="card-icon">🧠</span>
                <span class="card-title">Explain transformers</span>
                <span class="card-desc">Attention mechanism</span>
            </button>
            <button onclick="quickPrompt('Напиши красивый UI')">
                <span class="card-icon">🎨</span>
                <span class="card-title">UI design</span>
                <span class="card-desc">Modern interfaces</span>
            </button>
            <button onclick="quickPrompt('Помоги оптимизировать модель')">
                <span class="card-icon">🚀</span>
                <span class="card-title">AI optimization</span>
                <span class="card-desc">Training & inference</span>
            </button>
        </div>
    </div>
`

let state = loadState()
let typingAbort = null
let isGenerating = false


function loadState() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (!raw) return createDefaultState()

        const parsed = JSON.parse(raw)
        if (!parsed.chats?.length) return createDefaultState()

        return {
            chats: parsed.chats,
            activeChatId: parsed.activeChatId || parsed.chats[0].id,
            settings: { ...DEFAULT_SETTINGS, ...parsed.settings }
        }
    } catch {
        return createDefaultState()
    }
}


function createDefaultState() {
    const chat = createChatObject()
    return {
        chats: [chat],
        activeChatId: chat.id,
        settings: { ...DEFAULT_SETTINGS }
    }
}


function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
}


function createChatObject() {
    const now = Date.now()
    return {
        id: crypto.randomUUID(),
        title: "New chat",
        messages: [],
        createdAt: now,
        updatedAt: now
    }
}


function getActiveChat() {
    return state.chats.find(c => c.id === state.activeChatId)
}


function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}


function scrollToBottom() {
    chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" })
}


function truncateTitle(text) {
    const clean = text.trim().replace(/\s+/g, " ")
    if (clean.length <= 32) return clean
    return clean.slice(0, 32) + "…"
}


function formatDate(ts) {
    const d = new Date(ts)
    const now = new Date()
    const sameDay = d.toDateString() === now.toDateString()

    if (sameDay) {
        return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    }

    return d.toLocaleDateString([], { day: "numeric", month: "short" })
}


function applySettings() {
    const s = state.settings
    document.body.classList.toggle("light-theme", s.lightTheme)
    document.documentElement.style.setProperty("--msg-font-size", s.fontSize + "px")
    document.querySelectorAll(".setting-segment button").forEach(btn => {
        btn.classList.toggle("active", Number(btn.dataset.font) === s.fontSize)
    })
}


function renderChatList() {
    const sorted = [...state.chats].sort((a, b) => b.updatedAt - a.updatedAt)

    chatListEl.innerHTML = sorted.map(item => `
        <div class="chat-item ${item.id === state.activeChatId ? "active" : ""}"
             onclick="switchChat('${item.id}')">
            <span class="chat-item-dot"></span>
            <div class="chat-item-body">
                <span class="chat-item-title">${escapeHtml(item.title)}</span>
                <span class="chat-item-date">${formatDate(item.updatedAt)}</span>
            </div>
            <button class="chat-item-delete"
                    onclick="deleteChat(event, '${item.id}')"
                    aria-label="Delete chat">×</button>
        </div>
    `).join("")
}


function escapeHtml(text) {
    const div = document.createElement("div")
    div.textContent = text
    return div.innerHTML
}


function updateTopbar() {
    const active = getActiveChat()
    if (!active) return

    if (active.messages.length === 0) {
        chatTitleEl.textContent = "GDF AI"
        chatSubtitleEl.textContent = "Local neural network powered by PyTorch"
    } else {
        chatTitleEl.textContent = active.title
        chatSubtitleEl.textContent = `${active.messages.length} messages · ${formatDate(active.updatedAt)}`
    }
}


function renderMessages(animate = false) {
    const active = getActiveChat()
    if (!active) return

    if (active.messages.length === 0) {
        chatEl.innerHTML = WELCOME_HTML
        updateTopbar()
        return
    }

    chatEl.innerHTML = ""

    for (const msg of active.messages) {
        renderMessage(msg.content, msg.role, false)
    }

    updateTopbar()
    scrollToBottom()
}


function renderMessage(text, type, animate = true) {
    document.querySelector(".welcome")?.remove()

    const wrapper = document.createElement("div")
    wrapper.className = `message ${type}`
    if (!animate) wrapper.style.animation = "none"

    const avatar = document.createElement("div")
    avatar.className = "message-avatar"
    avatar.textContent = type === "user" ? "You" : "AI"

    const body = document.createElement("div")
    body.className = "message-body"

    const content = document.createElement("div")
    content.className = "message-content"
    content.textContent = text

    body.appendChild(content)
    wrapper.appendChild(avatar)
    wrapper.appendChild(body)
    chatEl.appendChild(wrapper)

    return content
}


function createMessageShell(type) {
    document.querySelector(".welcome")?.remove()
    return renderMessage("", type, true)
}


async function typeMessage(text) {
    if (!state.settings.typingAnimation) {
        renderMessage(text, "ai", true)
        scrollToBottom()
        return text
    }

    if (typingAbort) typingAbort.aborted = true
    typingAbort = { aborted: false }
    const token = typingAbort

    const content = createMessageShell("ai")
    const cursor = document.createElement("span")
    cursor.className = "typing-cursor"
    content.appendChild(cursor)

    let displayed = ""

    for (let i = 0; i < text.length; i++) {
        if (token.aborted) return displayed

        displayed += text[i]
        content.textContent = displayed
        content.appendChild(cursor)
        scrollToBottom()

        const ch = text[i]
        let pause = 14
        if (ch === " ") pause = 6
        else if (ch === "\n") pause = 40
        else if (".!?".includes(ch)) pause = 80
        else if (",;:".includes(ch)) pause = 50

        await delay(pause)
    }

    content.classList.add("typing-done")
    cursor.remove()
    scrollToBottom()
    return displayed
}


function showThinking() {
    const content = createMessageShell("ai")
    content.innerHTML = `<div class="thinking-dots"><span></span><span></span><span></span></div>`
    scrollToBottom()
    return content.closest(".message")
}


function addMessageToState(role, content) {
    const active = getActiveChat()
    if (!active) return

    active.messages.push({ role, content })
    active.updatedAt = Date.now()

    if (role === "user" && active.messages.filter(m => m.role === "user").length === 1) {
        active.title = truncateTitle(content)
    }

    saveState()
    renderChatList()
    updateTopbar()
}


function newChat() {
    if (typingAbort) typingAbort.aborted = true

    const empty = state.chats.find(c => c.messages.length === 0)
    if (empty) {
        switchChat(empty.id)
        return
    }

    const chat = createChatObject()
    state.chats.unshift(chat)
    state.activeChatId = chat.id
    saveState()
    renderChatList()
    renderMessages()
    input.focus()
}


function switchChat(id) {
    if (isGenerating || id === state.activeChatId) return

    if (typingAbort) typingAbort.aborted = true

    state.activeChatId = id
    saveState()
    renderChatList()
    renderMessages()
}


function deleteChat(event, id) {
    event.stopPropagation()

    if (state.chats.length === 1) {
        state.chats = [createChatObject()]
        state.activeChatId = state.chats[0].id
    } else {
        state.chats = state.chats.filter(c => c.id !== id)
        if (state.activeChatId === id) {
            state.activeChatId = state.chats[0].id
        }
    }

    saveState()
    renderChatList()
    renderMessages()
}


function clearAllChats() {
    if (!confirm("Delete all chats? This cannot be undone.")) return

    const chat = createChatObject()
    state.chats = [chat]
    state.activeChatId = chat.id
    saveState()
    renderChatList()
    renderMessages()
    closeSettings()
}


function quickPrompt(text) {
    input.value = text
    sendMessage()
}


function openSettings() {
    const s = state.settings

    document.getElementById("maxTokens").value = s.maxTokens
    document.getElementById("maxTokensValue").textContent = s.maxTokens
    document.getElementById("temperature").value = Math.round(s.temperature * 100)
    document.getElementById("temperatureValue").textContent = s.temperature.toFixed(1)
    document.getElementById("typingAnimation").checked = s.typingAnimation
    document.getElementById("enterToSend").checked = s.enterToSend
    document.getElementById("lightTheme").checked = s.lightTheme

    document.querySelectorAll(".setting-segment button").forEach(btn => {
        btn.classList.toggle("active", Number(btn.dataset.font) === s.fontSize)
    })

    settingsModal.classList.add("open")
}


function closeSettings() {
    settingsModal.classList.remove("open")
}


function closeSettingsOnBackdrop(event) {
    if (event.target === settingsModal) closeSettings()
}


function setFontSize(size) {
    document.querySelectorAll(".setting-segment button").forEach(btn => {
        btn.classList.toggle("active", Number(btn.dataset.font) === size)
    })
}


function saveSettings() {
    state.settings = {
        maxTokens: Number(document.getElementById("maxTokens").value),
        temperature: Number(document.getElementById("temperature").value) / 100,
        typingAnimation: document.getElementById("typingAnimation").checked,
        enterToSend: document.getElementById("enterToSend").checked,
        lightTheme: document.getElementById("lightTheme").checked,
        fontSize: Number(
            document.querySelector(".setting-segment button.active")?.dataset.font || 15
        )
    }

    saveState()
    applySettings()
    closeSettings()
}


async function sendMessage() {
    const text = input.value.trim()
    if (!text || isGenerating) return

    input.value = ""
    input.style.height = "auto"

    addMessageToState("user", text)
    renderMessage(text, "user", true)

    isGenerating = true
    const thinking = showThinking()

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text,
                max_tokens: state.settings.maxTokens,
                temperature: state.settings.temperature
            })
        })

        if (!response.ok) throw new Error(`HTTP ${response.status}`)

        const data = await response.json()
        thinking.remove()

        const reply = data.response || "Empty response"
        const displayed = await typeMessage(reply)
        addMessageToState("ai", displayed || reply)
    } catch {
        thinking.remove()
        const errMsg = "Error: failed to get response"
        await typeMessage(errMsg)
        addMessageToState("ai", errMsg)
    } finally {
        isGenerating = false
    }
}


input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && state.settings.enterToSend) {
        e.preventDefault()
        sendMessage()
    }
})

input.addEventListener("input", () => {
    input.style.height = "auto"
    input.style.height = Math.min(input.scrollHeight, 160) + "px"
})

document.getElementById("maxTokens").addEventListener("input", (e) => {
    document.getElementById("maxTokensValue").textContent = e.target.value
})

document.getElementById("temperature").addEventListener("input", (e) => {
    document.getElementById("temperatureValue").textContent =
        (Number(e.target.value) / 100).toFixed(1)
})

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSettings()
})


applySettings()
renderChatList()
renderMessages()
