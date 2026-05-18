const STORAGE_KEY = "gdf_ai_state"

const chatEl = document.getElementById("chat")
const chatListEl = document.getElementById("chatList")
const input = document.getElementById("message")
const sendBtn = document.getElementById("sendBtn")
const stopBtn = document.getElementById("stopBtn")
const sendControls = document.getElementById("sendControls")
const chatTitleEl = document.getElementById("chatTitle")
const chatSubtitleEl = document.getElementById("chatSubtitle")
const settingsModal = document.getElementById("settingsModal")
const sidebar = document.getElementById("sidebar")
const sidebarOverlay = document.getElementById("sidebarOverlay")
const toastEl = document.getElementById("toast")
const renameBtn = document.getElementById("renameBtn")
const renameModal = document.getElementById("renameModal")
const renameModalInput = document.getElementById("renameModalInput")
const exportBtn = document.getElementById("exportBtn")
const exportJsonBtn = document.getElementById("exportJsonBtn")
const regenBtn = document.getElementById("regenBtn")

const DEFAULT_SETTINGS = {
    maxTokens: 80,
    temperature: 0,
    typingAnimation: true,
    fontSize: 15,
    enterToSend: true,
    lightTheme: false
}

const WELCOME_HTML = document.getElementById("welcome")?.outerHTML || ""

let state = loadState()
let typingAbort = null
let fetchAbort = null
let isGenerating = false
let thinkingTimer = null
let thinkingStart = 0
let renameTargetId = null

if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true })
}


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
    return { chats: [chat], activeChatId: chat.id, settings: { ...DEFAULT_SETTINGS } }
}


function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
}


function createChatObject() {
    const now = Date.now()
    return { id: crypto.randomUUID(), title: "New chat", messages: [], createdAt: now, updatedAt: now }
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
    return clean.length <= 32 ? clean : clean.slice(0, 32) + "…"
}


function formatDate(ts) {
    const d = new Date(ts)
    const now = new Date()
    if (d.toDateString() === now.toDateString()) {
        return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    }
    return d.toLocaleDateString([], { day: "numeric", month: "short" })
}


function escapeHtml(text) {
    const div = document.createElement("div")
    div.textContent = text
    return div.innerHTML
}


function renderMarkdown(text) {
    if (typeof marked === "undefined") return escapeHtml(text).replace(/\n/g, "<br>")
    return marked.parse(text)
}


function showToast(msg) {
    toastEl.textContent = msg
    toastEl.classList.add("show")
    clearTimeout(showToast._t)
    showToast._t = setTimeout(() => toastEl.classList.remove("show"), 2000)
}


function applySettings() {
    const s = state.settings
    document.body.classList.toggle("light-theme", s.lightTheme)
    document.documentElement.style.setProperty("--msg-font-size", s.fontSize + "px")
    document.querySelectorAll(".setting-segment button").forEach(btn => {
        btn.classList.toggle("active", Number(btn.dataset.font) === s.fontSize)
    })
}


function setGenerating(active) {
    isGenerating = active
    sendControls.classList.toggle("generating", active)
    sendBtn.tabIndex = active ? -1 : 0
    stopBtn.tabIndex = active ? 0 : -1
}


function updateTopbarActions() {
    const active = getActiveChat()
    const hasMessages = active && active.messages.length > 0
    const canRegen = hasMessages && active.messages.at(-1)?.role === "ai"

    renameBtn.hidden = false
    exportBtn.hidden = !hasMessages
    exportJsonBtn.hidden = !hasMessages
    regenBtn.hidden = !canRegen
}


function updateTopbar() {
    const active = getActiveChat()
    if (!active) return

    const titleArea = document.querySelector(".topbar-title-area")
    if (titleArea) titleArea.classList.toggle("has-chat", active.messages.length > 0)

    if (active.messages.length === 0) {
        chatTitleEl.textContent = "GDF AI"
        chatSubtitleEl.textContent = "Local neural network powered by PyTorch"
        chatTitleEl.removeAttribute("title")
    } else {
        chatTitleEl.textContent = active.title
        chatSubtitleEl.textContent = `${active.messages.length} messages · ${formatDate(active.updatedAt)}`
        chatTitleEl.title = "Double-click to rename"
    }

    updateTopbarActions()
}


function renderChatList() {
    const sorted = [...state.chats].sort((a, b) => b.updatedAt - a.updatedAt)

    chatListEl.innerHTML = sorted.map((item, i) => `
        <div class="chat-item ${item.id === state.activeChatId ? "active" : ""}"
             data-id="${item.id}"
             style="animation-delay:${i * 0.04}s"
             onclick="switchChat('${item.id}')">
            <span class="chat-item-dot"></span>
            <div class="chat-item-body">
                <span class="chat-item-title" ondblclick="openRenameModal('${item.id}', event)">${escapeHtml(item.title)}</span>
                <span class="chat-item-date">${formatDate(item.updatedAt)}</span>
            </div>
            <button class="chat-item-rename" onclick="openRenameModal('${item.id}', event)" title="Rename">✎</button>
            <button class="chat-item-delete" onclick="deleteChat(event, '${item.id}')" title="Delete">×</button>
        </div>
    `).join("")
}


function removeWelcome() {
    const welcome = document.querySelector(".welcome")
    if (!welcome) return
    welcome.classList.add("welcome-leaving")
    setTimeout(() => welcome.remove(), 280)
}


function buildMessageActions(type, text, isLastAi) {
    if (type !== "ai") return null

    const actions = document.createElement("div")
    actions.className = "message-actions"

    const copyBtn = document.createElement("button")
    copyBtn.className = "msg-action-btn"
    copyBtn.title = "Copy"
    copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" stroke-width="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" stroke="currentColor" stroke-width="2"/></svg>`
    copyBtn.onclick = () => copyText(text)

    actions.appendChild(copyBtn)

    if (isLastAi) {
        const regenBtnEl = document.createElement("button")
        regenBtnEl.className = "msg-action-btn"
        regenBtnEl.title = "Regenerate"
        regenBtnEl.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M20.49 9A9 9 0 005.64 5.64L1 10M3.51 15a9 9 0 0014.85 3.36L23 14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`
        regenBtnEl.onclick = () => regenerateLast()
        actions.appendChild(regenBtnEl)
    }

    return actions
}


function renderMessage(text, type, animate = true, opts = {}) {
    if (type === "user") removeWelcome()

    const wrapper = document.createElement("div")
    wrapper.className = `message ${type}`
    if (!animate) wrapper.style.animation = "none"

    const avatar = document.createElement("div")
    avatar.className = "message-avatar"
    avatar.textContent = type === "user" ? "You" : "AI"

    const body = document.createElement("div")
    body.className = "message-body"

    const content = document.createElement("div")
    content.className = "message-content markdown-body"

    if (type === "ai") {
        content.innerHTML = renderMarkdown(text)
    } else {
        content.textContent = text
    }

    body.appendChild(content)

    const actions = buildMessageActions(type, text, opts.isLastAi)
    if (actions) body.appendChild(actions)

    wrapper.appendChild(avatar)
    wrapper.appendChild(body)
    chatEl.appendChild(wrapper)

    return { wrapper, content }
}


function renderMessages() {
    const active = getActiveChat()
    if (!active) return

    if (active.messages.length === 0) {
        chatEl.innerHTML = WELCOME_HTML
        updateTopbar()
        return
    }

    chatEl.innerHTML = ""
    const lastAiIdx = active.messages.map((m, i) => m.role === "ai" ? i : -1).filter(i => i >= 0).pop()

    active.messages.forEach((msg, i) => {
        renderMessage(msg.content, msg.role, false, { isLastAi: i === lastAiIdx })
    })

    updateTopbar()
    scrollToBottom()
}


function createMessageShell(type) {
    removeWelcome()
    return renderMessage("", type, true).content
}


async function typeMessage(text, contentEl) {
    if (!state.settings.typingAnimation) {
        contentEl.innerHTML = renderMarkdown(text)
        contentEl.classList.add("markdown-body")
        scrollToBottom()
        return text
    }

    if (typingAbort) typingAbort.aborted = true
    typingAbort = { aborted: false }
    const token = typingAbort

    contentEl.textContent = ""
    contentEl.classList.remove("markdown-body")
    const cursor = document.createElement("span")
    cursor.className = "typing-cursor"
    contentEl.appendChild(cursor)

    let displayed = ""

    for (let i = 0; i < text.length; i++) {
        if (token.aborted) break

        displayed += text[i]
        contentEl.textContent = displayed
        contentEl.appendChild(cursor)
        scrollToBottom()

        const ch = text[i]
        let pause = 14
        if (ch === " ") pause = 6
        else if (ch === "\n") pause = 40
        else if (".!?".includes(ch)) pause = 80
        else if (",;:".includes(ch)) pause = 50

        await delay(pause)
    }

    contentEl.classList.add("typing-done", "markdown-body")
    contentEl.innerHTML = renderMarkdown(displayed)
    scrollToBottom()
    return displayed
}


function startThinkingTimer(contentEl) {
    thinkingStart = Date.now()
    contentEl.innerHTML = `
        <div class="thinking-shimmer">
            <div class="shimmer-lines">
                <div class="shimmer-line"></div>
                <div class="shimmer-line short"></div>
                <div class="shimmer-line medium"></div>
            </div>
            <span class="thinking-time">0.0s</span>
        </div>
    `
    const timeEl = contentEl.querySelector(".thinking-time")
    thinkingTimer = setInterval(() => {
        const sec = ((Date.now() - thinkingStart) / 1000).toFixed(1)
        if (timeEl) timeEl.textContent = sec + "s"
    }, 100)
}


function stopThinkingTimer() {
    clearInterval(thinkingTimer)
    thinkingTimer = null
}


function showThinking() {
    const content = createMessageShell("ai")
    startThinkingTimer(content)
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


function stopGeneration() {
    if (typingAbort) typingAbort.aborted = true
    if (fetchAbort) fetchAbort.abort()
    stopThinkingTimer()

    const thinking = document.querySelector(".thinking-shimmer")
    if (thinking) thinking.closest(".message")?.remove()

    setGenerating(false)
}


async function requestAI(userText, thinkingEl) {
    fetchAbort = new AbortController()

    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            message: userText,
            max_tokens: state.settings.maxTokens,
            temperature: state.settings.temperature
        }),
        signal: fetchAbort.signal
    })

    if (!response.ok) throw new Error(`HTTP ${response.status}`)

    stopThinkingTimer()
    thinkingEl.remove()

    const data = await response.json()
    return data.response || "Empty response"
}


async function generateAIReply(userText) {
    const thinking = showThinking()
    setGenerating(true)

    try {
        const reply = await requestAI(userText, thinking)
        const { content } = renderMessage("", "ai", true, { isLastAi: true })
        const displayed = await typeMessage(reply, content)
        addMessageToState("ai", displayed || reply)
        renderMessages()
    } catch (err) {
        stopThinkingTimer()
        thinking?.remove()

        if (err.name === "AbortError") return

        const errMsg = "Error: failed to get response"
        const { content } = renderMessage("", "ai", true)
        await typeMessage(errMsg, content)
        addMessageToState("ai", errMsg)
    } finally {
        fetchAbort = null
        setGenerating(false)
    }
}


async function sendMessage() {
    const text = input.value.trim()
    if (!text || isGenerating) return

    input.value = ""
    input.style.height = "auto"

    addMessageToState("user", text)
    renderMessage(text, "user", true)

    await generateAIReply(text)
}


async function regenerateLast() {
    if (isGenerating) return

    const active = getActiveChat()
    if (!active || active.messages.at(-1)?.role !== "ai") return

    const lastUser = [...active.messages].reverse().find(m => m.role === "user")
    if (!lastUser) return

    active.messages.pop()
    saveState()
    renderMessages()

    await generateAIReply(lastUser.content)
}


function copyText(text) {
    navigator.clipboard.writeText(text).then(() => showToast("Copied!"))
}


function openRenameModal(chatId, ev) {
    if (ev) ev.stopPropagation()

    closeSettings()

    const id = chatId ?? state.activeChatId
    const chat = state.chats.find(c => c.id === id)
    if (!chat) return

    renameTargetId = id
    renameModalInput.value = chat.title
    renameModal.classList.add("open")

    requestAnimationFrame(() => {
        renameModalInput.focus()
        renameModalInput.select()
    })
}


function closeRenameModal() {
    renameModal.classList.remove("open")
    renameTargetId = null
}


function closeRenameModalBackdrop(event) {
    if (event.target === renameModal) closeRenameModal()
}


function submitRenameModal() {
    const id = renameTargetId
    if (!id) return

    const newTitle = renameModalInput.value.trim()
    if (!newTitle) {
        closeRenameModal()
        return
    }

    const chat = state.chats.find(c => c.id === id)
    if (chat) {
        chat.title = newTitle
        chat.updatedAt = Date.now()
        saveState()
    }

    closeRenameModal()
    renderChatList()
    updateTopbar()
    showToast("Chat renamed")
}


function exportChat(format) {
    const chat = getActiveChat()
    if (!chat || !chat.messages.length) return

    const safeName = chat.title.replace(/[^\w\s-]/g, "").trim() || "chat"

    if (format === "json") {
        downloadFile(JSON.stringify(chat, null, 2), `${safeName}.json`, "application/json")
    } else {
        const lines = chat.messages.map(m => {
            const label = m.role === "user" ? "You" : "GDF AI"
            return `${label}:\n${m.content}`
        })
        downloadFile(lines.join("\n\n---\n\n"), `${safeName}.txt`, "text/plain")
    }

    showToast(`Exported as ${format.toUpperCase()}`)
}


function downloadFile(content, filename, mime) {
    const blob = new Blob([content], { type: mime })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
}


function newChat() {
    stopGeneration()
    const empty = state.chats.find(c => c.messages.length === 0)
    if (empty) { switchChat(empty.id); return }

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
    if (renameModal.classList.contains("open")) closeRenameModal()
    stopGeneration()
    state.activeChatId = id
    saveState()
    renderChatList()
    renderMessages()
    closeSidebar()
}


function deleteChat(event, id) {
    event.stopPropagation()
    if (state.chats.length === 1) {
        state.chats = [createChatObject()]
        state.activeChatId = state.chats[0].id
    } else {
        state.chats = state.chats.filter(c => c.id !== id)
        if (state.activeChatId === id) state.activeChatId = state.chats[0].id
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
    closeRenameModal()
}


function toggleSidebar() {
    sidebar.classList.toggle("open")
    sidebarOverlay.classList.toggle("open")
}


function closeSidebar() {
    sidebar.classList.remove("open")
    sidebarOverlay.classList.remove("open")
}


function quickPrompt(text) {
    input.value = text
    sendMessage()
}


function openSettings() {
    closeRenameModal()
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


function closeSettings() { settingsModal.classList.remove("open") }
function closeSettingsOnBackdrop(e) { if (e.target === settingsModal) closeSettings() }


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
        fontSize: Number(document.querySelector(".setting-segment button.active")?.dataset.font || 15)
    }
    saveState()
    applySettings()
    closeSettings()
}


input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey && state.settings.enterToSend && !isGenerating) {
        e.preventDefault()
        sendMessage()
    }
})

input.addEventListener("input", () => {
    input.style.height = "auto"
    input.style.height = Math.min(input.scrollHeight, 160) + "px"
})

document.getElementById("maxTokens").addEventListener("input", e => {
    document.getElementById("maxTokensValue").textContent = e.target.value
})

document.getElementById("temperature").addEventListener("input", e => {
    document.getElementById("temperatureValue").textContent = (Number(e.target.value) / 100).toFixed(1)
})

document.addEventListener("keydown", e => {
    if (e.key === "Escape") {
        if (renameModal.classList.contains("open")) closeRenameModal()
        else { closeSettings(); closeSidebar() }
    }
})

renameModalInput.addEventListener("keydown", e => {
    if (e.key === "Enter") {
        e.preventDefault()
        submitRenameModal()
    }
})


applySettings()
renderChatList()
renderMessages()

chatTitleEl.addEventListener("dblclick", () => {
    const active = getActiveChat()
    if (active?.messages.length) openRenameModal()
})
