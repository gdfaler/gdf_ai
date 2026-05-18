const chat = document.getElementById("chat")
const input = document.getElementById("message")


input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault()
        sendMessage()
    }
})


function createMessage(text, type) {
    const wrapper = document.createElement("div")

    wrapper.className = `message ${type}`

    const content = document.createElement("div")

    content.className = "message-content"

    content.innerText = text

    wrapper.appendChild(content)

    chat.appendChild(wrapper)

    chat.scrollTop = chat.scrollHeight
}


function quickPrompt(text) {
    input.value = text
    sendMessage()
}


function newChat() {
    chat.innerHTML = ""
}


async function sendMessage() {
    const text = input.value.trim()

    if (!text) return

    document.querySelector(".welcome")?.remove()

    createMessage(text, "user")

    input.value = ""

    const thinking = document.createElement("div")

    thinking.className = "message ai"

    thinking.innerHTML = `
        <div class="message-content">
            Thinking...
        </div>
    `

    chat.appendChild(thinking)

    chat.scrollTop = chat.scrollHeight

    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            message: text
        })
    })

    const data = await response.json()

    thinking.remove()

    createMessage(data.response, "ai")
}