const chat = document.getElementById("chat")


function addMessage(text, type) {
    const div = document.createElement("div")

    div.className = `message ${type}`

    div.innerText = text

    chat.appendChild(div)

    chat.scrollTop = chat.scrollHeight
}


async function sendMessage() {
    const input = document.getElementById("message")

    const text = input.value.trim()

    if (!text) return

    addMessage(text, "user")

    input.value = ""

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

    addMessage(data.response, "ai")
}