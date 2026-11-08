import { useState } from 'react'
import './App.css'

function App() {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function handleSend() {
    if (loading) return;

    setLoading(true);
    setErrorMessage("");

    const latestConversation = [...chatHistory, { role: "user", content: message }];

    setChatHistory(latestConversation);

    const lastestSixMessages = latestConversation.slice(-6);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ chats: lastestSixMessages })
      });

      if (!response.ok) {
        throw new Error("Failed to get AI response");
      }

      const data = await response.json();

      setChatHistory(prev => {
        return [...prev, { role: "assistant", content: data.answer }];
      });

    } catch (error) {

      console.error(error);
      setErrorMessage("Unable to reach the AI assistant. Please try again.");

    } finally {
      setMessage("");
      setLoading(false);
    }
  }

  return (
    <>
      <h1>AI Support Chatbot</h1>

      <input type='text' value={message} onChange={(e) => setMessage(e.target.value)} />
      <button disabled={loading} onClick={handleSend}>Send</button>

      <div>
        {chatHistory.map((m, index) => (
          <p key={index}>{m.role}: {m.content}</p>
        )
        )}
      </div>

      {loading && <p>AI is thinking...</p>}

      {errorMessage && <p>{errorMessage}</p>}
    </>
  )
}

export default App


