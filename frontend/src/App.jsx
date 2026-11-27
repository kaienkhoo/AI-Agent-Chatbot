import { useState } from 'react'
import './App.css'

function App() {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [orderDraft, setOrderDraft] = useState(null);
  const [confirmedOrder, setConfirmedOrder] = useState(null);
  const [orderLoading, setOrderLoading] = useState(false);
  const [orderError, setOrderError] = useState("");

  const suggestedPrompts = [
    "Is product 102 in stock?",
    "What's the status of order 5001?",
    "What's the product in order 5001?"
  ];

  async function handleSend() {
    if (loading || !message.trim()) return;

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

      if (data.order_draft) {
        setOrderDraft(data.order_draft)
        setConfirmedOrder(null)
      }

      setMessage("");

    } catch (error) {

      console.error(error);
      setErrorMessage("Unable to reach the AI assistant. Please try again.");
      setChatHistory(prev => prev.slice(0, -1));

    } finally {
      setLoading(false);
    }
  }

  async function handleConfirmOrder() {
    if (!orderDraft || orderLoading) return;

    setOrderLoading(true);
    setOrderError("");

    try {
      const response = await fetch(`http://localhost:8000/order-drafts/${orderDraft.id}/confirm`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Unable to confirm order")
      }

      const data = await response.json();


      setConfirmedOrder(data);
      setOrderDraft(null);

    } catch (error) {
      console.error(error);
      setOrderError("Could not confirm the order. Please try again.");

    } finally {
      setOrderLoading(false);
    }
  }

  return (
    <>
      <div className="h-dvh bg-[#F7F8F5] text-[#202923]">
        <main className="mx-auto flex h-full max-w-4xl flex-col bg-white">

          <header className="border-b border-[#E4E8E2] px-6 py-4">
            <h1 className="text-base font-semibold">
              Store Support
            </h1>
            <p className="mt-1 text-xs text-[#65736A]">
              Product and order assistance · Automated demo
            </p>
          </header>

          <section className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-6">
            {chatHistory.length === 0 && (
              <div>
                <div className="space-y-2">
                  <div className="mb-6">
                    <h2 className="text-xl font-semibold tracking-tight">
                      How can we help?
                    </h2>
                    <p className="mt-2 text-sm text-[#65736A]">
                      Ask about products, availability, or an existing order.
                    </p>
                  </div>

                  {suggestedPrompts.map((prompt) => (
                    <button
                      className="flex w-full items-center justify-between rounded-lg border border-[#E4E8E2] px-4 py-3 text-left text-sm transition-colors hover:bg-[#F7F8F5]"
                      key={prompt}
                      type="button"
                      onClick={() => setMessage(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {chatHistory.map((m, index) => (
              <div
                key={index}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"
                  }`}
              >
                <p
                  className={`max-w-[85%] whitespace-pre-wrap rounded-xl px-4 py-3 text-sm leading-relaxed ${m.role === "user"
                    ? "bg-[#EAF0EB]"
                    : "border border-[#E4E8E2] bg-white"
                    }`}
                >
                  {m.content}
                </p>
              </div>
            ))}

            {loading && (
              <p className="text-sm text-[#65736A]">
                Checking your request...
              </p>
            )}

            {errorMessage && (
              <p role="alert" className="text-sm text-red-700">
                {errorMessage}
              </p>
            )}

            {orderDraft && (
              <div className="space-y-3 rounded-xl border border-[#E4E8E2] p-4">
                <h3 className="font-semibold">Review your order</h3>

                <p className="text-sm">{orderDraft.product_name}</p>
                <p className="text-sm">Quantity: {orderDraft.quantity}</p>
                <p className="text-sm">Unit price: S${orderDraft.unit_price}</p>
                <p className="font-semibold">Total: S${orderDraft.total_price}</p>

                <p className="text-xs text-[#65736A]">
                  No order has been placed yet.
                </p>

                <div className="flex gap-2">
                  <button
                    type="button"
                    disabled={orderLoading}
                    onClick={() => setOrderDraft(null)}
                    className="rounded-lg border px-4 py-2 text-sm disabled:opacity-50"
                  >
                    Cancel
                  </button>

                  <button
                    type="button"
                    disabled={orderLoading}
                    onClick={handleConfirmOrder}
                    className="rounded-lg bg-[#315A49] px-4 py-2 text-sm text-white disabled:opacity-50"
                  >
                    {orderLoading ? "Processing..." : "Place demo order"}
                  </button>
                </div>
              </div>
            )}

            {confirmedOrder && (
              <div className="rounded-xl border border-[#E4E8E2] p-4 space-y-2">
                <h3 className="font-semibold">Demo order placed!</h3>

                <p className="text-sm">
                  Order ID: #{confirmedOrder.id}
                </p>

                <p className="text-sm">
                  Status: {confirmedOrder.status}
                </p>

                <button
                  type="button"
                  onClick={() => setConfirmedOrder(null)}
                  className="text-sm font-medium text-[#315A49]"
                >
                  Start another demo order
                </button>
              </div>
            )}

            {orderError && (
              <p role="alert" className="text-sm text-red-700">
                {orderError}
              </p>
            )}
          </section>



          <footer className="shrink-0 border-t border-[#E4E8E2] p-4 sm:px-6">
            <div className="flex items-center gap-3 rounded-xl border border-[#E4E8E2] bg-white p-2 focus-within:border-[#315A49]">
              <input
                type="text"
                value={message}
                disabled={loading}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Write a message..."
                className="min-w-0 flex-1 px-2 py-2 text-sm outline-none"
              />

              <button
                disabled={loading || !message.trim()}
                onClick={handleSend}
                className="rounded-lg bg-[#315A49] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[#244535] disabled:cursor-not-allowed disabled:opacity-40"
              >
                Send
              </button>
            </div>
          </footer>
        </main>
      </div>
    </>
  )
}

export default App


