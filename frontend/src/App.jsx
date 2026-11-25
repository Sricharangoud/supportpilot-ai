import React, { useEffect, useState } from "react";
import { MessageSquare, LayoutDashboard, Compass } from "lucide-react";
import ChatView from "./components/ChatView.jsx";
import AdminView from "./components/AdminView.jsx";
import { api } from "./api";

export default function App() {
  const [tab, setTab] = useState("chat");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setHealth({ status: "offline" }));
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Compass size={22} color="#0f766e" />
          <span>SupportPilot AI</span>
          <span className="brand-sub">Northstar Shop</span>
        </div>

        <nav className="nav-tabs">
          <button
            className={`tab-btn ${tab === "chat" ? "active" : ""}`}
            onClick={() => setTab("chat")}
          >
            <MessageSquare size={15} /> Support Chat
          </button>
          <button
            className={`tab-btn ${tab === "admin" ? "active" : ""}`}
            onClick={() => setTab("admin")}
          >
            <LayoutDashboard size={15} /> Admin Console
          </button>
        </nav>

        <div className="health-pill">
          <span className={`dot ${health?.status === "ok" ? "ok" : "err"}`} />
          {health?.status === "ok"
            ? `API Online • ${health.indexed_chunks} chunks`
            : "API Offline"}
        </div>
      </header>

      <main className="main-container">
        {tab === "chat" ? <ChatView /> : <AdminView />}
      </main>
    </div>
  );
}
