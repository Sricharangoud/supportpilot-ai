import React, { useEffect, useState } from "react";
import { RefreshCw, Database } from "lucide-react";
import { api } from "../api";

export default function AdminView() {
  const [metrics, setMetrics] = useState(null);
  const [tickets, setTickets] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [subTab, setSubTab] = useState("tickets");
  const [statusMsg, setStatusMsg] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadAll() {
    setLoading(true);
    try {
      const [m, t, c] = await Promise.all([
        api.metrics(),
        api.tickets(),
        api.conversations(),
      ]);
      setMetrics(m);
      setTickets(t);
      setConversations(c);
    } catch (err) {
      setStatusMsg(`Failed to load admin data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAll();
  }, []);

  async function handleReingest() {
    setStatusMsg("Re-indexing knowledge base...");
    try {
      const res = await api.ingest();
      setStatusMsg(`Indexed ${res.docs_indexed} docs (${res.chunks_indexed} chunks).`);
    } catch (err) {
      setStatusMsg(`Ingest failed: ${err.message}`);
    }
  }

  return (
    <div>
      <div className="admin-header">
        <h2 style={{ margin: 0, fontSize: "1.2rem" }}>Operations & Escalation Console</h2>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button className="btn-secondary" onClick={handleReingest}>
            <Database size={15} /> Re-ingest KB
          </button>
          <button className="btn-secondary" onClick={loadAll} disabled={loading}>
            <RefreshCw size={15} /> Refresh
          </button>
        </div>
      </div>

      {statusMsg && (
        <div className="card" style={{ marginBottom: "1rem", padding: "0.75rem 1rem" }}>
          {statusMsg}
        </div>
      )}

      {metrics && (
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Total Requests</div>
            <div className="metric-value">{metrics.total_requests}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Auto-Resolve Rate</div>
            <div className="metric-value">{(metrics.auto_resolve_rate * 100).toFixed(1)}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Escalation Rate</div>
            <div className="metric-value">{(metrics.escalation_rate * 100).toFixed(1)}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Avg Latency</div>
            <div className="metric-value">{metrics.avg_latency_ms.toFixed(0)} ms</div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="nav-tabs" style={{ marginBottom: "1rem" }}>
          <button
            className={`tab-btn ${subTab === "tickets" ? "active" : ""}`}
            onClick={() => setSubTab("tickets")}
          >
            Escalated Tickets ({tickets.length})
          </button>
          <button
            className={`tab-btn ${subTab === "logs" ? "active" : ""}`}
            onClick={() => setSubTab("logs")}
          >
            Conversation Logs ({conversations.length})
          </button>
        </div>

        {subTab === "tickets" ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Intent</th>
                  <th>Reason</th>
                  <th>Confidence</th>
                  <th>Customer Message</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {tickets.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ color: "var(--muted)" }}>
                      No escalated tickets yet.
                    </td>
                  </tr>
                ) : (
                  tickets.map((t) => (
                    <tr key={t.id}>
                      <td>#{t.id}</td>
                      <td>{t.intent}</td>
                      <td>{t.reason}</td>
                      <td>{(t.confidence * 100).toFixed(0)}%</td>
                      <td>{t.user_message}</td>
                      <td>
                        <span className="badge escalate">{t.status}</span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Decision</th>
                  <th>Intent</th>
                  <th>Confidence</th>
                  <th>Latency</th>
                  <th>Message</th>
                  <th>Sources</th>
                </tr>
              </thead>
              <tbody>
                {conversations.length === 0 ? (
                  <tr>
                    <td colSpan={7} style={{ color: "var(--muted)" }}>
                      No conversations logged yet.
                    </td>
                  </tr>
                ) : (
                  conversations.map((c) => (
                    <tr key={c.id}>
                      <td>#{c.id}</td>
                      <td>
                        <span className={`badge ${c.decision}`}>{c.decision}</span>
                      </td>
                      <td>{c.intent}</td>
                      <td>{(c.confidence * 100).toFixed(0)}%</td>
                      <td>{c.latency_ms.toFixed(0)} ms</td>
                      <td>{c.user_message}</td>
                      <td>
                        {c.sources.map((s) => `${s.doc_name}#${s.chunk_id}`).join(", ") || "—"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
