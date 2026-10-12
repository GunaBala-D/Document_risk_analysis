import { createContext, useContext, useEffect, useState } from "react";
import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import { ApiError, api, getKey, setKey, type User } from "./api";
import { ErrorState, Loading } from "./components/ui";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";
import DocumentDetail from "./pages/DocumentDetail";
import Login from "./pages/Login";
import ModelAnalytics from "./pages/ModelAnalytics";
import Monitoring from "./pages/Monitoring";
import ReviewQueue from "./pages/ReviewQueue";

interface AuthCtx { user: User; can: (perm: string) => boolean; logout: () => void }
const Auth = createContext<AuthCtx | null>(null);
export const useAuth = () => useContext(Auth)!;

const ROLE_LABEL: Record<string, string> = {
  compliance_analyst: "Compliance Analyst", finance_analyst: "Finance Analyst", procurement_analyst: "Procurement Analyst",
  risk_manager: "Risk Manager", admin: "Administrator",
};

function Brand() {
  return (
    <div className="brand">
      <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
        <rect x="4" y="3" width="19" height="25" rx="3" fill="var(--ink)" />
        <rect x="8.5" y="9" width="10" height="2" rx="1" fill="var(--surface)" />
        <rect x="8.5" y="13.5" width="10" height="2" rx="1" fill="var(--surface)" />
        <rect x="8.5" y="18" width="6" height="2" rx="1" fill="var(--surface)" />
        <circle cx="24" cy="23" r="6.5" fill="var(--critical)" stroke="var(--surface)" strokeWidth="2" />
        <path d="M24 20v3.5" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
        <circle cx="24" cy="26" r="1" fill="#fff" />
      </svg>
      <div><b>Document Risk Engine</b><span>Vendor contracts · V1</span></div>
    </div>
  );
}

function Shell({ user, logout }: { user: User; logout: () => void }) {
  const [pending, setPending] = useState<number | null>(null);
  useEffect(() => {
    api.overview().then((o) => setPending(o.pending_reviews)).catch(() => setPending(null));
  }, []);
  const can = (p: string) => user.permissions.includes(p);
  return (
    <Auth.Provider value={{ user, can, logout }}>
      <div className="shell">
        <aside className="sidebar">
          <Brand />
          <nav className="nav">
            <div className="nav-label">Workspace</div>
            <NavLink to="/" end>Overview</NavLink>
            <NavLink to="/documents">Documents</NavLink>
            <NavLink to="/reviews">Review queue{pending ? <span className="count">{pending}</span> : null}</NavLink>
            <div className="nav-label">Model operations</div>
            <NavLink to="/models">Model analytics</NavLink>
            <NavLink to="/monitoring">Monitoring</NavLink>
          </nav>
          <div className="sidebar-foot">
            <div><b>{user.full_name}</b></div>
            <div className="role">{ROLE_LABEL[user.role] ?? user.role}</div>
            <button className="btn ghost small" style={{ padding: "6px 0", marginTop: 6 }} onClick={logout}>Sign out</button>
          </div>
        </aside>
        <main className="main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/documents/:id" element={<DocumentDetail />} />
            <Route path="/reviews" element={<ReviewQueue />} />
            <Route path="/models" element={<ModelAnalytics />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Auth.Provider>
  );
}

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(!!getKey());
  const [sessionError, setSessionError] = useState<ApiError | null>(null);

  const checkSession = () => {
    if (!getKey()) { setChecking(false); return; }
    setChecking(true);
    api.me()
      .then((u) => { setUser(u); setSessionError(null); })
      .catch((e: ApiError) => {
        // Only an explicit 401 means the key is invalid. A network/server error (API restarting,
        // database down) must not log the user out.
        if (e.status === 401) { setKey(null); setSessionError(null); } else { setSessionError(e); }
      })
      .finally(() => setChecking(false));
  };
  useEffect(checkSession, []);

  const logout = () => { setKey(null); setUser(null); setSessionError(null); };
  if (checking) return <Loading label="Checking session" />;
  if (sessionError && !user) {
    return (
      <div className="login"><div className="card">
        <ErrorState error={sessionError} onRetry={checkSession} />
        <div style={{ textAlign: "center" }}><button className="btn ghost small" onClick={logout}>Sign in with a different key</button></div>
      </div></div>
    );
  }
  return (
    <BrowserRouter>
      {user ? <Shell user={user} logout={logout} /> : <Login onLogin={setUser} />}
    </BrowserRouter>
  );
}
