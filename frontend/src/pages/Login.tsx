import { useState } from "react";
import { ApiError, api, setKey, type User } from "../api";

export default function Login({ onLogin }: { onLogin: (u: User) => void }) {
  const [key, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const user = await api.me(key.trim());
      setKey(key.trim());
      onLogin(user);
    } catch (err) {
      setError(err instanceof ApiError && err.status === 401 ? "That API key is not valid." :
        err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <form className="card" onSubmit={submit}>
        <h1 style={{ marginBottom: 6 }}>Document Risk Engine</h1>
        <p className="ink2" style={{ marginTop: 0 }}>
          Contract risk, compliance and anomaly analysis for procurement, finance and compliance teams.
        </p>
        <label className="small ink2" htmlFor="key">API key</label>
        <input id="key" type="password" value={key} onChange={(e) => setValue(e.target.value)} autoComplete="off"
          placeholder="dre_…" style={{ width: "100%", margin: "4px 0 12px" }} autoFocus />
        {error ? <div className="notice bad" style={{ marginBottom: 12 }}>{error}</div> : null}
        <button className="btn primary" disabled={!key.trim() || busy} style={{ width: "100%", justifyContent: "center" }}>
          {busy ? <span className="spinner" /> : null} Sign in
        </button>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Local demo: <code>python -m scripts.init_db</code> creates one user per role and writes their keys to
          <code> artifacts/demo_api_keys.json</code>.
        </p>
      </form>
    </div>
  );
}
