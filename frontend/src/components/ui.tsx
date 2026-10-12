import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError, type RiskLevel } from "../api";

/** Fetch helper: keeps the previous data while refetching (no layout jump). */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setData(await fnRef.current());
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(0, "error", String(e)));
    } finally {
      setLoading(false);
    }
  }, []);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { reload(); }, deps);
  return { data, error, loading, reload, setData };
}

const ICONS: Record<RiskLevel, ReactNode> = {
  LOW: <path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />,
  MEDIUM: <><circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M8 5v3.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /><circle cx="8" cy="11" r="1.1" fill="currentColor" /></>,
  HIGH: <><path d="M8 2l6.5 11.5h-13z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" /><path d="M8 6.5v3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /><circle cx="8" cy="11.6" r="1" fill="currentColor" /></>,
  CRITICAL: <><path d="M5.2 1.5h5.6l3.7 3.7v5.6l-3.7 3.7H5.2l-3.7-3.7V5.2z" fill="currentColor" /><path d="M8 4.5v4" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" /><circle cx="8" cy="11" r="1" fill="#fff" /></>,
};

export function RiskBadge({ level, score }: { level: RiskLevel | null | undefined; score?: number | null }) {
  if (!level) return <span className="badge neutral">Not scored</span>;
  return (
    <span className={`badge ${level}`} title={`Risk level ${level}`}>
      <svg viewBox="0 0 16 16" aria-hidden="true">{ICONS[level]}</svg>
      {level}{score !== undefined && score !== null ? <span className="tabular" style={{ fontWeight: 500 }}>· {score}</span> : null}
    </span>
  );
}

export function Kpi({ label, value, sub, icon }: { label: string; value: ReactNode; sub?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="kpi">
      <div className="label">{icon}{label}</div>
      <div className="value tabular">{value}</div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return <div className="state"><span className="spinner" /> <span style={{ marginLeft: 8 }}>{label}…</span></div>;
}

export function ErrorState({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div className="state error" role="alert">
      <b>{error.status === 0 ? "Can't reach the API" : `Request failed (${error.status || error.code})`}</b>
      {error.message}
      {onRetry ? <div style={{ marginTop: 12 }}><button className="btn" onClick={onRetry}>Retry</button></div> : null}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="state"><b>{title}</b>{children}</div>;
}

export function Card({ title, sub, action, children, id, className = "" }:
  { title?: ReactNode; sub?: ReactNode; action?: ReactNode; children: ReactNode; id?: string; className?: string }) {
  return (
    <section className={`card ${className}`} id={id}>
      {title || action ? (
        <div className="card-head">
          <div>{title ? <h2>{title}</h2> : null}{sub ? <p>{sub}</p> : null}</div>
          {action}
        </div>
      ) : null}
      {children}
    </section>
  );
}

export function Notice({ kind = "info", children }: { kind?: "info" | "warn" | "bad" | "good"; children: ReactNode }) {
  return <div className={`notice ${kind}`}>{children}</div>;
}
