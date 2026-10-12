// Typed API client. Every call goes through /api (proxied to FastAPI) with the user's API key.

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export const LEVELS: RiskLevel[] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];

export interface User { id: number; username: string; full_name: string; role: string; permissions: string[] }

export interface DocumentSummary {
  id: string; filename: string; uploaded_at: string; document_type: string | null; processing_status: string;
  vendor_name: string | null; review_status: string; risk_score: number | null; risk_level: RiskLevel | null;
  extraction_method: string | null;
}

export interface Entity { name: string; value: string | number | null; source_text: string | null; confidence: number; method: string }
export interface Section { name: string; heading: string; start: number; end: number; confidence: number; method: string }

export interface DocumentDetail {
  id: string; filename: string; file_format: string; mime_type: string; file_size: number; uploaded_at: string;
  uploaded_by: string | null; document_type: string | null; doc_type_confidence: number | null; page_count: number | null;
  extraction_method: string | null; extraction_status: string; ocr_pages: number | null; processing_status: string;
  status_message: string | null; vendor_name: string | null; review_status: string; sections: Section[] | null;
  entities: Entity[]; timings_ms: Record<string, number> | null; latest_prediction_id: number | null;
}

export interface Violation { rule_id: string; rule_name: string; severity: RiskLevel; type: string; message: string; remediation: string; evidence: Record<string, unknown> }
export interface Anomaly { type: string; score: number; is_anomaly: boolean; description: string; details: Record<string, unknown> | null }
export interface Action { source: string; priority: string; action: string }

export interface RiskReport {
  document_id: string; prediction_id: number; analyzed_at: string; risk_score: number; risk_level: RiskLevel;
  ml_predicted_class: RiskLevel | null; ml_probabilities: Record<RiskLevel, number> | null;
  high_risk_probability: number | null; anomaly_score: number | null;
  score_components: Record<"ml" | "anomaly" | "compliance" | "business", number | null>;
  score_contributions: Record<string, number> | null; floors_applied: string[] | null;
  compliance_violations: Violation[]; anomalies: Anomaly[]; top_risk_factors: string[];
  recommended_actions: Action[]; needs_review: boolean; review_reasons: string[] | null; degraded: boolean;
  component_errors: Record<string, string> | null; model_version: string | null;
  model_versions: Record<string, string | null> | null; timings_ms: Record<string, number> | null;
}

export interface ShapFactor { rank: number; feature: string; label: string; value: number | null; shap_value: number; direction: string; display: string }
export interface Explanation {
  document_id: string; prediction_id: number;
  shap: { explained_class: RiskLevel | null; output_space: string | null; base_value: number | null; factors: ShapFactor[] };
  llm_status: "pending" | "ready" | "unavailable" | "disabled"; llm_model: string | null; llm_latency_ms: number | null;
  llm_error: string | null; recommended_actions: Action[];
  llm_explanation: null | { summary: string; factor_explanations: { factor: string; explanation: string }[];
    recommended_actions: string[]; grounded: boolean; grounding_warnings: string[] };
}

export interface Feedback {
  id: number; document_id: string; prediction_id: number; reviewer_id: number; created_at: string;
  model_prediction: string | null; model_risk_level: RiskLevel; model_risk_score: number; model_version: string | null;
  decision: "confirm" | "correct"; reviewer_label: RiskLevel; comment: string | null;
  validation_status: "pending" | "validated" | "rejected"; validation_flags: string[] | null;
  validated_by: number | null; validated_at: string | null; validation_comment: string | null;
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public details?: unknown) { super(message); }
}

const KEY_STORAGE = "dre.apiKey";
export function getKey(): string | null {
  try { return localStorage.getItem(KEY_STORAGE); } catch { return null; }
}
export function setKey(key: string | null) {
  try { key ? localStorage.setItem(KEY_STORAGE, key) : localStorage.removeItem(KEY_STORAGE); } catch { /* private mode */ }
}

async function request<T>(path: string, init: RequestInit = {}, key = getKey()): Promise<T> {
  const headers = new Headers(init.headers);
  if (key) headers.set("X-API-Key", key);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let res: Response;
  try {
    res = await fetch(`/api${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "network_error", "The API is not reachable. Is the backend running?");
  }
  const text = await res.text();
  let body: any = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = null; }  // proxies may return plain text/HTML
  if (!res.ok) {
    if (!body && (res.status === 502 || res.status === 503 || res.status === 504 || res.status === 500)) {
      throw new ApiError(res.status, "api_unreachable",
        `The API did not respond (HTTP ${res.status}). It may still be starting up — retry in a few seconds.`);
    }
    const msg = body?.message ?? body?.detail ?? res.statusText;
    throw new ApiError(res.status, body?.error ?? "error", typeof msg === "string" ? msg : JSON.stringify(msg), body?.details);
  }
  return body as T;
}

const qs = (params: Record<string, string | number | undefined | null>) => {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") p.set(k, String(v)); });
  const s = p.toString();
  return s ? `?${s}` : "";
};

export const api = {
  me: (key?: string) => request<User>("/auth/me", {}, key ?? getKey()),
  health: () => request<any>("/health"),
  overview: () => request<any>("/analytics/overview"),
  documents: (params: Record<string, string | number | undefined>) =>
    request<{ total: number; items: DocumentSummary[] }>(`/documents${qs(params)}`),
  document: (id: string) => request<DocumentDetail>(`/documents/${id}`),
  risk: (id: string) => request<RiskReport>(`/documents/${id}/risk`),
  explanation: (id: string) => request<Explanation>(`/documents/${id}/explanation`),
  regenerate: (id: string) => request<any>(`/documents/${id}/explanation/regenerate`, { method: "POST" }),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<{ document_id: string; filename: string }>("/documents/upload", { method: "POST", body: fd });
  },
  analyze: (id: string) => request<any>(`/documents/${id}/analyze?explain=true`, { method: "POST" }),
  feedback: (id: string, body: { decision: string; corrected_label?: string; comment?: string }) =>
    request<Feedback>(`/documents/${id}/feedback`, { method: "POST", body: JSON.stringify(body) }),
  queue: () => request<{ total: number; items: any[] }>("/reviews/queue"),
  feedbackList: (status?: string) => request<Feedback[]>(`/feedback${qs({ validation_status: status })}`),
  validate: (id: number, approve: boolean, comment?: string) =>
    request<Feedback>(`/feedback/${id}/validate`, { method: "POST", body: JSON.stringify({ approve, comment }) }),
  currentModel: () => request<any>("/models/current"),
  modelVersions: () => request<any>("/models/versions"),
  monitoring: () => request<any>("/monitoring/summary"),
  monitoringRun: () => request<any>("/monitoring/run", { method: "POST" }),
  monitoringHistory: () => request<any[]>("/monitoring/history"),
};

export const fmt = {
  pct: (v: number | null | undefined, d = 0) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`),
  num: (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "–" : v.toFixed(d)),
  date: (s: string | null | undefined) => (s ? new Date(s).toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" }) : "–"),
  dateTime: (s: string | null | undefined) => (s ? new Date(s).toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : "–"),
  title: (s: string) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
};
