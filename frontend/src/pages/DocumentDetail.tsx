import { Fragment, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useAuth } from "../App";
import {
  ApiError, LEVELS, api, fmt,
  type DocumentDetail as Doc, type Explanation, type Feedback, type RiskLevel, type RiskReport,
} from "../api";
import { DivergingBars, HBars, StackedScore } from "../components/charts";
import { Card, Empty, ErrorState, Loading, Notice, RiskBadge, useAsync } from "../components/ui";

const LEVEL_COLOR: Record<string, string> = {
  LOW: "var(--good)", MEDIUM: "var(--warning)", HIGH: "var(--serious)", CRITICAL: "var(--critical)",
};
const COMPONENTS = [
  { key: "ml", label: "ML risk model", color: "var(--s1)" },
  { key: "anomaly", label: "Anomaly detection", color: "var(--s2)" },
  { key: "compliance", label: "Compliance rules", color: "var(--s3)" },
  { key: "business", label: "Business exposure", color: "var(--s4)" },
] as const;
const ENTITY_LABEL: Record<string, string> = {
  vendor_name: "Vendor", customer_name: "Customer", contract_number: "Contract number", signing_date: "Signed",
  start_date: "Start date", end_date: "End date", contract_value: "Contract value", currency: "Currency",
  payment_days: "Payment terms (days)", termination_notice_days: "Termination notice (days)", penalty_pct: "Penalty (% of value)",
  renewal_months: "Auto-renewal (months)", governing_law: "Governing law", jurisdiction: "Courts", liability_cap: "Liability",
};

export default function DocumentDetail() {
  const { id = "" } = useParams();
  const doc = useAsync(() => api.document(id), [id]);
  const [risk, setRisk] = useState<RiskReport | null>(null);
  const [riskErr, setRiskErr] = useState<ApiError | null>(null);
  const [exp, setExp] = useState<Explanation | null>(null);

  const loadRisk = async () => {
    try {
      setRisk(await api.risk(id));
      setExp(await api.explanation(id));
      setRiskErr(null);
    } catch (e) { setRiskErr(e as ApiError); }
  };
  useEffect(() => { if (doc.data?.processing_status === "analyzed") loadRisk(); }, [doc.data?.processing_status, id]);

  // Poll for the optional LLM explanation (generated in the background).
  useEffect(() => {
    if (exp?.llm_status !== "pending") return;
    let tries = 0;
    const t = setInterval(async () => {
      tries += 1;
      try {
        const e = await api.explanation(id);
        setExp(e);
        if (e.llm_status !== "pending" || tries > 45) clearInterval(t);
      } catch { clearInterval(t); }
    }, 4000);
    return () => clearInterval(t);
  }, [exp?.llm_status, id]);

  if (doc.error) return <ErrorState error={doc.error} onRetry={doc.reload} />;
  if (!doc.data) return <Loading />;
  const d = doc.data;

  return (
    <>
      <div className="page-head">
        <div>
          <div className="small muted"><Link to="/documents">Documents</Link> / <span className="mono">{d.id}</span></div>
          <h1 style={{ marginTop: 4 }}>{d.filename}</h1>
          <p>{d.vendor_name ?? "Vendor not identified"} · uploaded {fmt.dateTime(d.uploaded_at)}{d.uploaded_by ? ` by ${d.uploaded_by}` : ""}</p>
        </div>
        <div className="row">
          {risk ? <RiskBadge level={risk.risk_level} score={risk.risk_score} /> : null}
          <span className="pill">Review: {fmt.title(d.review_status)}</span>
          {risk?.model_version ? <span className="pill">Model v{risk.model_version}</span> : null}
        </div>
      </div>

      {d.processing_status === "unsupported_type" ? (
        <Card><Notice kind="info"><div><b>{fmt.title(d.document_type ?? "unknown")} detected.</b> {d.status_message}</div></Notice>
          <EntitySection d={d} /></Card>
      ) : d.processing_status === "failed" ? (
        <Card><Notice kind="bad"><div><b>Extraction failed.</b> {d.status_message}</div></Notice></Card>
      ) : d.processing_status !== "analyzed" ? (
        <Card><AnalyzeNow id={d.id} onDone={doc.reload} /></Card>
      ) : riskErr ? <ErrorState error={riskErr} onRetry={loadRisk} /> : !risk || !exp ? <Loading label="Loading analysis" /> : (
        <div className="detail">
          <FlowRail d={d} risk={risk} exp={exp} />
          <div className="stack">
            <RiskSummary risk={risk} />
            <Card id="factors" title="Top risk factors" sub={`What drove the ML model towards ${exp.shap.explained_class}`}>
              {risk.top_risk_factors.length ? (
                <ol style={{ margin: 0, paddingLeft: 20 }}>{risk.top_risk_factors.map((f) => <li key={f} style={{ padding: "3px 0" }}>{f}</li>)}</ol>
              ) : <p className="ink2" style={{ margin: 0 }}>The model predicts LOW risk; no factor pushed it towards a risky class. See the full explanation below.</p>}
            </Card>
            <ComplianceSection risk={risk} />
            <AnomalySection risk={risk} />
            <Explainability exp={exp} id={d.id} onUpdate={setExp} />
            <ActionsSection risk={risk} />
            <ReviewSection id={d.id} risk={risk} onSaved={doc.reload} />
            <Card id="document" title="Extracted information" sub={`${fmt.title(d.extraction_method ?? "")} extraction · ${d.page_count ?? "?"} page(s)${d.ocr_pages ? ` · ${d.ocr_pages} OCR page(s)` : ""}`}>
              <EntitySection d={d} />
            </Card>
          </div>
        </div>
      )}
    </>
  );
}

function AnalyzeNow({ id, onDone }: { id: string; onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="row">
      <span>This document has not been analysed yet.</span>
      <button className="btn primary" disabled={busy} onClick={async () => {
        setBusy(true); setErr(null);
        try { await api.analyze(id); onDone(); } catch (e) { setErr((e as ApiError).message); } finally { setBusy(false); }
      }}>{busy ? <span className="spinner" /> : null} Analyse now</button>
      {err ? <Notice kind="bad">{err}</Notice> : null}
    </div>
  );
}

function FlowRail({ d, risk, exp }: { d: Doc; risk: RiskReport; exp: Explanation }) {
  const t = { ...(d.timings_ms ?? {}) };
  const ms = (k: string) => (t[k] !== undefined ? `${t[k] < 10 ? t[k].toFixed(1) : Math.round(t[k])} ms` : "");
  const steps: { label: string; time?: string; state: string; href?: string }[] = [
    { label: "Upload & validation", time: ms("validation_ms"), state: "done" },
    { label: d.extraction_method === "ocr" || d.extraction_method === "mixed" ? "Text extraction (OCR)" : "Text extraction", time: ms("extraction_ms"), state: d.extraction_method === "ocr" ? "warn" : "done" },
    { label: "Document type detection", time: ms("doc_type_ms"), state: "done" },
    { label: "Section detection", time: ms("section_detection_ms"), state: "done", href: "#document" },
    { label: "Entity extraction", time: ms("entity_extraction_ms"), state: "done", href: "#document" },
    { label: "Feature engineering", time: ms("feature_engineering_ms"), state: "done" },
    { label: "Risk classification", time: ms("ml_inference_ms"), state: risk.ml_predicted_class ? "done" : "warn", href: "#summary" },
    { label: "Anomaly detection", time: ms("anomaly_detection_ms"), state: t.anomaly_detection_ms !== undefined ? "done" : "warn", href: "#anomalies" },
    { label: "Compliance rules", time: ms("compliance_ms"), state: "done", href: "#compliance" },
    { label: "Risk scoring", time: ms("scoring_ms"), state: "done", href: "#summary" },
    { label: "SHAP explanation", time: ms("shap_ms"), state: exp.shap.factors.length ? "done" : "warn", href: "#explain" },
    { label: "LLM explanation", time: exp.llm_status === "ready" && exp.llm_latency_ms ? `${(exp.llm_latency_ms / 1000).toFixed(1)} s` : fmt.title(exp.llm_status), state: exp.llm_status === "ready" ? "done" : exp.llm_status === "pending" ? "" : "warn", href: "#explain" },
    { label: "Human review", time: fmt.title(d.review_status), state: ["confirmed", "corrected"].includes(d.review_status) ? "done" : "", href: "#review" },
  ];
  return (
    <aside className="flow card">
      <h3 style={{ marginBottom: 12 }}>Analysis flow</h3>
      <ol>{steps.map((s) => (
        <li key={s.label} className={s.state}><span className="dot" />
          {s.href ? <a href={s.href}>{s.label}</a> : s.label}<span className="t">{s.time}</span></li>
      ))}</ol>
    </aside>
  );
}

function RiskSummary({ risk }: { risk: RiskReport }) {
  const parts = COMPONENTS.map((c) => ({ label: c.label, color: c.color, value: risk.score_contributions?.[c.key] ?? 0 }));
  return (
    <Card id="summary" title="Risk summary" sub={`Analysed ${fmt.dateTime(risk.analyzed_at)}`}>
      <div className="hero">
        <div>
          <div className="score tabular">{risk.risk_score}<small> / 100</small></div>
          <div style={{ marginTop: 8 }}><RiskBadge level={risk.risk_level} /></div>
        </div>
        <div style={{ flex: 1, minWidth: 260 }}>
          <div className="small ink2" style={{ marginBottom: 6 }}>How the score is composed (points)</div>
          <StackedScore parts={parts} />
          <div className="small muted" style={{ marginTop: 6 }}>
            Components (0–1): {COMPONENTS.map((c) => `${c.label.split(" ")[0]} ${fmt.num(risk.score_components[c.key], 2)}`).join(" · ")}
          </div>
        </div>
      </div>
      {risk.degraded ? <div style={{ marginTop: 14 }}><Notice kind="warn"><div><b>Degraded analysis.</b> A model component was unavailable ({Object.keys(risk.component_errors ?? {}).join(", ")}); its weight was redistributed. Rule-based results are complete.</div></Notice></div> : null}
      {risk.floors_applied?.length ? <div style={{ marginTop: 10 }}><Notice kind="info">Policy guardrail applied: {risk.floors_applied.join("; ")}</Notice></div> : null}
      <div className="grid g2" style={{ marginTop: 16 }}>
        <div>
          <h3 style={{ marginBottom: 8 }}>ML class probabilities</h3>
          {risk.ml_probabilities ? (
            <HBars labelWidth={80} format={(v) => fmt.pct(v, 1)} max={1}
              items={LEVELS.map((l) => ({ label: l, value: risk.ml_probabilities![l], color: LEVEL_COLOR[l],
                note: l === risk.ml_predicted_class ? `${l} (predicted)` : l }))} />
          ) : <p className="muted">Classifier unavailable.</p>}
          {risk.ml_predicted_class ? <div className="small ink2">Predicted class <b>{risk.ml_predicted_class}</b> · P(HIGH or CRITICAL) {fmt.pct(risk.high_risk_probability, 1)}</div> : null}
        </div>
        <div>
          <h3 style={{ marginBottom: 8 }}>Why this needs review</h3>
          {risk.review_reasons?.length ? <ul style={{ margin: 0, paddingLeft: 18 }}>{risk.review_reasons.map((r) => <li key={r}>{r}</li>)}</ul>
            : <p className="ink2" style={{ margin: 0 }}>No review trigger fired.</p>}
        </div>
      </div>
    </Card>
  );
}

function ComplianceSection({ risk }: { risk: RiskReport }) {
  const v = risk.compliance_violations;
  return (
    <Card id="compliance" title="Compliance issues" sub="Deterministic policy rules (configs/compliance_rules.yaml)">
      {v.length === 0 ? <p className="ink2" style={{ margin: 0 }}>No compliance rule fired.</p> : v.map((r) => (
        <div className="issue" key={r.rule_id}>
          <div><RiskBadge level={r.severity} /><div className="rid mono" style={{ marginTop: 4 }}>{r.rule_id}</div>
            <div className="rid">{r.type === "warning" ? "Warning" : "Violation"}</div></div>
          <div><b>{r.rule_name}</b><div>{r.message}</div><div className="fix">→ {r.remediation}</div></div>
        </div>
      ))}
    </Card>
  );
}

function AnomalySection({ risk }: { risk: RiskReport }) {
  const flagged = risk.anomalies.filter((a) => a.is_anomaly);
  const iso = risk.anomalies.find((a) => a.type === "multivariate_isolation_forest");
  return (
    <Card id="anomalies" title="Anomalies" sub="Isolation Forest + vendor-behaviour checks (separate from the risk classifier)">
      {iso ? <div className="small ink2" style={{ marginBottom: 10 }}>Multivariate anomaly percentile: <b className="tabular">{fmt.pct(iso.score)}</b> (flag threshold 97%)</div> : null}
      {risk.anomalies.length === 0 ? <p className="muted" style={{ margin: 0 }}>Anomaly detector unavailable.</p>
        : flagged.length === 0 ? <p className="ink2" style={{ margin: 0 }}>Nothing unusual compared with historical contracts and this vendor's history.</p>
          : flagged.map((a) => (
            <div className="issue" key={a.type}>
              <div><span className="badge HIGH">Anomaly</span><div className="rid" style={{ marginTop: 4 }}>{fmt.title(a.type)}</div></div>
              <div>{a.description}
                {Array.isArray((a.details as any)?.top_contributors) && (a.details as any).top_contributors.length ? (
                  <div className="fix">Most unusual: {(a.details as any).top_contributors.map((c: any) => fmt.title(c.feature)).join(", ")}</div>) : null}
              </div>
            </div>
          ))}
    </Card>
  );
}

function Explainability({ exp, id, onUpdate }: { exp: Explanation; id: string; onUpdate: (e: Explanation) => void }) {
  const { can } = useAuth();
  const cls = exp.shap.explained_class ?? "";
  const llm = exp.llm_explanation;
  const regenerate = async () => {
    await api.regenerate(id);
    onUpdate({ ...exp, llm_status: "pending" });
  };
  return (
    <Card id="explain" title="Explainability"
      sub={`SHAP values from the production model (${exp.shap.output_space} space) for the predicted class ${cls}`}>
      {exp.shap.factors.length ? (
        <DivergingBars explainedClass={cls} items={exp.shap.factors.map((f) => ({ label: f.display, value: f.shap_value }))} />
      ) : <Empty title="No SHAP explanation (classifier unavailable)" />}

      <div style={{ borderTop: "1px solid var(--hair)", marginTop: 16, paddingTop: 14 }}>
        <div className="row" style={{ marginBottom: 8 }}>
          <h2>AI explanation</h2>
          <span className="pill">{exp.llm_model ?? "Ollama"} · local</span>
          {llm ? <span className={`badge ${llm.grounded ? "LOW" : "HIGH"}`}>{llm.grounded ? "Grounding check passed" : "Grounding issues"}</span> : null}
          <span className="spacer" />
          {can("documents:write") && exp.llm_status !== "pending" ? <button className="btn" onClick={regenerate}>Regenerate</button> : null}
        </div>
        {exp.llm_status === "pending" ? <div className="ink2"><span className="spinner" /> Generating a narrative from the structured evidence (runs locally; can take ~30 s)…</div>
          : exp.llm_status === "unavailable" ? <Notice kind="warn"><div><b>LLM unavailable.</b> The local Ollama service did not respond. Everything above (score, rules, anomalies, SHAP) is complete without it.</div></Notice>
            : exp.llm_status === "disabled" ? <Notice kind="info">LLM explanations are disabled in this deployment.</Notice>
              : llm ? (
                <>
                  <p style={{ marginTop: 0 }}>{llm.summary}</p>
                  {llm.factor_explanations.length ? (
                    <dl className="kv">{llm.factor_explanations.map((f) => <Fragment key={f.factor}><dt>{f.factor}</dt><dd>{f.explanation}</dd></Fragment>)}</dl>
                  ) : null}
                  {llm.grounding_warnings.length ? <div style={{ marginTop: 10 }}><Notice kind="warn"><div><b>Grounding warnings</b><ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>{llm.grounding_warnings.map((w) => <li key={w}>{w}</li>)}</ul></div></Notice></div> : null}
                  <p className="small muted" style={{ marginBottom: 0 }}>Generated only from the structured evidence above; the LLM never sees the contract text and cannot change the score.</p>
                </>
              ) : null}
      </div>
    </Card>
  );
}

function ActionsSection({ risk }: { risk: RiskReport }) {
  return (
    <Card id="actions" title="Recommended actions" sub="Deterministic: from the risk level, rule remediations and anomaly checks">
      <ol style={{ margin: 0, paddingLeft: 20 }}>
        {risk.recommended_actions.map((a, i) => (
          <li key={i} style={{ padding: "4px 0" }}>{a.action} <span className="small muted mono">· {a.source}</span></li>
        ))}
      </ol>
    </Card>
  );
}

function ReviewSection({ id, risk, onSaved }: { id: string; risk: RiskReport; onSaved: () => void }) {
  const { can } = useAuth();
  const history = useAsync(() => api.feedbackList().then((all) => all.filter((f) => f.document_id === id)), [id]);
  const [decision, setDecision] = useState<"confirm" | "correct">("confirm");
  const [label, setLabel] = useState<RiskLevel>(risk.risk_level);
  const [comment, setComment] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const submit = async () => {
    setSaving(true); setErr(null);
    try {
      await api.feedback(id, decision === "confirm" ? { decision, comment: comment || undefined }
        : { decision, corrected_label: label, comment });
      setComment("");
      history.reload();
      onSaved();
    } catch (e) { setErr((e as ApiError).message); } finally { setSaving(false); }
  };

  return (
    <Card id="review" title="Human review" sub="Your decision is stored with the model version and is validated by a second person before it can be used for retraining.">
      {history.data?.length ? (
        <div style={{ marginBottom: 14 }}>
          {history.data.map((f: Feedback) => (
            <div key={f.id} className="notice info" style={{ marginBottom: 6 }}>
              <div><b>{fmt.title(f.decision)}</b> → <RiskBadge level={f.reviewer_label} /> · {fmt.dateTime(f.created_at)} · model v{f.model_version}
                · validation <b>{f.validation_status}</b>{f.comment ? <div className="small">“{f.comment}”</div> : null}
                {f.validation_flags?.length ? <div className="small">Flags: {f.validation_flags.map(fmt.title).join(", ")}</div> : null}</div>
            </div>
          ))}
        </div>
      ) : null}
      {can("feedback:write") ? (
        <div className="stack" style={{ gap: 10 }}>
          <div className="row">
            <div className="seg">
              <button className={decision === "confirm" ? "on" : ""} onClick={() => setDecision("confirm")}>Confirm {risk.risk_level}</button>
              <button className={decision === "correct" ? "on" : ""} onClick={() => setDecision("correct")}>Correct</button>
            </div>
            {decision === "correct" ? (
              <select value={label} onChange={(e) => setLabel(e.target.value as RiskLevel)}>
                {LEVELS.map((l) => <option key={l}>{l}</option>)}
              </select>
            ) : null}
          </div>
          <textarea placeholder={decision === "correct" ? "Why is the assessment wrong? (required)" : "Comment (optional)"}
            value={comment} onChange={(e) => setComment(e.target.value)} />
          {err ? <Notice kind="bad">{err}</Notice> : null}
          <div><button className="btn primary" disabled={saving || (decision === "correct" && comment.trim().length < 5)} onClick={submit}>
            {saving ? <span className="spinner" /> : null} Submit review</button></div>
        </div>
      ) : <p className="muted">Your role cannot submit reviews.</p>}
    </Card>
  );
}

function EntitySection({ d }: { d: Doc }) {
  return (
    <div className="grid g-2-1">
      <div className="table-wrap"><table>
        <thead><tr><th>Field</th><th>Value</th><th className="num">Confidence</th><th>Method</th></tr></thead>
        <tbody>{d.entities.map((e) => (
          <tr key={e.name}><td className="ink2">{ENTITY_LABEL[e.name] ?? e.name}</td>
            <td style={{ fontWeight: 500 }}>{typeof e.value === "number" && e.name === "contract_value" ? e.value.toLocaleString("en-IN") : String(e.value ?? "–")}</td>
            <td className="num">{fmt.pct(e.confidence)}</td><td className="small muted mono">{e.method}</td></tr>
        ))}</tbody>
      </table></div>
      <div>
        <h3 style={{ marginBottom: 8 }}>Detected sections</h3>
        <div className="row" style={{ gap: 6 }}>
          {(d.sections ?? []).map((s) => <span key={s.name} className="pill" title={s.heading || "found via clause language"}>{fmt.title(s.name)}{s.method === "clause_language" ? " *" : ""}</span>)}
        </div>
        <p className="small muted">* detected from clause language (no heading).</p>
        <dl className="kv" style={{ marginTop: 12 }}>
          <dt>Document type</dt><dd>{fmt.title(d.document_type ?? "–")} ({fmt.pct(d.doc_type_confidence)})</dd>
          <dt>Format</dt><dd>{d.file_format.toUpperCase()} · {(d.file_size / 1024).toFixed(0)} KB</dd>
          <dt>Extraction</dt><dd>{d.extraction_method ?? "–"} · {d.extraction_status}</dd>
        </dl>
      </div>
    </div>
  );
}
