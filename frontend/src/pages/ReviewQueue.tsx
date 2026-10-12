import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../App";
import { ApiError, api, fmt, type Feedback } from "../api";
import { Card, Empty, ErrorState, Loading, Notice, RiskBadge, useAsync } from "../components/ui";

export default function ReviewQueue() {
  const { can, user } = useAuth();
  const [tab, setTab] = useState<"queue" | "validate">("queue");
  return (
    <>
      <div className="page-head">
        <div><h1>Human review</h1>
          <p>Analysts confirm or correct assessments; a risk manager validates feedback before it can be used for retraining.</p></div>
        <div className="seg">
          <button className={tab === "queue" ? "on" : ""} onClick={() => setTab("queue")}>Review queue</button>
          <button className={tab === "validate" ? "on" : ""} onClick={() => setTab("validate")}>Feedback validation</button>
        </div>
      </div>
      {tab === "queue" ? <Queue /> : <Validation canValidate={can("feedback:validate")} myId={user.id} />}
    </>
  );
}

function Queue() {
  const q = useAsync(() => api.queue());
  const navigate = useNavigate();
  if (q.error) return <ErrorState error={q.error} onRetry={q.reload} />;
  if (!q.data) return <Loading />;
  return (
    <Card title={`${q.data.total} documents awaiting review`} sub="Highest risk first. Open a document to review it.">
      {q.data.items.length === 0 ? <Empty title="Queue is empty">Nothing needs human validation right now.</Empty> : (
        <div className="table-wrap"><table>
          <thead><tr><th>Document</th><th>Vendor</th><th>Risk</th><th>ML class</th><th className="num">Violations</th><th className="num">Anomalies</th><th>Why</th></tr></thead>
          <tbody>{q.data.items.map((i) => (
            <tr key={i.document_id} className="clickable" onClick={() => navigate(`/documents/${i.document_id}#review`)}>
              <td><div style={{ fontWeight: 500 }}>{i.filename}</div><div className="small muted">{fmt.dateTime(i.uploaded_at)}</div></td>
              <td>{i.vendor_name ?? "–"}</td>
              <td><RiskBadge level={i.risk_level} score={i.risk_score} /></td>
              <td className="small">{i.ml_class ?? "–"}</td>
              <td className="num">{i.n_violations}</td>
              <td className="num">{i.n_anomalies}</td>
              <td className="small ink2">{(i.review_reasons ?? []).join(" · ")}</td>
            </tr>
          ))}</tbody>
        </table></div>
      )}
    </Card>
  );
}

function Validation({ canValidate, myId }: { canValidate: boolean; myId: number }) {
  const [status, setStatus] = useState("pending");
  const list = useAsync(() => api.feedbackList(status), [status]);
  return (
    <Card title="Reviewer feedback" sub="Only validated feedback enters the retraining dataset. You cannot validate your own reviews."
      action={<select value={status} onChange={(e) => setStatus(e.target.value)}>
        <option value="pending">Pending</option><option value="validated">Validated</option><option value="rejected">Rejected</option></select>}>
      {!canValidate && status === "pending" ? <div style={{ marginBottom: 12 }}><Notice kind="info">Your role can view feedback but only a Risk Manager or Administrator can validate it.</Notice></div> : null}
      {list.error ? <ErrorState error={list.error} onRetry={list.reload} /> : !list.data ? <Loading /> :
        list.data.length === 0 ? <Empty title={`No ${status} feedback`} /> : (
          <div className="table-wrap"><table>
            <thead><tr><th>Document</th><th>System</th><th>Reviewer label</th><th>Comment</th><th>Flags</th><th>{status === "pending" ? "Decision" : "Validated"}</th></tr></thead>
            <tbody>{list.data.map((f) => <FeedbackRow key={f.id} f={f} canValidate={canValidate && f.reviewer_id !== myId} onDone={list.reload} />)}</tbody>
          </table></div>
        )}
    </Card>
  );
}

function FeedbackRow({ f, canValidate, onDone }: { f: Feedback; canValidate: boolean; onDone: () => void }) {
  const [comment, setComment] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const act = async (approve: boolean) => {
    setErr(null);
    try { await api.validate(f.id, approve, comment || undefined); onDone(); } catch (e) { setErr((e as ApiError).message); }
  };
  return (
    <tr>
      <td><Link to={`/documents/${f.document_id}`} className="mono small">{f.document_id}</Link><div className="small muted">{fmt.dateTime(f.created_at)}</div></td>
      <td><RiskBadge level={f.model_risk_level} score={f.model_risk_score} /><div className="small muted">ML {f.model_prediction ?? "–"} · v{f.model_version}</div></td>
      <td><span className="small">{fmt.title(f.decision)} →</span> <RiskBadge level={f.reviewer_label} /></td>
      <td className="small" style={{ maxWidth: 260 }}>{f.comment ?? <span className="muted">–</span>}</td>
      <td className="small">{f.validation_flags?.length ? f.validation_flags.map((x) => <div key={x} style={{ color: "var(--warning-ink)" }}>⚑ {fmt.title(x)}</div>) : <span className="muted">none</span>}</td>
      <td style={{ minWidth: 220 }}>
        {f.validation_status !== "pending" ? <span className="small">{fmt.title(f.validation_status)} {fmt.dateTime(f.validated_at)}{f.validation_comment ? ` · “${f.validation_comment}”` : ""}</span>
          : canValidate ? (
            <div className="stack" style={{ gap: 6 }}>
              <input placeholder="Comment (required to reject)" value={comment} onChange={(e) => setComment(e.target.value)} />
              <div className="row"><button className="btn primary" onClick={() => act(true)}>Validate</button>
                <button className="btn danger" onClick={() => act(false)} disabled={!comment.trim()}>Reject</button></div>
              {err ? <span className="small" style={{ color: "var(--critical-ink)" }}>{err}</span> : null}
            </div>
          ) : <span className="small muted">Awaiting a second reviewer</span>}
      </td>
    </tr>
  );
}
