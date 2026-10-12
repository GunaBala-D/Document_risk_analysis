import { useState } from "react";
import { useAuth } from "../App";
import { ApiError, LEVELS, api, fmt } from "../api";
import { GroupedBars, HBars, LineChart } from "../components/charts";
import { Card, ErrorState, Kpi, Loading, Notice, useAsync } from "../components/ui";

const SEV: Record<string, string> = { drift: "CRITICAL", warn: "MEDIUM", ok: "LOW" };

export default function Monitoring() {
  const { can } = useAuth();
  const m = useAsync(() => api.monitoring());
  const hist = useAsync(() => api.monitoringHistory());
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const snapshot = async () => {
    setRunning(true); setErr(null);
    try { m.setData(await api.monitoringRun()); hist.reload(); } catch (e) { setErr((e as ApiError).message); } finally { setRunning(false); }
  };

  if (m.error) return <ErrorState error={m.error} onRetry={m.reload} />;
  if (!m.data) return <Loading label="Computing monitoring report" />;
  const s = m.data;
  const dd = s.data_drift, pd = s.prediction_drift, perf = s.performance, sh = s.service_health, rt = s.retraining;

  return (
    <div className={m.loading ? "refetching" : ""}>
      <div className="page-head">
        <div><h1>Monitoring</h1>
          <p>Model v{s.model_version} · last {s.window_days} days · computed live from operational data</p></div>
        <div className="row">
          <button className="btn" onClick={m.reload}>Refresh</button>
          {can("monitoring:run") ? <button className="btn primary" disabled={running} onClick={snapshot}>{running ? <span className="spinner" /> : null} Save snapshot</button> : null}
        </div>
      </div>
      {err ? <Notice kind="bad">{err}</Notice> : null}

      <Card title="Retraining decision" sub="Retraining needs NEW validated labels plus drift or measured degradation — drift alone never triggers it.">
        <div className="row" style={{ alignItems: "flex-start" }}>
          <span className={`badge ${rt.retrain ? "HIGH" : "LOW"}`} style={{ fontSize: 13 }}>{rt.retrain ? "Retraining recommended" : "No retraining needed"}</span>
          <ul style={{ margin: 0, paddingLeft: 18 }}>{rt.reasons.map((r: string) => <li key={r}>{r}</li>)}</ul>
        </div>
        {rt.last_run ? (
          <div style={{ marginTop: 12 }}>
            <Notice kind={rt.last_run.promoted ? "good" : "warn"}>
              <div><b>Last retraining run ({fmt.dateTime(rt.last_run.at)}):</b> candidate v{rt.last_run.candidate_version}{" "}
                {rt.last_run.promoted ? "passed the quality gate and was promoted." : `was rejected by the quality gate (${rt.last_run.gate_failures.join(", ")}).`}
                {" "}Test macro-F1 {fmt.num(rt.last_run.candidate_f1_macro)} vs production {fmt.num(rt.last_run.production_f1_macro)}.
                {rt.last_run.simulation ? " (Deliberately corrupted candidate - gate demonstration.)" : ""}
                {!rt.last_run.promoted ? " Rejected runs leave their labels unused, so the recommendation stays until a candidate succeeds or the cause is fixed (e.g. a policy change belongs in the compliance rules)." : ""}</div>
            </Notice>
          </div>
        ) : null}
        {rt.retrain ? <p className="small muted" style={{ marginBottom: 0 }}>Run <code>python -m training.retrain</code>: the candidate is evaluated against production and must pass the quality gate.</p> : null}
      </Card>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <Card title="1 · Data drift" sub="Has the input distribution P(X) changed vs training? PSI + KS / χ² per feature.">
          {dd.status !== "ok" ? <Notice kind="info">{dd.status === "insufficient_data" ? `Not enough recent documents (${dd.n_current} of ${dd.min_samples} needed) to judge drift reliably.` : fmt.title(dd.status)}</Notice> : (
            <>
              <div className="row" style={{ marginBottom: 10 }}>
                <span className={`badge ${dd.dataset_drift ? "CRITICAL" : "LOW"}`}>{dd.dataset_drift ? "Dataset drift" : "No dataset-level drift"}</span>
                <span className="small ink2">{fmt.pct(dd.drift_share)} of features drifted (threshold 30%) · {dd.n_current} recent vs {dd.n_reference} reference rows</span>
              </div>
              <div className="table-wrap" style={{ maxHeight: 360, overflowY: "auto" }}><table>
                <thead><tr><th>Feature</th><th className="num">PSI</th><th className="num">p-value</th><th className="num">Ref. mean</th><th className="num">Now</th><th>Status</th></tr></thead>
                <tbody>{dd.features.map((f: any) => (
                  <tr key={f.feature}><td className="small">{f.feature}</td><td className="num">{fmt.num(f.psi, 2)}</td>
                    <td className="num">{f.p_value === null ? "–" : f.p_value < 0.0001 ? "<0.0001" : f.p_value.toFixed(4)}</td>
                    <td className="num">{fmt.num(f.reference_mean, 2)}</td><td className="num">{fmt.num(f.current_mean, 2)}</td>
                    <td><span className={`badge ${SEV[f.severity]}`}>{f.severity}</span></td></tr>
                ))}</tbody>
              </table></div>
              <p className="small muted" style={{ marginBottom: 0 }}>Drift = PSI ≥ 0.2 <i>and</i> p &lt; 0.01; requiring both avoids false alarms on small samples.</p>
            </>
          )}
        </Card>

        <div className="stack">
          <Card title="2 · Prediction drift" sub="Has the distribution of model outputs P(ŷ) changed?">
            {pd.status !== "ok" ? <Notice kind="info">Not enough recent predictions.</Notice> : (
              <>
                <div className="row" style={{ marginBottom: 6 }}>
                  <span className={`badge ${pd.drifted ? "CRITICAL" : "LOW"}`}>{pd.drifted ? "Drifted" : "Stable"}</span>
                  <span className="small ink2">PSI {fmt.num(pd.psi, 3)} over {pd.n_current} predictions</span>
                </div>
                <GroupedBars categories={LEVELS} format={(v) => fmt.pct(v)}
                  series={[{ name: "Training reference", color: "var(--s1)", values: LEVELS.map((l) => pd.reference_distribution[l]) },
                    { name: "Last 30 days", color: "var(--s2)", values: LEVELS.map((l) => pd.current_distribution[l]) }]} />
              </>
            )}
          </Card>
          <Card title="3 · Model performance (concept drift)" sub="Only measurable with labels: model vs validated reviewer decisions.">
            {perf.status !== "ok" ? <Notice kind="info">{perf.n_labels} validated labels so far — {perf.min_labels} needed before performance is reported.</Notice> : (
              <>
                <div className="kpis" style={{ marginBottom: 8 }}>
                  <Kpi label="Macro-F1 on feedback" value={fmt.num(perf.f1_macro)} sub={`test baseline ${fmt.num(perf.baseline_test_f1_macro)}`} />
                  <Kpi label="Agreement" value={fmt.pct(perf.agreement)} sub={`${perf.n_labels} validated labels`} />
                  <Kpi label="H+C recall" value={fmt.num(perf.high_risk_recall)} />
                </div>
                <Notice kind={perf.degraded ? "bad" : "good"}>{!perf.degraded
                  ? "Performance on validated feedback is within tolerance."
                  : dd.status === "ok" && dd.drift_share > 0
                    ? `Performance has degraded beyond tolerance (5 F1 points). ${fmt.pct(dd.drift_share)} of features also drifted, so the drop may come from new kinds of inputs (data drift), a changed definition of risk (concept drift), or both — compare reviewer corrections on familiar vs. unfamiliar contracts.`
                    : "Performance has degraded beyond tolerance (5 F1 points) without matching data drift — this points to concept drift: the correct answer changed for similar inputs."}</Notice>
              </>
            )}
          </Card>
        </div>
      </div>

      <Card title="4 · Service health" sub={`API traffic in the last ${sh.window_hours ?? 24} hours (all endpoints)`}>
        {sh.status === "no_traffic" ? <Notice kind="info">No API traffic recorded in this window.</Notice> : (
          <>
            <div className="kpis">
              <Kpi label="Requests" value={sh.requests} />
              <Kpi label="Error rate (5xx)" value={fmt.pct(sh.error_rate, 2)} sub={`SLO ≤ ${fmt.pct(sh.slo.error_rate)}`} />
              <Kpi label="Latency p50" value={`${Math.round(sh.p50_ms)} ms`} />
              <Kpi label="Latency p95" value={`${Math.round(sh.p95_ms)} ms`} sub={`SLO ≤ ${sh.slo.latency_p95_ms} ms`} />
              <Kpi label="Status" value={<span className={`badge ${sh.status === "ok" ? "LOW" : "HIGH"}`}>{sh.status === "ok" ? "Healthy" : "Degraded"}</span>} />
            </div>
            <div className="grid g2">
              <div><h3 style={{ marginBottom: 6 }}>Requests per hour</h3>
                <LineChart label="Requests per hour" points={sh.hourly.map((h: any) => ({ x: fmt.dateTime(h.hour), y: h.requests }))} /></div>
              <div><h3 style={{ marginBottom: 6 }}>p95 latency per hour (ms)</h3>
                <LineChart label="p95 latency" color="var(--s2)" format={(v) => `${Math.round(v)}`} points={sh.hourly.map((h: any) => ({ x: fmt.dateTime(h.hour), y: h.p95_ms }))} /></div>
            </div>
            <h3 style={{ margin: "14px 0 6px" }}>Slowest endpoints (p95)</h3>
            <HBars labelWidth={300} format={(v) => `${Math.round(v)} ms`}
              items={[...sh.by_route].sort((a: any, b: any) => b.p95_ms - a.p95_ms).slice(0, 8)
                .map((r: any) => ({ label: r.route, value: r.p95_ms, note: `${r.requests} requests · ${fmt.pct(r.error_rate, 1)} errors` }))} />
          </>
        )}
      </Card>

      <Card title="Snapshot history" sub="Persisted monitoring reports (Save snapshot, or a daily scheduler calling POST /monitoring/run)">
        {!hist.data ? <Loading /> : hist.data.length === 0 ? <p className="muted">No snapshots saved yet.</p> : (
          <div className="table-wrap"><table>
            <thead><tr><th>When</th><th>Type</th><th>Model</th><th className="num">Drift share</th><th>Dataset drift</th><th className="num">Prediction PSI</th><th>Retrain?</th></tr></thead>
            <tbody>{hist.data.map((h) => (
              <tr key={h.id}><td className="small">{fmt.dateTime(h.created_at)}</td><td className="small">{fmt.title(h.type)}</td><td>v{h.model_version}</td>
                <td className="num">{h.drift_share === null || h.drift_share === undefined ? "–" : fmt.pct(h.drift_share)}</td>
                <td>{h.dataset_drift === null || h.dataset_drift === undefined ? "–" : h.dataset_drift ? "yes" : "no"}</td>
                <td className="num">{fmt.num(h.prediction_psi, 3)}</td><td>{h.retrain === undefined || h.retrain === null ? "–" : h.retrain ? "yes" : "no"}</td></tr>
            ))}</tbody>
          </table></div>
        )}
      </Card>
    </div>
  );
}
