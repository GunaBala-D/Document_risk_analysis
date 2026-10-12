import { LEVELS, api, fmt } from "../api";
import { ConfusionMatrix, HBars } from "../components/charts";
import { Card, ErrorState, Kpi, Loading, Notice, useAsync } from "../components/ui";

const FAMILY: Record<string, string> = {
  logistic_regression: "Logistic Regression", random_forest: "Random Forest", xgboost: "XGBoost", lightgbm: "LightGBM",
};

export default function ModelAnalytics() {
  const cur = useAsync(() => api.currentModel());
  const versions = useAsync(() => api.modelVersions());
  if (cur.error) return <ErrorState error={cur.error} onRetry={cur.reload} />;
  if (!cur.data) return <Loading />;
  const c = cur.data.classifier;
  const t = c.test_metrics ?? {};
  const ta = c.test_metrics_argmax;
  const an = cur.data.anomaly_detector;
  const perClass = t.per_class ?? {};
  const shap = (c.global_shap ?? []).slice(0, 12);

  return (
    <>
      <div className="page-head">
        <div><h1>Model analytics</h1>
          <p>Risk classifier <b>v{c.version}</b> · {FAMILY[c.family] ?? c.family} · feature set {c.feature_version} · trained {fmt.date(c.trained_at)}</p></div>
        <span className="synthetic-note">Metrics measured on a frozen hold-out set of {c.n_test} synthetic contracts.</span>
      </div>

      <div className="kpis">
        <Kpi label="Precision (macro)" value={fmt.num(t.precision_macro)} />
        <Kpi label="Recall (macro)" value={fmt.num(t.recall_macro)} />
        <Kpi label="F1 (macro)" value={fmt.num(t.f1_macro)} sub="selection metric" />
        <Kpi label="ROC-AUC (OvR)" value={fmt.num(t.roc_auc_ovr_macro)} />
        <Kpi label="PR-AUC (macro)" value={fmt.num(t.pr_auc_macro)} />
        <Kpi label="HIGH+CRITICAL recall" value={fmt.num(t.high_risk_recall)} sub="gate minimum 0.90" />
        <Kpi label="CRITICAL recall" value={fmt.num(t.critical_recall)} sub="gate minimum 0.70" />
        <Kpi label="Escalation recall" value={fmt.num(t.escalation_recall)} sub={`threshold P(H+C) ≥ ${c.escalation_threshold}`} />
      </div>

      <div className="grid g2">
        <Card title="Confusion matrix (test set)" sub={`Decision weights ${JSON.stringify(c.decision_weights)} applied to class probabilities`}>
          <ConfusionMatrix matrix={t.confusion_matrix ?? []} labels={LEVELS} />
        </Card>
        <Card title="Per-class performance" sub="Test set">
          <div className="table-wrap"><table>
            <thead><tr><th>Class</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr></thead>
            <tbody>{LEVELS.map((l) => (
              <tr key={l}><td>{l}</td><td className="num">{fmt.num(perClass[l]?.precision)}</td><td className="num">{fmt.num(perClass[l]?.recall)}</td>
                <td className="num">{fmt.num(perClass[l]?.f1)}</td><td className="num">{perClass[l]?.support}</td></tr>
            ))}</tbody>
          </table></div>
          {ta ? (
            <div style={{ marginTop: 14 }}>
              <h3 style={{ marginBottom: 6 }}>Effect of threshold tuning</h3>
              <table><thead><tr><th>Decision rule</th><th className="num">F1</th><th className="num">H+C recall</th><th className="num">CRITICAL recall</th></tr></thead>
                <tbody>
                  <tr><td>Plain argmax</td><td className="num">{fmt.num(ta.f1_macro)}</td><td className="num">{fmt.num(ta.high_risk_recall)}</td><td className="num">{fmt.num(ta.critical_recall)}</td></tr>
                  <tr><td>Tuned weights (OOF)</td><td className="num">{fmt.num(t.f1_macro)}</td><td className="num">{fmt.num(t.high_risk_recall)}</td><td className="num">{fmt.num(t.critical_recall)}</td></tr>
                </tbody></table>
            </div>
          ) : null}
        </Card>
      </div>

      <div className="grid g-2-1" style={{ marginTop: 16 }}>
        <Card title="Global feature importance" sub="Mean |SHAP value| on training data, averaged over classes">
          <HBars labelWidth={240} format={(v) => v.toFixed(3)} items={shap.map((r: any) => ({ label: r.label, value: r.mean_abs_shap, note: r.feature }))} />
        </Card>
        <Card title="Why this model" sub="Recorded at training time">
          <p style={{ marginTop: 0 }}>{c.selection_reason}</p>
          <dl className="kv">
            <dt>Imbalance strategy</dt><dd>{c.imbalance}</dd>
            <dt>Hyper-parameters</dt><dd className="mono">{Object.entries(c.params ?? {}).map(([k, v]) => `${k}=${v}`).join(", ")}</dd>
            <dt>CV macro-F1</dt><dd>{fmt.num(c.cv_metrics?.f1_macro)}</dd>
            <dt>Temporal check F1</dt><dd>{fmt.num(c.temporal_check?.f1_macro)} <span className="muted small">(train on older, test on newer)</span></dd>
            <dt>Dataset</dt><dd className="mono">{c.dataset_version}</dd>
            <dt>Label policy</dt><dd className="mono">{c.label_policy_version}</dd>
          </dl>
          <h3 style={{ margin: "16px 0 6px" }}>Anomaly detector v{an.version}</h3>
          <dl className="kv">
            <dt>ROC-AUC vs injected anomalies</dt><dd>{fmt.num(an.test_metrics?.if_roc_auc)}</dd>
            <dt>Flag precision / recall</dt><dd>{fmt.num(an.test_metrics?.if_flag_precision, 2)} / {fmt.num(an.test_metrics?.if_flag_recall, 2)}</dd>
          </dl>
        </Card>
      </div>

      <Card title="Model registry history" sub="MLflow registered versions. A version reaches @production only if the quality gate passes." className="" >
        {versions.error ? <Notice kind="warn">Registry unavailable: {versions.error.message}</Notice> : !versions.data ? <Loading /> : (
          <div className="table-wrap"><table>
            <thead><tr><th>Model</th><th>Version</th><th>Aliases</th><th>Quality gate</th><th className="num">Test F1</th><th className="num">PR-AUC</th><th className="num">H+C recall</th><th className="num">CRITICAL recall</th><th>Created</th></tr></thead>
            <tbody>{Object.entries(versions.data).flatMap(([name, rows]: [string, any]) => rows.map((r: any) => (
              <tr key={`${name}-${r.version}`}>
                <td className="small">{name}</td><td>v{r.version}</td>
                <td>{r.aliases.length ? r.aliases.map((a: string) => <span key={a} className="pill" style={{ marginRight: 4 }}>@{a}</span>) : <span className="muted">–</span>}</td>
                <td>{r.quality_gate ? <span className={`badge ${r.quality_gate === "PASS" ? "LOW" : "CRITICAL"}`}>{r.quality_gate}</span> : "–"}</td>
                <td className="num">{fmt.num(r.metrics.test_f1_macro)}</td><td className="num">{fmt.num(r.metrics.test_pr_auc_macro)}</td>
                <td className="num">{fmt.num(r.metrics.test_high_risk_recall)}</td><td className="num">{fmt.num(r.metrics.test_critical_recall)}</td>
                <td className="small">{new Date(r.created_at).toLocaleString()}</td>
              </tr>
            )))}</tbody>
          </table></div>
        )}
      </Card>
    </>
  );
}
