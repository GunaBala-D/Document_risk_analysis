import { Link, useNavigate } from "react-router-dom";
import { LEVELS, api, fmt } from "../api";
import { HBars } from "../components/charts";
import UploadPanel from "../components/UploadPanel";
import { Card, Empty, ErrorState, Kpi, Loading, RiskBadge, useAsync } from "../components/ui";

const LEVEL_COLOR: Record<string, string> = {
  LOW: "var(--good)", MEDIUM: "var(--warning)", HIGH: "var(--serious)", CRITICAL: "var(--critical)",
};

export default function Dashboard() {
  const ov = useAsync(() => api.overview());
  const recent = useAsync(() => api.documents({ limit: 8 }));
  const navigate = useNavigate();
  const reload = () => { ov.reload(); recent.reload(); };

  if (ov.error) return <ErrorState error={ov.error} onRetry={ov.reload} />;
  if (!ov.data) return <Loading />;
  const o = ov.data;
  const levels = o.risk_levels as Record<string, number>;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Contract risk overview</h1>
          <p>Is this document risky, why, and what should the reviewer do?</p>
        </div>
        <span className="synthetic-note">All documents in this demo are synthetic.</span>
      </div>

      <div className="kpis">
        <Kpi label="Total documents" value={o.total_documents} sub={`${o.analyzed_documents} risk-scored`} />
        <Kpi label="contracts" value={levels.HIGH} icon={<RiskBadge level="HIGH" />} sub="latest analysis" />
        <Kpi label="contracts" value={levels.CRITICAL} icon={<RiskBadge level="CRITICAL" />} sub="latest analysis" />
        <Kpi label="With anomalies" value={o.documents_with_anomalies} sub="flagged by any detector" />
        <Kpi label="Compliance violations" value={o.compliance_violations} sub="across latest analyses" />
        <Kpi label="Pending reviews" value={<Link to="/reviews">{o.pending_reviews}</Link>} sub="awaiting human validation" />
        <Kpi label="Model in production" value={o.current_model_version ? `v${o.current_model_version}` : "–"}
          sub={o.model_status?.classifier === "loaded" ? "classifier loaded" : "classifier unavailable"} />
      </div>

      <div className="grid g-2-1">
        <Card title="Analyse a contract" sub="The file goes through validation, extraction, ML, anomaly detection, compliance rules and scoring.">
          <UploadPanel onDone={reload} />
        </Card>
        <Card title="Risk distribution" sub="Latest analysis per document">
          {o.analyzed_documents ? (
            <HBars labelWidth={90} format={(v) => String(v)}
              items={LEVELS.map((l) => ({ label: l, value: levels[l] ?? 0, color: LEVEL_COLOR[l],
                note: `${l}: ${fmt.pct((levels[l] ?? 0) / o.analyzed_documents)} of scored documents` }))} />
          ) : <Empty title="No scored documents yet" />}
        </Card>
      </div>

      <div className="grid g-2-1" style={{ marginTop: 16 }}>
        <Card title="Recent documents" action={<Link to="/documents" className="small">All documents →</Link>}>
          {recent.data?.items.length ? (
            <div className="table-wrap"><table>
              <thead><tr><th>Document</th><th>Vendor</th><th>Uploaded</th><th>Risk</th><th>Review</th></tr></thead>
              <tbody>
                {recent.data.items.map((d) => (
                  <tr key={d.id} className="clickable" onClick={() => navigate(`/documents/${d.id}`)}>
                    <td><div style={{ fontWeight: 500 }}>{d.filename}</div><div className="small muted mono">{d.id}</div></td>
                    <td>{d.vendor_name ?? <span className="muted">–</span>}</td>
                    <td className="small">{fmt.dateTime(d.uploaded_at)}</td>
                    <td><RiskBadge level={d.risk_level} score={d.risk_score} /></td>
                    <td><span className="pill">{fmt.title(d.review_status)}</span></td>
                  </tr>
                ))}
              </tbody>
            </table></div>
          ) : recent.loading ? <Loading /> : <Empty title="No documents yet">Upload a contract to get started.</Empty>}
        </Card>
        <Card title="Most frequent compliance issues" sub="Rule hits in latest analyses">
          {o.top_compliance_rules.length ? (
            <HBars labelWidth={230} items={o.top_compliance_rules.map((r: any) => ({ label: r.rule_name, value: r.count, note: r.rule_id }))} />
          ) : <Empty title="No rule hits yet" />}
        </Card>
      </div>
    </>
  );
}
