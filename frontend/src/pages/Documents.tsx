import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { LEVELS, api, fmt } from "../api";
import UploadPanel from "../components/UploadPanel";
import { Card, Empty, ErrorState, Loading, RiskBadge, useAsync } from "../components/ui";

const PAGE = 25;

export default function Documents() {
  const [params, setParams] = useSearchParams();
  const [showUpload, setShowUpload] = useState(false);
  const navigate = useNavigate();
  const f = Object.fromEntries(params.entries());
  const offset = Number(f.offset ?? 0);
  const docs = useAsync(() => api.documents({ ...f, limit: PAGE, offset }), [params.toString()]);

  const set = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    value ? next.set(key, value) : next.delete(key);
    next.delete("offset");
    setParams(next);
  };
  const page = (delta: number) => {
    const next = new URLSearchParams(params);
    next.set("offset", String(Math.max(0, offset + delta)));
    setParams(next);
  };

  return (
    <>
      <div className="page-head">
        <div><h1>Documents</h1><p>Every uploaded file, its processing status and latest risk assessment.</p></div>
        <button className="btn primary" onClick={() => setShowUpload((s) => !s)}>{showUpload ? "Close" : "Upload contract"}</button>
      </div>
      {showUpload ? <Card><UploadPanel onDone={docs.reload} /></Card> : null}

      <div className="filters" style={{ marginTop: showUpload ? 16 : 0 }}>
        <label>Search<input placeholder="file, ID or vendor" defaultValue={f.q ?? ""}
          onKeyDown={(e) => e.key === "Enter" && set("q", (e.target as HTMLInputElement).value)} /></label>
        <label>Risk<select value={f.risk_level ?? ""} onChange={(e) => set("risk_level", e.target.value)}>
          <option value="">All levels</option>{LEVELS.map((l) => <option key={l}>{l}</option>)}</select></label>
        <label>Type<select value={f.document_type ?? ""} onChange={(e) => set("document_type", e.target.value)}>
          <option value="">All types</option><option value="vendor_contract">Vendor contract</option>
          <option value="invoice">Invoice</option><option value="purchase_order">Purchase order</option>
          <option value="unknown">Unknown</option></select></label>
        <label>Vendor<input placeholder="contains…" defaultValue={f.vendor ?? ""}
          onKeyDown={(e) => e.key === "Enter" && set("vendor", (e.target as HTMLInputElement).value)}
          onBlur={(e) => e.target.value !== (f.vendor ?? "") && set("vendor", e.target.value)} /></label>
        <label>Review<select value={f.review_status ?? ""} onChange={(e) => set("review_status", e.target.value)}>
          <option value="">Any</option><option value="pending">Pending</option><option value="confirmed">Confirmed</option>
          <option value="corrected">Corrected</option><option value="not_required">Not required</option></select></label>
        <label>From<input type="date" value={f.date_from ?? ""} onChange={(e) => set("date_from", e.target.value)} /></label>
        <label>To<input type="date" value={f.date_to ?? ""} onChange={(e) => set("date_to", e.target.value)} /></label>
        {params.toString() ? <button className="btn ghost" style={{ alignSelf: "flex-end" }} onClick={() => setParams({})}>Clear</button> : null}
      </div>

      <Card>
        {docs.error ? <ErrorState error={docs.error} onRetry={docs.reload} /> : !docs.data ? <Loading /> :
          docs.data.items.length === 0 ? <Empty title="No documents match these filters" /> : (
            <div className={docs.loading ? "refetching" : ""}>
              <div className="table-wrap"><table>
                <thead><tr><th>Document</th><th>Vendor</th><th>Type</th><th>Uploaded</th><th>Extraction</th><th>Status</th><th>Risk</th><th>Review</th></tr></thead>
                <tbody>
                  {docs.data.items.map((d) => (
                    <tr key={d.id} className="clickable" onClick={() => navigate(`/documents/${d.id}`)}>
                      <td><div style={{ fontWeight: 500 }}>{d.filename}</div><div className="small muted mono">{d.id}</div></td>
                      <td>{d.vendor_name ?? <span className="muted">–</span>}</td>
                      <td className="small">{d.document_type ? fmt.title(d.document_type) : "–"}</td>
                      <td className="small">{fmt.dateTime(d.uploaded_at)}</td>
                      <td className="small">{d.extraction_method ? <span className="pill">{d.extraction_method.toUpperCase()}</span> : "–"}</td>
                      <td className="small">{fmt.title(d.processing_status)}</td>
                      <td><RiskBadge level={d.risk_level} score={d.risk_score} /></td>
                      <td><span className="pill">{fmt.title(d.review_status)}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table></div>
              <div className="row" style={{ marginTop: 12 }}>
                <span className="small muted tabular">{offset + 1}–{Math.min(offset + PAGE, docs.data.total)} of {docs.data.total}</span>
                <span className="spacer" />
                <button className="btn" disabled={offset === 0} onClick={() => page(-PAGE)}>Previous</button>
                <button className="btn" disabled={offset + PAGE >= docs.data.total} onClick={() => page(PAGE)}>Next</button>
              </div>
            </div>
          )}
      </Card>
    </>
  );
}
