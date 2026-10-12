import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, api } from "../api";

type Stage = "idle" | "upload" | "analyze" | "done" | "failed";
const STEPS: { key: Stage; label: string }[] = [
  { key: "upload", label: "Upload & file validation" },
  { key: "analyze", label: "Extraction · OCR · ML · rules · scoring" },
  { key: "done", label: "Risk report ready" },
];
const ORDER: Stage[] = ["upload", "analyze", "done"];

export default function UploadPanel({ onDone }: { onDone?: () => void }) {
  const [stage, setStage] = useState<Stage>("idle");
  const [failedAt, setFailedAt] = useState<Stage | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const run = async (file: File) => {
    setMessage(null);
    setFailedAt(null);
    let docId: string | null = null;
    try {
      setStage("upload");
      docId = (await api.upload(file)).document_id;
      setStage("analyze");
      const res = await api.analyze(docId);
      setStage("done");
      onDone?.();
      if (res.processing_status === "unsupported_type") {
        setMessage(res.message);
        return;
      }
      navigate(`/documents/${docId}`);
    } catch (e) {
      const err = e as ApiError;
      setFailedAt(docId ? "analyze" : "upload");
      setStage("failed");
      if (err.code === "duplicate_document" && (err.details as any)?.existing_document_id) {
        setMessage(`This file was already uploaded as ${(err.details as any).existing_document_id}.`);
      } else {
        setMessage(err.message);
      }
    }
  };

  const busy = stage === "upload" || stage === "analyze";
  return (
    <div>
      <div className={`dropzone ${over ? "over" : ""}`} role="button" tabIndex={0}
        onClick={() => !busy && input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !busy && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && !busy) run(f); }}>
        <div style={{ fontWeight: 600, color: "var(--ink)" }}>{busy ? "Analysing contract…" : "Drop a vendor contract here, or click to choose"}</div>
        <div className="small muted">PDF, DOCX, PNG or JPG · up to 10 MB · scanned files are OCR'd</div>
        <input ref={input} type="file" hidden accept=".pdf,.docx,.png,.jpg,.jpeg"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f); e.target.value = ""; }} />
        {stage !== "idle" ? (
          <div className="steps">
            {STEPS.map((s) => {
              const cur = ORDER.indexOf(stage === "failed" ? failedAt! : stage);
              const idx = ORDER.indexOf(s.key);
              const cls = stage === "failed" && s.key === failedAt ? "failed" : idx < cur || stage === "done" ? "done"
                : idx === cur ? "active" : "";
              return <span key={s.key} className={`step ${cls}`}>{cls === "active" ? <span className="spinner" style={{ width: 10, height: 10, borderWidth: 1.5, borderTopColor: "var(--surface)" }} /> : null}{s.label}</span>;
            })}
          </div>
        ) : null}
      </div>
      {message ? <div className={`notice ${stage === "failed" ? "bad" : "info"}`} style={{ marginTop: 10 }}>{message}</div> : null}
    </div>
  );
}
