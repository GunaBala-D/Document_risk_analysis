// Hand-rolled SVG charts following the data-viz rules: thin marks, 4px rounded data-ends,
// 2px gaps, recessive hairline grid, one axis, direct labels + hover tooltips, text in ink tokens.
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

/** Measure the container so 1 SVG unit = 1 CSS px (text never scales down in narrow columns). */
function useWidth(min = 260): [React.RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(640);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(min, Math.round(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [min]);
  return [ref, w];
}

interface Tip { x: number; y: number; value: ReactNode; label: ReactNode }

function useTip() {
  const [tip, setTip] = useState<Tip | null>(null);
  const show = (e: React.PointerEvent | React.FocusEvent, value: ReactNode, label: ReactNode) => {
    const pe = e as React.PointerEvent;
    const rect = (e.target as Element).getBoundingClientRect();
    const x = pe.clientX ?? rect.left + rect.width / 2;
    const y = pe.clientY ?? rect.top;
    setTip({ x: x + 12, y: y + 12, value, label });
  };
  const node = tip ? (
    <div className="tooltip" style={{ left: Math.min(tip.x, window.innerWidth - 240), top: tip.y }}>
      <div className="tv">{tip.value}</div>
      <div className="tl">{tip.label}</div>
    </div>
  ) : null;
  return { show, hide: () => setTip(null), node };
}

/** Horizontal bars for one series (or per-item status colours with labels). */
export function HBars({ items, format = (v) => String(v), max, color = "var(--s1)", labelWidth = 190 }: {
  items: { label: string; value: number; color?: string; note?: string }[];
  format?: (v: number) => string; max?: number; color?: string; labelWidth?: number;
}) {
  const t = useTip();
  const [ref, w] = useWidth();
  const valueW = 64;
  // Narrow container + long labels: put each label above its bar instead of truncating it.
  const stacked = labelWidth > 120 && w < labelWidth * 2.2;
  const rowH = stacked ? 36 : 26;
  const lw = stacked ? 0 : Math.min(labelWidth, Math.round(w * 0.45));
  const plotW = Math.max(40, w - lw - valueW);
  const barY = stacked ? 18 : 6, barH = stacked ? 12 : rowH - 12;
  const m = max ?? Math.max(1e-9, ...items.map((i) => i.value));
  return (
    <div ref={ref}>
      <svg className="chart" viewBox={`0 0 ${w} ${items.length * rowH}`} role="img"
        aria-label={items.map((i) => `${i.label}: ${format(i.value)}`).join("; ")}>
        {items.map((it, i) => {
          const bw = Math.max(it.value > 0 ? 2 : 0, (it.value / m) * plotW);
          const y = i * rowH;
          return (
            <g key={it.label}>
              {stacked
                ? <text x={0} y={y + 12} className="label-ink">{truncate(it.label, Math.floor(w / 6.2))}</text>
                : <text x={lw - 8} y={y + rowH / 2 + 4} textAnchor="end" className="label-ink">{truncate(it.label, Math.floor(lw / 6.5))}</text>}
              <rect x={lw} y={y + barY} width={plotW} height={barH} fill="var(--surface-2)" rx={4} />
              <rect className="mark" x={lw} y={y + barY} width={bw} height={barH} rx={4} fill={it.color ?? color} />
              <text x={lw + plotW + 8} y={y + barY + barH / 2 + 4} className="value-ink tabular">{format(it.value)}</text>
              <rect className="hit" x={0} y={y} width={w} height={rowH} tabIndex={0}
                onPointerMove={(e) => t.show(e, format(it.value), it.note ?? it.label)} onPointerLeave={t.hide}
                onFocus={(e) => t.show(e, format(it.value), it.note ?? it.label)} onBlur={t.hide} />
            </g>
          );
        })}
      </svg>
      {t.node}
    </div>
  );
}

/** SHAP contributions: diverging bars around zero (red = pushes towards the explained class). */
export function DivergingBars({ items, explainedClass }: {
  items: { label: string; value: number }[]; explainedClass: string;
}) {
  const t = useTip();
  const [ref, w] = useWidth(320);
  const rowH = 28, labelW = Math.min(300, Math.round(w * 0.42));
  const plotW = w - labelW - 16;
  const m = Math.max(1e-9, ...items.map((i) => Math.abs(i.value)));
  const mid = labelW + plotW / 2;
  return (
    <div ref={ref}>
      <svg className="chart" viewBox={`0 0 ${w} ${items.length * rowH + 22}`} role="img"
        aria-label={items.map((i) => `${i.label}: ${i.value.toFixed(3)}`).join("; ")}>
        <line className="axis-line" x1={mid} x2={mid} y1={0} y2={items.length * rowH} />
        {items.map((it, i) => {
          const len = (Math.abs(it.value) / m) * (plotW / 2 - 40);
          const pos = it.value > 0;
          const y = i * rowH;
          return (
            <g key={it.label}>
              <text x={labelW - 10} y={y + rowH / 2 + 4} textAnchor="end" className="label-ink">{truncate(it.label, Math.floor(labelW / 6.3))}</text>
              <rect className="mark" x={pos ? mid + 1 : mid - 1 - len} y={y + 7} width={Math.max(2, len)} height={rowH - 14} rx={4}
                fill={pos ? "var(--div-pos)" : "var(--div-neg)"} />
              <text x={pos ? mid + len + 6 : mid - len - 6} y={y + rowH / 2 + 4} textAnchor={pos ? "start" : "end"}
                className="value-ink tabular">{it.value > 0 ? "+" : ""}{it.value.toFixed(3)}</text>
              <rect className="hit" x={0} y={y} width={w} height={rowH} tabIndex={0}
                onPointerMove={(e) => t.show(e, `${it.value > 0 ? "+" : ""}${it.value.toFixed(4)}`,
                  `${it.label} — ${pos ? "pushes towards" : "pushes away from"} ${explainedClass}`)}
                onPointerLeave={t.hide} onBlur={t.hide}
                onFocus={(e) => t.show(e, it.value.toFixed(4), it.label)} />
            </g>
          );
        })}
        <text x={mid - 8} y={items.length * rowH + 16} textAnchor="end">← lowers {explainedClass} score</text>
        <text x={mid + 8} y={items.length * rowH + 16}>raises {explainedClass} score →</text>
      </svg>
      {t.node}
    </div>
  );
}

/** Part-to-whole: how the 0-100 score is made up. Legend + values always visible. */
export function StackedScore({ parts, total = 100 }: { parts: { label: string; value: number; color: string }[]; total?: number }) {
  const t = useTip();
  const [ref, w] = useWidth(200);
  const h = 22;
  let x = 0;
  return (
    <div ref={ref}>
      <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Score contributions">
        <rect x={0} y={0} width={w} height={h} rx={5} fill="var(--surface-2)" />
        {parts.filter((p) => p.value > 0).map((p) => {
          const pw = (p.value / total) * w;
          const r = <rect key={p.label} className="mark" x={x} y={0} width={Math.max(0, pw - 2)} height={h} rx={4} fill={p.color}
            onPointerMove={(e) => t.show(e, `${p.value.toFixed(1)} pts`, p.label)} onPointerLeave={t.hide} />;
          x += pw;
          return r;
        })}
      </svg>
      <div className="legend">
        {parts.map((p) => (
          <span key={p.label}><i style={{ background: p.color }} />{p.label} <b className="tabular">{p.value.toFixed(1)}</b></span>
        ))}
      </div>
      {t.node}
    </div>
  );
}

/** Two-series grouped bars (e.g. reference vs current distribution). */
export function GroupedBars({ categories, series, format = (v) => v.toFixed(2) }: {
  categories: string[]; series: { name: string; color: string; values: number[] }[]; format?: (v: number) => string;
}) {
  const t = useTip();
  const [ref, w] = useWidth(280);
  const h = 190, pad = { l: 40, b: 24, t: 14 };
  const plotH = h - pad.b - pad.t, groupW = (w - pad.l) / categories.length;
  const barW = Math.min(36, (groupW - 24) / series.length);
  const m = Math.max(1e-9, ...series.flatMap((s) => s.values));
  const ticks = [0, m / 2, m];
  return (
    <div ref={ref}>
      <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Grouped bar chart">
        {ticks.map((tk) => {
          const y = pad.t + plotH - (tk / m) * plotH;
          return <g key={tk}><line className="grid-line" x1={pad.l} x2={w} y1={y} y2={y} />
            <text x={pad.l - 6} y={y + 4} textAnchor="end">{format(tk)}</text></g>;
        })}
        {categories.map((c, ci) => {
          const gx = pad.l + ci * groupW + (groupW - barW * series.length - 2 * (series.length - 1)) / 2;
          return (
            <g key={c}>
              {series.map((s, si) => {
                const v = s.values[ci] ?? 0;
                const bh = (v / m) * plotH;
                const x = gx + si * (barW + 2);
                return <rect key={s.name} className="mark" x={x} y={pad.t + plotH - bh} width={barW} height={Math.max(bh, v > 0 ? 2 : 0)}
                  rx={4} fill={s.color} tabIndex={0}
                  onPointerMove={(e) => t.show(e, format(v), `${s.name} · ${c}`)} onPointerLeave={t.hide}
                  onFocus={(e) => t.show(e, format(v), `${s.name} · ${c}`)} onBlur={t.hide} />;
              })}
              <text x={pad.l + ci * groupW + groupW / 2} y={h - 6} textAnchor="middle" className="label-ink">{c}</text>
            </g>
          );
        })}
        <line className="axis-line" x1={pad.l} x2={w} y1={pad.t + plotH} y2={pad.t + plotH} />
      </svg>
      <div className="legend">{series.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>
      {t.node}
    </div>
  );
}

/** Single-series line with crosshair tooltip. */
export function LineChart({ points, format = (v) => String(v), color = "var(--s1)", label }: {
  points: { x: string; y: number }[]; format?: (v: number) => string; color?: string; label: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [ref, w] = useWidth(260);
  const h = 150, pad = { l: 44, r: 10, t: 10, b: 22 };
  if (points.length === 0) return <div ref={ref} className="state small">No data in this window.</div>;
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const m = Math.max(1e-9, ...points.map((p) => p.y));
  const px = (i: number) => pad.l + (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const py = (v: number) => pad.t + plotH - (v / m) * plotH;
  const d = points.map((p, i) => `${i ? "L" : "M"}${px(i)},${py(p.y)}`).join(" ");
  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const rel = ((e.clientX - r.left) / r.width) * w - pad.l;
    setHover(Math.max(0, Math.min(points.length - 1, Math.round((rel / plotW) * (points.length - 1)))));
  };
  return (
    <div ref={ref}><svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      {[0, m / 2, m].map((tk) => <g key={tk}><line className="grid-line" x1={pad.l} x2={w - pad.r} y1={py(tk)} y2={py(tk)} />
        <text x={pad.l - 6} y={py(tk) + 4} textAnchor="end">{format(tk)}</text></g>)}
      <path d={d} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
      {points.length <= 2 ? points.map((p, i) => <circle key={p.x} cx={px(i)} cy={py(p.y)} r={4} fill={color} />) : null}
      <text x={pad.l} y={h - 4}>{points[0].x}</text>
      <text x={w - pad.r} y={h - 4} textAnchor="end">{points[points.length - 1].x}</text>
      {hover !== null ? (
        <g>
          <line x1={px(hover)} x2={px(hover)} y1={pad.t} y2={pad.t + plotH} stroke="var(--axis)" />
          <circle cx={px(hover)} cy={py(points[hover].y)} r={4.5} fill={color} stroke="var(--surface)" strokeWidth={2} />
          <g transform={`translate(${Math.min(px(hover) + 8, w - 150)}, ${pad.t + 4})`}>
            <rect width={140} height={36} rx={6} fill="var(--surface)" stroke="var(--ring)" />
            <text x={8} y={15} className="value-ink">{format(points[hover].y)}</text>
            <text x={8} y={29}>{points[hover].x}</text>
          </g>
        </g>
      ) : null}
      <rect className="hit" x={pad.l} y={pad.t} width={plotW} height={plotH} onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
    </svg></div>
  );
}

/** Confusion matrix as a sequential (one-hue) heat grid with counts. */
export function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const t = useTip();
  const max = Math.max(1, ...matrix.flat());
  const steps = ["var(--seq-0)", "var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)"];
  return (
    <div>
      <div className="cm" style={{ gridTemplateColumns: `90px repeat(${labels.length}, 1fr)` }}>
        <div className="hdr">actual ↓ / pred →</div>
        {labels.map((l) => <div key={l} className="hdr">{l}</div>)}
        {matrix.map((row, i) => [
          <div key={`h${i}`} className="hdr" style={{ justifyContent: "flex-end", paddingRight: 8 }}>{labels[i]}</div>,
          ...row.map((v, j) => {
            const k = v === 0 ? 0 : Math.min(4, 1 + Math.floor((v / max) * 3.999));
            return (
              <div key={`${i}-${j}`} className="cell" tabIndex={0}
                style={{ background: steps[k], color: k >= 3 ? "#fff" : "var(--ink)", outline: i === j ? "1.5px solid var(--ink-2)" : undefined }}
                onPointerMove={(e) => t.show(e, `${v} contracts`, `actual ${labels[i]} → predicted ${labels[j]}`)}
                onPointerLeave={t.hide} onFocus={(e) => t.show(e, `${v}`, `actual ${labels[i]} → predicted ${labels[j]}`)} onBlur={t.hide}>
                {v}
              </div>
            );
          }),
        ])}
      </div>
      <div className="legend small">Diagonal (outlined) = correct. Darker = more contracts.</div>
      {t.node}
    </div>
  );
}

function truncate(s: string, n: number) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}
