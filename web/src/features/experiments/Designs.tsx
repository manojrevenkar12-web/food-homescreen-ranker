import { useState } from "react";
import type { Estimate, Experiment } from "../../shared/data/types";
import { pp } from "../../shared/format";
import { color, font, radius, size, space, stroke, weight } from "../../shared/theme/tokens";
import { Segmented } from "../../shared/ui/Segmented";
import { EmptyNotice } from "../../shared/ui/States";

const TREATMENTS: { value: string; label: string; what: string }[] = [
  { value: "page", label: "Assembled page", what: "the assembled page" },
  { value: "reranked", label: "Re-ranked rows", what: "today's rows, re-ranked" },
];

const DESIGNS: { key: keyof Omit<Experiment, "truth">; label: string; note: string }[] = [
  { key: "customer_split_no_cuped", label: "Customer split", note: "half the customers each" },
  { key: "customer_split", label: "Customer split + CUPED", note: "pre-period orders as covariate" },
  { key: "switchback", label: "Switchback, 3-hour blocks", note: "the whole city per block, first hour dropped" },
];

function Row({ label, note, est, truth, lo, hi }: { label: string; note: string; est: Estimate[]; truth: number; lo: number; hi: number }) {
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  const covered = est.filter((e) => e.lo <= truth && truth <= e.hi).length;
  const bias = est.reduce((a, e) => a + e.diff, 0) / est.length / truth - 1;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(180px, 1fr) 3fr", gap: space[5], alignItems: "center", padding: `${space[4]}px 0`, borderTop: `${stroke.hair}px solid ${color.line}` }}>
      <div>
        <div style={{ fontWeight: weight.semibold }}>{label}</div>
        <div style={{ color: color.muted, fontSize: size.xs }}>{note}</div>
        <div style={{ fontSize: size.sm, marginTop: space[1], color: covered === est.length ? color.basil : color.chili }}>
          interval holds the truth in {covered} of {est.length}
        </div>
        <div style={{ fontSize: size.xs, color: color.muted }}>
          on average {Math.abs(Math.round(bias * 100))}% {bias < 0 ? "below" : "above"} it
        </div>
      </div>
      <div style={{ position: "relative", height: est.length * space[4] + space[2] }} role="img" aria-label={`${label}: ${est.map((e) => pp(e.diff)).join(", ")}; truth ${pp(truth)}`}>
        <span aria-hidden style={{ position: "absolute", left: x(truth), top: 0, bottom: 0, borderLeft: `${stroke.rule}px dashed ${color.ink}` }} />
        {est.map((e, i) => (
          <div key={i} style={{ position: "absolute", top: i * space[4] + space[1], left: x(e.lo), width: `calc(${x(e.hi)} - ${x(e.lo)})`, height: stroke.rule, background: color.muted }}>
            <span style={{ position: "absolute", left: `${((e.diff - e.lo) / (e.hi - e.lo)) * 100}%`, top: -4, width: 10, height: 10, marginLeft: -5, borderRadius: radius.pill, background: e.lo <= truth && truth <= e.hi ? color.basil : color.chili }} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function Designs({ experiments }: { experiments: Record<string, Experiment> }) {
  const [which, setWhich] = useState("page");
  const options = TREATMENTS.filter((t) => experiments[t.value]);
  const e = experiments[which];
  if (!e) return <EmptyNotice what="experiment results" command="python scripts/experiments.py" />;
  const all = DESIGNS.flatMap((d) => e[d.key]).flatMap((x) => [x.lo, x.hi]).concat([e.truth]);
  const pad = (Math.max(...all) - Math.min(...all)) * 0.08;
  const lo = Math.min(...all) - pad;
  const hi = Math.max(...all) + pad;
  return (
    <div>
      {options.length > 1 && (
        <div style={{ marginBottom: space[4] }}>
          <Segmented label="Treatment" options={options} value={which} onChange={setWhich} />
        </div>
      )}
      <p style={{ margin: `0 0 ${space[3]}px`, fontSize: size.sm }}>
        <span style={{ fontFamily: font.display, fontWeight: weight.heavy, fontSize: size.lg }}>{pp(e.truth)}</span> orders per session:
        the true effect of giving every customer {TREATMENTS.find((t) => t.value === which)?.what} for the whole week (dashed line).
      </p>
      {DESIGNS.map((d) => (
        <Row key={d.key} label={d.label} note={d.note} est={e[d.key]} truth={e.truth} lo={lo} hi={hi} />
      ))}
    </div>
  );
}
