import type { Data } from "../../shared/data/types";
import { color, font, radius, size, space, weight } from "../../shared/theme/tokens";
import { EmptyNotice } from "../../shared/ui/States";

const NAMES: Record<string, string> = { page: "Assembled page", reranked: "Today's rows, re-ranked" };

export function Gate({ gate }: { gate: Data["gate"] }) {
  const entries = Object.entries(gate);
  if (entries.length === 0) return <EmptyNotice what="gate decisions" command="python scripts/gate.py" />;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: space[6] }}>
      {entries.map(([k, g]) => (
        <div key={k} style={{ background: color.card, borderRadius: radius.lg, padding: space[5] }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <span style={{ fontFamily: font.display, fontWeight: weight.heavy, fontSize: size.lg }}>{NAMES[k] ?? k}</span>
            <span style={{ fontWeight: weight.semibold, color: g.ship ? color.basil : color.chili }}>{g.ship ? "Ships" : "Held"}</span>
          </div>
          <ul style={{ margin: `${space[3]}px 0 0`, padding: 0, listStyle: "none", display: "grid", gap: space[2] }}>
            {g.checks.map((c) => (
              <li key={c.name} style={{ fontSize: size.sm, display: "grid", gridTemplateColumns: "20px 1fr", gap: space[2] }}>
                <span aria-label={c.passed ? "passed" : "failed"} style={{ color: c.passed ? color.basil : color.chili, fontWeight: weight.bold }}>{c.passed ? "✓" : "✗"}</span>
                <span>
                  <b style={{ fontWeight: weight.semibold }}>{c.name}</b>: {c.detail}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
