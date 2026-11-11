import { motion, useReducedMotion } from "framer-motion";
import type { PolicyRow } from "../../shared/data/types";
import { pct, signed } from "../../shared/format";
import { color, font, motion as m, radius, size, space, stroke, weight } from "../../shared/theme/tokens";
import { EmptyNotice } from "../../shared/ui/States";

/** Lift over today's screen, with the range across cities, and the guardrails next to it. */
export function Policies({ rows }: { rows: PolicyRow[] }) {
  const reduce = useReducedMotion();
  if (rows.length === 0) return <EmptyNotice what="policy results" command="python scripts/evaluate.py" />;
  const max = Math.max(...rows.map((r) => r.lift_max), 0.01);
  const th = { fontWeight: weight.regular, color: color.muted, fontSize: size.xs, textAlign: "right" as const, padding: `0 0 ${space[2]}px ${space[3]}px` };
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 720 }}>
        <thead>
          <tr>
            <th scope="col" style={{ ...th, textAlign: "left", paddingLeft: 0 }}>Home screen</th>
            <th scope="col" style={{ ...th, textAlign: "left", width: "34%" }}>Orders per session vs today</th>
            <th scope="col" style={th}>Gap to oracle closed</th>
            <th scope="col" style={th}>Late orders</th>
            <th scope="col" style={th}>To new merchants</th>
            <th scope="col" style={th}>Top 10% share</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.name} style={{ borderTop: `${stroke.hair}px solid ${color.line}` }}>
              <th scope="row" style={{ textAlign: "left", fontWeight: weight.semibold, padding: `${space[3]}px 0` }}>{r.label}</th>
              <td style={{ padding: `0 ${space[3]}px` }}>
                {r.name === "current" ? (
                  <span style={{ color: color.muted, fontSize: size.sm }}>{pct(r.orders_per_session)} of sessions order</span>
                ) : (
                  <div style={{ display: "flex", alignItems: "center", gap: space[3] }}>
                    <div style={{ position: "relative", flex: 1, height: space[3], background: color.panel, borderRadius: radius.pill }}>
                      <motion.div initial={reduce ? false : { width: 0 }} whileInView={{ width: `${(r.lift / max) * 100}%` }} viewport={{ once: true }}
                        transition={{ delay: i * m.stagger, duration: m.base * 2, ease: m.ease }}
                        style={{ height: "100%", borderRadius: radius.pill, background: r.name === "oracle" ? color.muted : color.basil }} />
                      <span aria-hidden style={{ position: "absolute", top: -2, bottom: -2, left: `${(r.lift_min / max) * 100}%`, width: `${((r.lift_max - r.lift_min) / max) * 100}%`, borderLeft: `${stroke.rule}px solid ${color.ink}`, borderRight: `${stroke.rule}px solid ${color.ink}` }} />
                    </div>
                    <span style={{ fontFamily: font.display, fontWeight: weight.heavy, minWidth: 64, textAlign: "right" }}>{signed(r.lift)}</span>
                  </div>
                )}
              </td>
              <td style={{ textAlign: "right" }}>{r.gap_closed == null ? "–" : pct(r.gap_closed, 0)}</td>
              <td style={{ textAlign: "right" }}>{pct(r.late_share)}</td>
              <td style={{ textAlign: "right" }}>{pct(r.new_merchant_share)}</td>
              <td style={{ textAlign: "right" }}>{pct(r.top10_share)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ color: color.muted, fontSize: size.sm }}>
        Bars: mean lift over three simulated cities; brackets: the lowest and highest city. The oracle knows every customer's taste and
        sets the ceiling.
      </p>
    </div>
  );
}
