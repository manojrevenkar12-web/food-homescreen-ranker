import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { Card, PageRow } from "../../shared/data/types";
import { color, font, motion as m, phone, radius, size, space, weight } from "../../shared/theme/tokens";

export type Lens = "none" | "new" | "promoted" | "busy";

const ROW_LABELS: Record<string, string> = {
  for_you: "For you",
  recommended: "Recommended",
  order_again: "Order again",
  deals: "Deals",
  fast: "Fast delivery",
  new: "New on the app",
};

export const rowLabel = (kind: string) =>
  ROW_LABELS[kind] ?? (kind.startsWith("cuisine:") ? kind.slice(8).replace(/^\w/, (c) => c.toUpperCase()) : kind);

const BUSY = 0.7;

function lit(c: Card, lens: Lens) {
  return (lens === "new" && c.new) || (lens === "promoted" && c.promoted) || (lens === "busy" && c.load >= BUSY);
}

function CardView({ c, lens, i }: { c: Card; lens: Lens; i: number }) {
  const reduce = useReducedMotion();
  const on = lit(c, lens);
  const dim = lens !== "none" && !on;
  const accent = lens === "busy" ? color.saffron : color.basil;
  return (
    <motion.li
      initial={reduce ? false : { opacity: 0, y: space[2] }}
      animate={{ opacity: dim ? 0.35 : 1, y: 0, boxShadow: on ? `0 0 0 2px ${accent}` : "0 0 0 0px transparent" }}
      transition={{ delay: reduce ? 0 : i * m.stagger, duration: m.base, ease: m.ease }}
      style={{ listStyle: "none", flex: `0 0 ${phone.cardWidth}px`, height: phone.cardHeight, borderRadius: radius.md, background: color.card, padding: space[2], display: "flex", flexDirection: "column", justifyContent: "space-between" }}
      aria-label={`${c.name}, ${c.cuisine}, ${c.eta} minutes${c.promoted ? ", promoted" : ""}${c.new ? ", new" : ""}${c.load >= BUSY ? ", busy kitchen" : ""}`}
    >
      <span style={{ fontSize: size.sm, fontWeight: weight.semibold, lineHeight: 1.2 }}>{c.name}</span>
      <span style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: size.xs, color: color.muted }}>
        <span>{c.eta} min</span>
        <span style={{ display: "flex", gap: space[1] }}>
          {c.promoted && <span title="Promoted" style={{ color: color.ink }}>Ad</span>}
          {c.new && <span title="New" style={{ color: color.basil, fontWeight: weight.semibold }}>New</span>}
          {c.load >= BUSY && <span title="Kitchen is busy" style={{ color: color.saffron, fontWeight: weight.semibold }}>Busy</span>}
        </span>
      </span>
    </motion.li>
  );
}

/** A phone home screen: rows the customer scrolls down, cards they swipe along. */
export function Phone({ title, rows, lens, k }: { title: string; rows: PageRow[]; lens: Lens; k: string }) {
  return (
    <figure style={{ margin: 0, width: phone.width, flex: `0 0 ${phone.width}px` }}>
      <figcaption style={{ fontFamily: font.display, fontWeight: weight.heavy, fontSize: size.lg, marginBottom: space[3] }}>{title}</figcaption>
      <div style={{ background: color.panel, borderRadius: radius.phone, padding: `${space[5]}px ${space[3]}px`, height: 560, overflow: "hidden" }}>
        <AnimatePresence mode="wait">
          <motion.div key={k} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: m.fast }}>
            {rows.map((r, ri) => (
              <div key={`${r.kind}-${ri}`} style={{ marginBottom: space[4] }}>
                <div style={{ fontSize: size.sm, fontWeight: weight.semibold, margin: `0 ${space[1]}px ${space[2]}px` }}>{rowLabel(r.kind)}</div>
                <ul style={{ display: "flex", gap: space[2], margin: 0, padding: 0, overflow: "hidden" }}>
                  {r.cards.map((c, i) => (
                    <CardView key={c.id} c={c} lens={lens} i={ri * 2 + i} />
                  ))}
                </ul>
              </div>
            ))}
          </motion.div>
        </AnimatePresence>
      </div>
    </figure>
  );
}
