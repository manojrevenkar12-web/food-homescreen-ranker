import type { Data } from "../../shared/data/types";
import { pct } from "../../shared/format";
import { color, font, size, space, stroke, weight } from "../../shared/theme/tokens";

function Stat({ n, label }: { n: string; label: string }) {
  return (
    <div style={{ borderTop: `${stroke.rule}px solid ${color.ink}`, paddingTop: space[3] }}>
      <div style={{ fontFamily: font.display, fontWeight: weight.heavy, fontSize: size.xl }}>{n}</div>
      <div style={{ color: color.muted, fontSize: size.sm }}>{label}</div>
    </div>
  );
}

export function Models({ t }: { t: Data["training"] }) {
  const r = t.retrieval;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: space[6] }}>
      <Stat n={`${pct(r.all["tt@50"], 0)} vs ${pct(r.all["pop@50"], 0)}`} label="next orders in the two-tower's top 50, against the 50 most popular nearby" />
      <Stat n={`${pct(r.cold_start["tt@50"], 0)} vs 0%`} label={`orders to merchants with no order history (${r.cold_start_orders} of them) found in the top 50: the content path at work`} />
      <Stat n={`${t.ndcg8.model.toFixed(3)} vs ${t.ndcg8.served.toFixed(3)}`} label="NDCG@8 on held-out days: LambdaRank against the order the page showed" />
      <Stat n={`${pct(t.sequential.recall10_gru, 0)} vs ${pct(t.sequential.recall10_counts, 0)}`}
        label={`next-order recall@10: GRU against the count-based order-again row (${t.sequential.ships ? "the GRU ships" : "the GRU does not ship"})`} />
      <Stat n={`${t.attention.row_decay.toFixed(2)} · ${t.attention.card_decay.toFixed(2)}`} label="row and card attention, measured from shuffled exploration pages" />
    </div>
  );
}
