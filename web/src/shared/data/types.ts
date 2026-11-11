export type Card = { id: number; name: string; cuisine: string; eta: number; promoted: boolean; new: boolean; load: number };
export type PageRow = { kind: string; cards: Card[] };
export type Sample = { customer: number; past_orders: number; today: PageRow[]; assembled: PageRow[] };

export type PolicyRow = {
  name: string;
  label: string;
  orders_per_session: number;
  lift: number;
  lift_min: number;
  lift_max: number;
  gap_closed: number | null;
  late_share: number;
  new_merchant_share: number;
  top10_share: number;
  gmv_per_session: number;
};

export type Estimate = { diff: number; lo: number; hi: number };
export type Experiment = {
  truth: number;
  customer_split: Estimate[];
  customer_split_no_cuped: Estimate[];
  switchback: Estimate[];
};

export type Check = { name: string; passed: boolean; detail: string };

export type Data = {
  seeds: number[];
  policies: PolicyRow[];
  training: {
    ndcg8: Record<string, number>;
    retrieval: { all: Record<string, number>; cold_start: Record<string, number>; cold_start_orders: number };
    sequential: { recall10_gru: number; recall10_counts: number; ships: boolean };
    attention: { row_decay: number; card_decay: number };
  };
  experiments: Record<string, Experiment>;
  gate: Record<string, { ship: boolean; checks: Check[] }>;
  serving: { p50_ms: number; p95_ms: number };
  heldout: { lift: Record<string, Estimate & { relative: number }>; metrics: Record<string, Record<string, number>> } | null;
  samples: { minute: string; customers: Sample[] };
};
