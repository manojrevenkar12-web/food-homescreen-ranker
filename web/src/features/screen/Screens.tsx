import { useState } from "react";
import type { Data } from "../../shared/data/types";
import { color, size, space } from "../../shared/theme/tokens";
import { Segmented } from "../../shared/ui/Segmented";
import { EmptyNotice } from "../../shared/ui/States";
import { type Lens, Phone } from "./Phone";

const LENSES: { value: Lens; label: string }[] = [
  { value: "none", label: "All cards" },
  { value: "new", label: "New merchants" },
  { value: "promoted", label: "Promoted" },
  { value: "busy", label: "Busy kitchens" },
];

export function Screens({ samples }: { samples: Data["samples"] }) {
  const [who, setWho] = useState("0");
  const [lens, setLens] = useState<Lens>("none");
  if (samples.customers.length === 0) return <EmptyNotice what="sample pages" command="python scripts/serve_bench.py" />;
  const options = samples.customers.map((c, i) => ({
    value: String(i),
    label: c.past_orders === 0 ? "New customer" : `${c.past_orders} past orders`,
  }));
  const s = samples.customers[Number(who)];
  return (
    <div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: space[4], marginBottom: space[6] }}>
        <Segmented label="Customer" options={options} value={who} onChange={setWho} />
        <Segmented label="Highlight" options={LENSES} value={lens} onChange={setLens} />
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: space[7] }}>
        <Phone title="Today" rows={s.today} lens={lens} k={`t${who}`} />
        <Phone title="Assembled" rows={s.assembled} lens={lens} k={`a${who}`} />
      </div>
      <p style={{ color: color.muted, fontSize: size.sm, marginTop: space[4] }}>
        Customer {s.customer} of the simulated city at {samples.minute}, both pages built by the running service from the same live
        counters. A kitchen is marked busy at 70% of the orders per hour it said it can take.
      </p>
    </div>
  );
}
