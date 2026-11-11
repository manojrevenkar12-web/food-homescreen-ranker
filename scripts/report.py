"""Collect reports/*.json into web/public/data.json and the results block of README.md.

Usage: python scripts/report.py
"""

from __future__ import annotations

import json
import re

import numpy as np

from _common import REPORTS, ROOT, read

LABELS = {
    "current": "Today's screen",
    "reranked": "Today's rows, re-ranked",
    "page": "Assembled page",
    "page_no_congestion": "Assembled, no kitchen-load discount",
    "page_no_explore": "Assembled, no exploration",
    "page_no_gmv": "Assembled, orders only",
    "oracle": "Oracle (knows every taste)",
}


def policies(ev: dict) -> list[dict]:
    seeds = list(ev["seeds"].values())
    out = []
    for name in LABELS:
        m = [s["metrics"][name] for s in seeds]
        lifts = [s["lift_vs_current"][name]["relative"] if name != "current" else 0.0 for s in seeds]
        gaps = []
        for s in seeds:
            cur, orc = s["metrics"]["current"]["orders_per_session"], s["metrics"]["oracle"]["orders_per_session"]
            gaps.append((s["metrics"][name]["orders_per_session"] - cur) / (orc - cur))
        out.append(
            {
                "name": name,
                "label": LABELS[name],
                **{
                    k: float(np.mean([x[k] for x in m]))
                    for k in (
                        "orders_per_session",
                        "late_share",
                        "new_merchant_share",
                        "top10_share",
                        "gmv_per_session",
                    )
                },
                "lift": float(np.mean(lifts)),
                "lift_min": float(min(lifts)),
                "lift_max": float(max(lifts)),
                "gap_closed": None if name in ("current", "oracle") else float(np.mean(gaps)),
            }
        )
    return out


def build() -> dict:
    ev, ex, gate, serving, samples = (
        read(n) for n in ("evaluate.json", "experiments.json", "gate.json", "serving.json", "sample_pages.json")
    )
    first = next(iter(ev["seeds"].values()))
    heldout = None
    if (REPORTS / "heldout.json").exists():
        h = read("heldout.json")
        heldout = {
            "lift": h["lift_vs_current"],
            "metrics": h["metrics"],
            "training": h["training"],
            "guardrails": h["guardrails"],
        }
    return {
        "seeds": [int(s) for s in ev["seeds"]],
        "policies": policies(ev),
        "training": first["training"],
        "experiments": {
            k: {
                "truth": v["truth"]["diff"],
                **{
                    d: [{x: e[x] for x in ("diff", "lo", "hi")} for e in v[d]]
                    for d in ("customer_split", "customer_split_no_cuped", "switchback")
                },
            }
            for k, v in ex["treatments"].items()
        },
        "gate": gate,
        "serving": serving,
        "heldout": heldout,
        "samples": samples,
    }


def readme_block(d: dict) -> str:
    p = {r["name"]: r for r in d["policies"]}

    def pc(x, n=1):
        return f"{100 * x:.{n}f}%"

    def sg(x, n=1):
        return f"{100 * x:+.{n}f}%"

    page, rer = p["page"], p["reranked"]
    e = d["experiments"]["page"]
    cover = {
        k: sum(x["lo"] <= e["truth"] <= x["hi"] for x in e[k])
        for k in ("customer_split_no_cuped", "customer_split", "switchback")
    }
    width = {k: np.mean([x["hi"] - x["lo"] for x in e[k]]) for k in cover}
    t = d["training"]
    rows = [
        "| What | Result |",
        "|---|---|",
        f"| Orders per session vs today's screen (3 simulated cities) | re-ranked rows {sg(rer['lift'])}, **assembled page {sg(page['lift'])}** "
        f"(range {sg(page['lift_min'])} to {sg(page['lift_max'])}); closes {pc(page['gap_closed'], 0)} of the gap to an oracle |",
        f"| Late orders (over 45 min) | today {pc(p['current']['late_share'])}, assembled {pc(page['late_share'])}, "
        f"assembled without the kitchen-load discount {pc(p['page_no_congestion']['late_share'])} |",
        f"| Orders to new merchants / on the top 10% of merchants | today {pc(p['current']['new_merchant_share'])} / {pc(p['current']['top10_share'])}, "
        f"assembled {pc(page['new_merchant_share'])} / {pc(page['top10_share'])} |",
        f"| Retrieval: next orders in the two-tower's top 50 | {pc(t['retrieval']['all']['tt@50'], 0)} vs {pc(t['retrieval']['all']['pop@50'], 0)} for popularity; "
        f"never-ordered merchants {pc(t['retrieval']['cold_start']['tt@50'], 0)} vs 0% |",
        f"| Ranking: NDCG@8 on held-out days | LambdaRank {t['ndcg8']['model']:.3f}, the page as served {t['ndcg8']['served']:.3f}, popularity {t['ndcg8']['popularity']:.3f} |",
        f"| Order-again row: next-order recall@10 | GRU {pc(t['sequential']['recall10_gru'], 0)} vs counts {pc(t['sequential']['recall10_counts'], 0)}: "
        f"{'the GRU ships' if t['sequential']['ships'] else 'the GRU stays out'} |",
        f"| A/B design, true effect {100 * e['truth']:+.2f} pp | interval holds the truth: customer split {cover['customer_split_no_cuped']}/4, "
        f"with CUPED {cover['customer_split']}/4, switchback {cover['switchback']}/4 |",
        f"| Serving (HTTP, one core) | p50 {d['serving']['p50_ms']:.1f} ms, p95 {d['serving']['p95_ms']:.1f} ms |",
        "| Gate | " + ", ".join(f"{k}: **{'ships' if g['ship'] else 'held'}**" for k, g in d["gate"].items()) + " |",
    ]
    if d["heldout"]:
        h = d["heldout"]["lift"]
        rows.append(
            f"| Held-out city (scored once) | re-ranked {sg(h['reranked']['relative'])}, assembled **{sg(h['page']['relative'])}** "
            "orders per session |"
        )
    _ = width
    return "\n".join(rows)


def main() -> None:
    d = build()
    out = ROOT / "web" / "public" / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1, default=float) + "\n", encoding="utf-8")
    print("wrote web/public/data.json")
    readme = ROOT / "README.md"
    block = readme_block(d)
    if readme.exists():
        text = re.sub(
            r"(<!-- results:start -->\n).*?(\n<!-- results:end -->)",
            lambda m: m.group(1) + block + m.group(2),
            readme.read_text(encoding="utf-8"),
            flags=re.S,
        )
        readme.write_text(text, encoding="utf-8")
        print("updated README.md")
    else:
        print(block)


if __name__ == "__main__":
    main()
