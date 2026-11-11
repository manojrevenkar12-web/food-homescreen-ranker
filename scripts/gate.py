"""Ship or hold, per treatment, from what an experiment and the serving benchmark would show.

The experiment is the first switchback replication (the one design whose interval holds the true effect);
the truth is never read here.
Usage: python scripts/gate.py
"""

from __future__ import annotations

from fhr.config import platform

from _common import read, write


def decide(exp: dict, serving: dict) -> dict:
    g = platform()["gate"]
    sb = exp["switchback"][0]
    arms = sb["guardrails"]
    t, c = arms["treatment"], arms["control"]
    checks = [
        (
            "orders per session",
            sb["lo"] > g["min_lift_lb"],
            f"lift {sb['diff']:+.4f} (95% {sb['lo']:+.4f} to {sb['hi']:+.4f}) must clear {g['min_lift_lb']}",
        ),
        (
            "late orders",
            (t["late_share"] - c["late_share"]) * 100 <= g["max_late_increase_pp"],
            f"late share {100 * c['late_share']:.2f}% -> {100 * t['late_share']:.2f}%",
        ),
        (
            "new merchants",
            t["new_merchant_share"] >= g["min_new_merchant_share"],
            f"orders to new merchants {100 * t['new_merchant_share']:.1f}% (floor {100 * g['min_new_merchant_share']:.0f}%)",
        ),
        (
            "concentration",
            (t["top10_share"] - c["top10_share"]) * 100 <= g["max_top10_share_increase_pp"],
            f"top-10% merchants' share {100 * c['top10_share']:.1f}% -> {100 * t['top10_share']:.1f}%",
        ),
        (
            "latency",
            serving["p95_ms"] <= g["max_p95_ms"],
            f"p95 {serving['p95_ms']:.1f} ms (limit {g['max_p95_ms']} ms)",
        ),
    ]
    return {
        "ship": all(ok for _, ok, _ in checks),
        "checks": [{"name": n, "passed": bool(ok), "detail": d} for n, ok, d in checks],
    }


def main() -> None:
    exp, serving = read("experiments.json"), read("serving.json")
    out = {name: decide(t, serving) for name, t in exp["treatments"].items()}
    for name, d in out.items():
        print(name, "SHIP" if d["ship"] else "HOLD", [c["detail"] for c in d["checks"] if not c["passed"]])
    write("gate.json", out)


if __name__ == "__main__":
    main()
