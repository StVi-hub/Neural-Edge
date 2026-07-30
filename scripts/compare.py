"""
Build the results comparison table from the committed benchmark JSONs.

Reads every file in results/, sorts fastest-first, and prints a markdown table with
the speedup of each variant against the reference baseline. Nothing here measures
anything -- it only reads what benchmark.py already committed, which is the point:
the table in the README is derived from committed evidence, not retyped by hand.

The reference defaults to `finetuned-fp32` rather than `fp32-baseline`. The pretrained
model has an 80-class COCO head against 17 fine-tuned, worth ~2.8% of latency that has
nothing to do with optimisation, so dividing by it would inflate every speedup claim.

Usage:
    python scripts/compare.py
    python scripts/compare.py --reference finetuned-fp32 --out results/comparison.md
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        summary = data.get("summary", {})
        accuracy = data.get("map50_95") or {}
        out.append(
            {
                "label": data.get("label", path.stem),
                "p50_ms": summary.get("p50_ms"),
                "p95_ms": summary.get("p95_ms"),
                "fps": summary.get("fps"),
                "map50_95": accuracy.get("map50_95"),
                "vram_mb": summary.get("vram_mb"),
                "power_w": summary.get("power_w_mean"),
                "runs": data.get("config", {}).get("runs", 1),
            }
        )
    return out


def fmt(value, spec="", dash="-"):
    """Format a value, or return a dash when it was never measured."""
    if value is None:
        return dash
    return format(value, spec)


def main():
    ap = argparse.ArgumentParser(description="Compare committed benchmark results")
    ap.add_argument("--results", default=str(RESULTS_DIR))
    ap.add_argument("--reference", default="finetuned-fp32",
                    help="label whose latency is the denominator for speedup")
    ap.add_argument("--out", default=None, help="also write the table to this path")
    args = ap.parse_args()

    rows = load_results(Path(args.results))
    if not rows:
        raise SystemExit(f"no result JSONs found in {args.results}")

    ref = next((r for r in rows if r["label"] == args.reference), None)
    if ref is None or not ref["p50_ms"]:
        raise SystemExit(
            f"reference '{args.reference}' not found among: "
            f"{', '.join(r['label'] for r in rows)}"
        )
    ref_p50 = ref["p50_ms"]
    ref_map = ref["map50_95"]

    # Fastest first; anything without a latency number sorts last.
    rows.sort(key=lambda r: r["p50_ms"] if r["p50_ms"] is not None else float("inf"))

    header = (
        "| variant | p50 (ms) | p95 (ms) | FPS | speedup | mAP50-95 | mAP delta | VRAM (MB) | power (W) | runs |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in rows:
        speedup = f"{ref_p50 / r['p50_ms']:.2f}x" if r["p50_ms"] else "-"
        # Accuracy delta only means anything when both sides were measured the same way.
        if r["map50_95"] is not None and ref_map is not None:
            delta = r["map50_95"] - ref_map
            delta_s = f"{delta:+.4f}"
        else:
            delta_s = "-"
        marker = " *(ref)*" if r["label"] == args.reference else ""
        lines.append(
            f"| `{r['label']}`{marker} | {fmt(r['p50_ms'], '.2f')} | {fmt(r['p95_ms'], '.2f')} "
            f"| {fmt(r['fps'], '.1f')} | {speedup} | {fmt(r['map50_95'], '.4f')} | {delta_s} "
            f"| {fmt(r['vram_mb'], '.0f')} | {fmt(r['power_w'], '.1f')} | {r['runs']} |"
        )

    table = header + "\n".join(lines) + "\n"
    print(table)

    # Flag anything that undermines a published claim, rather than quietly tabulating it.
    single_run = [r["label"] for r in rows if r["runs"] < 3]
    if single_run:
        print(f"NOTE: single-run results (no variance estimate): {', '.join(single_run)}")
        print("      re-run with --runs 3 before quoting these publicly.")
    missing_map = [r["label"] for r in rows if r["map50_95"] is None]
    if missing_map:
        print(f"NOTE: no accuracy measured for: {', '.join(missing_map)}")
        print("      a speedup without its mAP cost is only half a claim; add --val-data.")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(table)
        print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
