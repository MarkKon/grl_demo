"""Build tables and simple SVG plots from ablation metric JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ["mean_recall", "mean_regret"]
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def parse_k(value: Any) -> int:
    return int(float(value))


def load_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(metrics_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if rows:
        return rows
    for path in sorted(metrics_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "kind",
        "dataset_label",
        "dataset",
        "k",
        "num_samples",
        "mean_recall",
        "mean_regret",
        "checkpoint",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["dataset_label"], item["method"], parse_k(item["k"]))):
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_metric_tables(rows: list[dict[str, Any]], tables_dir: Path) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_label"]].append(row)

    written = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        methods = sorted({row["method"] for row in dataset_rows})
        ks = sorted({parse_k(row["k"]) for row in dataset_rows})
        lookup = {
            (row["method"], parse_k(row["k"])): row
            for row in dataset_rows
        }
        for metric in METRICS:
            table_rows = []
            for method in methods:
                values = []
                for k in ks:
                    value = lookup.get((method, k), {}).get(metric)
                    values.append("" if value is None else f"{float(value):.4f}")
                table_rows.append([method, *values])
            output_path = tables_dir / f"{dataset}_{metric}.md"
            output_path.write_text(
                f"# {dataset}: {metric}\n\n"
                + markdown_table(["method", *[f"K={k}" for k in ks]], table_rows),
                encoding="utf-8",
            )
            written.append(output_path)
    return written


def svg_line_plot(dataset: str, metric: str, rows: list[dict[str, Any]]) -> str:
    width, height = 920, 560
    left, right, top, bottom = 90, 260, 40, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    methods = sorted({row["method"] for row in rows})
    ks = sorted({parse_k(row["k"]) for row in rows})
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    y_min = 0.0
    y_max = 1.0 if not values else max(1.0, max(values))
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    def x_pos(k: int) -> float:
        if len(ks) == 1:
            return left + plot_width / 2
        return left + (ks.index(k) / (len(ks) - 1)) * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="Arial" font-size="18">{dataset}: {metric}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
    ]
    for k in ks:
        x = x_pos(k)
        lines.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 5}" stroke="#222"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="Arial" font-size="12">K={k}</text>')
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(tick)
        lines.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')

    lookup: dict[tuple[str, int], float] = {
        (row["method"], parse_k(row["k"])): float(row[metric])
        for row in rows
        if row.get(metric) is not None
    }
    for index, method in enumerate(methods):
        color = COLORS[index % len(COLORS)]
        points = [
            (x_pos(k), y_pos(lookup[(method, k)]))
            for k in ks
            if (method, k) in lookup
        ]
        if len(points) >= 2:
            point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            lines.append(f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        legend_y = top + 20 + index * 20
        lines.append(f'<line x1="{left + plot_width + 30}" y1="{legend_y}" x2="{left + plot_width + 55}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{left + plot_width + 65}" y="{legend_y + 4}" font-family="Arial" font-size="12">{method}</text>')

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_plots(rows: list[dict[str, Any]], plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_label"]].append(row)

    written = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        for metric in METRICS:
            output_path = plots_dir / f"{dataset}_{metric}.svg"
            output_path.write_text(svg_line_plot(dataset, metric, dataset_rows), encoding="utf-8")
            written.append(output_path)
    return written


def write_summary(run_dir: Path, rows: list[dict[str, Any]], tables: list[Path], plots: list[Path]) -> None:
    datasets = sorted({row["dataset_label"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    text = [
        "# Ablation Summary",
        "",
        f"- metrics: `{run_dir / 'metrics.csv'}`",
        f"- datasets: {', '.join(datasets)}",
        f"- methods: {', '.join(methods)}",
        "",
        "## Tables",
        "",
    ]
    text.extend(f"- `{path.relative_to(run_dir)}`" for path in tables)
    text.extend(["", "## Plots", ""])
    text.extend(f"- `{path.relative_to(run_dir)}`" for path in plots)
    text.append("")
    (run_dir / "summary.md").write_text("\n".join(text), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = load_rows(run_dir / "metrics")
    if not rows:
        raise SystemExit(f"no metric JSON files found in {run_dir / 'metrics'}")
    write_csv(rows, run_dir / "metrics.csv")
    tables = write_metric_tables(rows, run_dir / "tables")
    plots = write_plots(rows, run_dir / "plots")
    write_summary(run_dir, rows, tables, plots)
    print(f"wrote {run_dir / 'metrics.csv'}")
    print(f"wrote {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
