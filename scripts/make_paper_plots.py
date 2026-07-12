"""Generate paper-ready plots for the main GRL ablation run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


METHOD_LABELS = {
    "random": "Random",
    "local": "Local discrepancy",
    "depth0_no_graph": "No message passing",
    "depth1_rank_knn_linf": "1-layer rank kNN",
    "depth2_rank_knn_linf": "2-layer rank kNN",
    "rank_knn_linf": "Rank kNN",
    "rank_adjacency": "Rank adjacency",
    "knn_euclidean": "Euclidean kNN",
    "knn_linf": r"$\ell_\infty$ kNN",
    "rank_knn_euclidean": "Rank Euclidean kNN",
    "coord_unshared_rank_knn_linf": "Coord-unshared",
    "flat_mlp": "Flat MLP",
}

DATASET_LABELS = {
    "uniform_d3_n64_test": r"$d=3,\ n=64$",
    "uniform_d3_n128_test": r"$d=3,\ n=128$",
    "uniform_d4_n64_test": r"$d=4,\ n=64$",
    "jittered_grid4_d3_n64_test": r"Jittered grid",
}

DATASET_SHORT_LABELS = {
    "uniform_d3_n64_test": "In-distribution",
    "uniform_d3_n128_test": "Larger n",
    "uniform_d4_n64_test": "Larger d",
    "jittered_grid4_d3_n64_test": "Jittered",
}

COLORS = {
    "random": "#8c8c8c",
    "local": "#A65628",
    "depth0_no_graph": "#785EF0",
    "depth1_rank_knn_linf": "#117733",
    "depth2_rank_knn_linf": "#332288",
    "rank_knn_linf": "#0072B2",
    "rank_adjacency": "#CC6677",
    "knn_euclidean": "#D55E00",
    "knn_linf": "#AA4499",
    "rank_knn_euclidean": "#44AA99",
    "coord_unshared_rank_knn_linf": "#DDCC77",
    "flat_mlp": "#000000",
}

MARKERS = {
    "random": "o",
    "local": "s",
    "depth0_no_graph": "^",
    "depth1_rank_knn_linf": "D",
    "depth2_rank_knn_linf": "P",
    "rank_knn_linf": "P",
    "rank_adjacency": "X",
    "knn_euclidean": "v",
    "knn_linf": "<",
    "rank_knn_euclidean": ">",
    "coord_unshared_rank_knn_linf": "h",
    "flat_mlp": "x",
}


@dataclass(frozen=True)
class Row:
    method: str
    dataset_label: str
    k: int
    recall: float
    regret: float


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_rows(path: Path) -> list[Row]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Row(
                    method=row["method"],
                    dataset_label=row["dataset_label"],
                    k=int(float(row["k"])),
                    recall=float(row["mean_recall"]),
                    regret=float(row["mean_regret"]),
                )
            )
    return rows


def by_key(rows: Iterable[Row]) -> dict[tuple[str, str, int], Row]:
    return {(row.method, row.dataset_label, row.k): row for row in rows}


def save_figure(fig: mpl.figure.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def line_for_method(
    ax: mpl.axes.Axes,
    rows: list[Row],
    method: str,
    metric: str,
    *,
    label: str | None = None,
) -> None:
    method_rows = sorted((row for row in rows if row.method == method), key=lambda row: row.k)
    if not method_rows:
        return
    values = [getattr(row, metric) for row in method_rows]
    ax.plot(
        [row.k for row in method_rows],
        values,
        color=COLORS[method],
        marker=MARKERS[method],
        markersize=4.5,
        linewidth=1.9,
        label=label or METHOD_LABELS[method],
    )


def format_curve_axes(ax: mpl.axes.Axes, ylabel: str, ymin: float, ymax: float) -> None:
    ax.set_xlabel(r"Subset budget $K$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([4, 8, 16])
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y")


def format_unit_square(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel(r"First coordinate")
    ax.set_ylabel(r"Second coordinate")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    for spine in ax.spines.values():
        spine.set_visible(True)


def plot_discrepancy_search_space(output_dir: Path) -> None:
    points = np.array(
        [
            [0.12, 0.18],
            [0.28, 0.72],
            [0.42, 0.38],
            [0.58, 0.88],
            [0.72, 0.27],
            [0.86, 0.58],
        ]
    )
    grid = np.array([(x, y) for x in points[:, 0] for y in points[:, 1]])
    critical = np.unique(
        np.array(
            [
                np.max(points[list(support)], axis=0)
                for size in (1, 2)
                for support in combinations(range(len(points)), size)
            ]
        ),
        axis=0,
    )
    def discrepancy(corner: np.ndarray) -> float:
        dominated = np.all(points <= corner, axis=1).sum()
        return float(dominated / len(points) - np.prod(corner))

    grid_scores = np.array([discrepancy(corner) for corner in grid])
    critical_scores = np.array([discrepancy(corner) for corner in critical])
    optimum = grid[np.argmax(grid_scores)]
    optimum_score = float(np.max(grid_scores))
    if not np.isclose(optimum_score, np.max(critical_scores)):
        raise RuntimeError("critical corners do not contain the grid maximizer")

    q_indices = [1, 2, 4]
    q_points = points[q_indices]
    q_pair_corners = np.unique(
        np.array(
            [
                np.max(q_points[list(support)], axis=0)
                for support in combinations(range(len(q_points)), 2)
            ]
        ),
        axis=0,
    )
    q_critical = np.unique(np.concatenate([q_points, q_pair_corners], axis=0), axis=0)

    point_color = "#0072B2"
    selected_color = "#117733"
    grid_color = "#A8A8A8"
    critical_color = "#785EF0"
    optimum_color = "#D55E00"

    fig, axes = plt.subplots(1, 4, figsize=(9.45, 2.62))
    for ax in axes:
        format_unit_square(ax)

    ax = axes[0]
    ax.add_patch(
        Rectangle(
            (0, 0),
            optimum[0],
            optimum[1],
            facecolor="#DCEAF7",
            edgecolor=point_color,
            linewidth=1.4,
            alpha=0.72,
            zorder=0,
        )
    )
    ax.scatter(points[:, 0], points[:, 1], s=28, color=point_color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(*optimum, s=94, marker="*", color=optimum_color, edgecolor="white", linewidth=0.7, zorder=5)
    ax.text(
        0.04,
        0.96,
        r"$\delta^-(y)=\frac{|\{p_i\leq y\}|}{n}-y_1y_2$",
        transform=ax.transAxes,
        va="top",
        fontsize=8.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8},
    )
    ax.text(
        0.04,
        0.05,
        rf"$\delta^-(y^\star)=1-{optimum[0]:.2f}\cdot {optimum[1]:.2f}={optimum_score:.3f}$",
        transform=ax.transAxes,
        fontsize=7.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    ax.annotate(
        r"$y^\star$",
        xy=optimum,
        xytext=(-19, -14),
        textcoords="offset points",
        fontsize=8,
        color=optimum_color,
    )
    ax.set_title(r"(a) Objective on $P$", pad=5)

    ax = axes[1]
    for coordinate in points[:, 0]:
        ax.axvline(coordinate, color="#E1E1E1", linewidth=0.6, zorder=0)
    for coordinate in points[:, 1]:
        ax.axhline(coordinate, color="#E1E1E1", linewidth=0.6, zorder=0)
    ax.scatter(
        grid[:, 0],
        grid[:, 1],
        s=14,
        facecolor="white",
        edgecolor=grid_color,
        linewidth=0.65,
        zorder=1,
    )
    ax.scatter(points[:, 0], points[:, 1], s=28, color=point_color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(*optimum, s=94, marker="*", color=optimum_color, edgecolor="white", linewidth=0.7, zorder=5)
    ax.text(
        0.04,
        0.05,
        rf"$|\Gamma(P)|=n^2={len(grid)}$ candidates",
        transform=ax.transAxes,
        fontsize=7.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    ax.set_title(r"(b) Full coordinate grid $\Gamma(P)$", pad=5)

    ax = axes[2]
    ax.scatter(
        critical[:, 0],
        critical[:, 1],
        s=34,
        marker="D",
        color=critical_color,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )
    ax.scatter(points[:, 0], points[:, 1], s=17, color=point_color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(*optimum, s=94, marker="*", color=optimum_color, edgecolor="white", linewidth=0.7, zorder=5)
    ax.text(
        0.04,
        0.05,
        rf"$y(S)=\max_{{p\in S}}p$: {len(critical)} candidates",
        transform=ax.transAxes,
        fontsize=7.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    ax.set_title(r"(c) Critical corners, $1\leq |S|\leq 2$", pad=5)

    ax = axes[3]
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=20,
        color=point_color,
        alpha=0.22,
        edgecolor="none",
        zorder=1,
    )
    for corner in q_pair_corners:
        ax.add_patch(
            Rectangle(
                (0, 0),
                corner[0],
                corner[1],
                fill=False,
                edgecolor=critical_color,
                linewidth=0.65,
                linestyle=(0, (3, 3)),
                alpha=0.48,
                zorder=0,
            )
        )
    ax.scatter(
        q_pair_corners[:, 0],
        q_pair_corners[:, 1],
        s=39,
        marker="D",
        color=critical_color,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.scatter(
        q_points[:, 0],
        q_points[:, 1],
        s=36,
        color=selected_color,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    for q_index, point in enumerate(q_points, start=1):
        ax.annotate(
            rf"$q_{q_index}$",
            xy=point,
            xytext=(-10, -12),
            textcoords="offset points",
            fontsize=8,
            color=selected_color,
        )
    if np.any(np.all(np.isclose(q_critical, optimum), axis=1)):
        ax.scatter(*optimum, s=94, marker="*", color=optimum_color, edgecolor="white", linewidth=0.7, zorder=5)
    ax.text(
        0.04,
        0.05,
        rf"$|Q|=3$: {len(q_points)} singleton $+$ {len(q_pair_corners)} pair corners",
        transform=ax.transAxes,
        fontsize=7.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    ax.set_title(r"(d) Critical corners spanned by $Q$", pad=5)

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=point_color, markersize=5, label=r"Point $p_i$"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=grid_color,
            markersize=4,
            label="Grid candidate",
        ),
        Line2D([], [], marker="D", linestyle="", color=critical_color, markersize=5, label="Critical corner"),
        Line2D([], [], marker="o", linestyle="", color=selected_color, markersize=5, label=r"Selected point $q_i$"),
        Line2D([], [], marker="*", linestyle="", color=optimum_color, markersize=8, label=r"Maximizer $y^\star$"),
    ]
    fig.legend(
        handles=legend_handles,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        columnspacing=1.15,
        handletextpad=0.5,
    )
    save_figure(fig, output_dir, "discrepancy_search_space_schematic")


def plot_main_regret(rows: list[Row], output_dir: Path) -> None:
    dataset = "uniform_d3_n64_test"
    methods = [
        "random",
        "local",
        "flat_mlp",
        "depth0_no_graph",
        "depth1_rank_knn_linf",
        "rank_knn_linf",
        "rank_adjacency",
    ]
    subset = [row for row in rows if row.dataset_label == dataset]
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    for method in methods:
        line_for_method(ax, subset, method, "regret")
    format_curve_axes(ax, r"Mean regret ratio $\delta^-_Q / D^-_P$", 0.35, 1.01)
    ax.set_title(r"In-distribution performance ($d=3,\ n=64$)", pad=6)
    ax.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.5,
    )
    save_figure(fig, output_dir, "main_regret_uniform_d3_n64")


def plot_depth_ablation(rows: list[Row], output_dir: Path) -> None:
    methods = ["depth0_no_graph", "depth1_rank_knn_linf", "depth2_rank_knn_linf"]
    fig, axes = plt.subplots(1, 2, figsize=(6.05, 2.55), sharey=True)
    for ax, dataset in zip(axes, ["uniform_d3_n64_test", "uniform_d3_n128_test"]):
        subset = [row for row in rows if row.dataset_label == dataset]
        for method in methods:
            line_for_method(ax, subset, method, "regret")
        format_curve_axes(ax, r"Mean regret ratio", 0.5, 1.01)
        ax.set_title(DATASET_LABELS[dataset], pad=5)
    axes[1].set_ylabel("")
    axes[0].legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(1.02, -0.24),
        columnspacing=1.1,
        handlelength=1.8,
        handletextpad=0.5,
    )
    save_figure(fig, output_dir, "depth_ablation_regret")


def plot_coordinate_equivariance_ablation(rows: list[Row], output_dir: Path) -> None:
    methods = [
        ("rank_knn_linf", "Coordinate-equivariant"),
        ("coord_unshared_rank_knn_linf", "Coordinate-unshared"),
    ]
    datasets = [
        "uniform_d3_n64_test",
        "uniform_d3_n128_test",
        "jittered_grid4_d3_n64_test",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.55), sharey=True)
    for ax, dataset in zip(axes, datasets):
        subset = [row for row in rows if row.dataset_label == dataset]
        for method, label in methods:
            line_for_method(ax, subset, method, "regret", label=label)
        format_curve_axes(ax, r"Mean regret ratio", 0.72, 1.01)
        ax.set_title(DATASET_LABELS[dataset], pad=5)
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(1.58, -0.24),
        columnspacing=1.1,
        handlelength=1.8,
        handletextpad=0.5,
    )
    save_figure(fig, output_dir, "coordinate_equivariance_ablation_regret")


def plot_transfer_k8(rows: list[Row], output_dir: Path) -> None:
    lookup = by_key(rows)
    datasets = [
        "uniform_d3_n64_test",
        "uniform_d3_n128_test",
        "uniform_d4_n64_test",
        "jittered_grid4_d3_n64_test",
    ]
    methods = ["random", "local", "depth0_no_graph", "rank_knn_linf", "rank_adjacency", "knn_euclidean"]
    width = 0.12
    x_positions = list(range(len(datasets)))
    fig, ax = plt.subplots(figsize=(6.25, 2.7))
    for method_index, method in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2) * width
        values = [lookup[(method, dataset, 8)].regret for dataset in datasets]
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            label=METHOD_LABELS[method],
        )
    ax.set_ylabel(r"Mean regret ratio at $K=8$")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([DATASET_SHORT_LABELS[dataset] for dataset in datasets])
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y")
    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.1,
        handlelength=1.3,
        handletextpad=0.5,
    )
    save_figure(fig, output_dir, "transfer_regret_k8")


def plot_connectivity_k8(rows: list[Row], output_dir: Path) -> None:
    lookup = by_key(rows)
    datasets = ["uniform_d3_n64_test", "uniform_d4_n64_test", "jittered_grid4_d3_n64_test"]
    methods = ["knn_euclidean", "knn_linf", "rank_knn_euclidean", "rank_knn_linf", "rank_adjacency"]
    width = 0.14
    x_positions = list(range(len(datasets)))
    fig, ax = plt.subplots(figsize=(5.9, 2.65))
    for method_index, method in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2) * width
        values = [lookup[(method, dataset, 8)].regret for dataset in datasets]
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            label=METHOD_LABELS[method],
        )
    ax.set_ylabel(r"Mean regret ratio at $K=8$")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([DATASET_SHORT_LABELS[dataset] for dataset in datasets])
    ax.set_ylim(0.88, 0.985)
    ax.grid(axis="y")
    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.5,
    )
    save_figure(fig, output_dir, "connectivity_ablation_k8")


def plot_recall_regret_scatter(rows: list[Row], output_dir: Path) -> None:
    dataset = "uniform_d3_n64_test"
    methods = [
        "random",
        "local",
        "flat_mlp",
        "depth0_no_graph",
        "rank_knn_linf",
        "rank_adjacency",
        "knn_euclidean",
    ]
    lookup = by_key(rows)
    fig, ax = plt.subplots(figsize=(4.65, 2.85))
    for method in methods:
        row = lookup[(method, dataset, 8)]
        edge_kwargs = {} if MARKERS[method] == "x" else {"edgecolor": "white"}
        ax.scatter(
            row.recall,
            row.regret,
            s=46,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=0.6,
            zorder=3,
            label=METHOD_LABELS[method],
            **edge_kwargs,
        )
    ax.set_xlabel(r"Support recall at $K=8$")
    ax.set_ylabel(r"Mean regret ratio at $K=8$")
    ax.set_xlim(0.1, 0.86)
    ax.set_ylim(0.55, 1.0)
    ax.grid(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        handletextpad=0.5,
    )
    save_figure(fig, output_dir, "recall_regret_scatter_k8")


def write_readme(output_dir: Path) -> None:
    entries = [
        (
            "discrepancy_search_space_schematic",
            "Geometric view of the minus-discrepancy objective, full grid, critical corners, and restriction to a three-point subset Q.",
        ),
        ("main_regret_uniform_d3_n64", "Main in-distribution regret curves over K."),
        ("transfer_regret_k8", "Transfer comparison at K=8 across all test datasets."),
        ("depth_ablation_regret", "Message-passing depth ablation on in-distribution and larger-n tests."),
        (
            "coordinate_equivariance_ablation_regret",
            "Coordinate-equivariant versus coordinate-unshared models on compatible test datasets.",
        ),
        ("connectivity_ablation_k8", "Graph connectivity ablation at K=8."),
        ("recall_regret_scatter_k8", "Recall-vs-regret diagnostic at K=8."),
    ]
    lines = [
        "# Paper Plots",
        "",
        "Each plot is written as PDF, SVG, and 320 DPI PNG. Use PDF for LaTeX unless the venue requires another format.",
        "",
    ]
    for stem, description in entries:
        lines.append(f"- `{stem}.pdf`: {description}")
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="results/paper/metrics.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="results/paper/figures",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    rows = load_rows(args.metrics)
    plot_discrepancy_search_space(args.output_dir)
    plot_main_regret(rows, args.output_dir)
    plot_transfer_k8(rows, args.output_dir)
    plot_depth_ablation(rows, args.output_dir)
    plot_coordinate_equivariance_ablation(rows, args.output_dir)
    plot_connectivity_k8(rows, args.output_dir)
    plot_recall_regret_scatter(rows, args.output_dir)
    write_readme(args.output_dir)
    print(f"wrote paper plots to {args.output_dir}")


if __name__ == "__main__":
    main()
