import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def plot_pareto_frontier(
    all_candidates: list[dict[str, Any]],
    pareto_rank_1: list[dict[str, Any]],
    benchmark_variants: list[dict[str, Any]] | None,
    selected_top5: list[dict[str, Any]],
    save_path: str | Path,
) -> None:
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    if all_candidates:
        x_all = [c.get("esm2_fitness_score", 0.0) for c in all_candidates]
        y_all = [-c.get("catalytic_distortion", 0.0) for c in all_candidates]
        ax1.scatter(
            x_all,
            y_all,
            c="#cbd5e1",
            alpha=0.6,
            s=35,
            label="Candidate Pool",
            edgecolors="none",
        )

    if pareto_rank_1:
        sorted_rank1 = sorted(pareto_rank_1, key=lambda c: c.get("esm2_fitness_score", 0.0))
        x_p1 = [c.get("esm2_fitness_score", 0.0) for c in sorted_rank1]
        y_p1 = [-c.get("catalytic_distortion", 0.0) for c in sorted_rank1]
        ax1.plot(x_p1, y_p1, color="#ef4444", linestyle="--", linewidth=2.0, zorder=3)
        ax1.scatter(x_p1, y_p1, color="#ef4444", s=55, label="Pareto Frontier (Rank 1)", zorder=4)

    if selected_top5:
        sorted_selected = sorted(selected_top5, key=lambda c: c.get("esm2_fitness_score", 0.0))
        offsets = [(0, 14), (0, -22), (14, 12), (-14, -18), (0, 14)]

        for idx, cand in enumerate(sorted_selected):
            x_s = cand.get("esm2_fitness_score", 0.0)
            y_s = -cand.get("catalytic_distortion", 0.0)
            
            wt = cand.get("wt_aa", "")
            pos = cand.get("pdb_idx", "")
            mut = cand.get("mut_aa", "")
            label_name = cand.get("mutation_str") or f"{wt}{pos}{mut}"

            ax1.scatter(
                x_s,
                y_s,
                marker="o",
                color="#06b6d4",
                edgecolors="#083344",
                linewidth=1.5,
                s=150,
                zorder=7,
                label="Selected Top Diverse" if idx == 0 else "",
            )

            offset = offsets[idx % len(offsets)]
            ax1.annotate(
                label_name,
                (x_s, y_s),
                textcoords="offset points",
                xytext=offset,
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "white",
                    "alpha": 0.95,
                    "ec": "#083344",
                    "lw": 0.8,
                },
                zorder=8,
            )

    x_max = max([c.get("esm2_fitness_score", 0.0) for c in all_candidates] + [3.0])
    ax1.set_xlim(-0.2, x_max + 0.3)
    ax1.set_ylim(-0.045, 0.015)

    ax1.text(
        0.03,
        0.05,
        "Disruptive outliers (ΔGeom > 0.05 Å) & lethal controls\n(S160A, D206A, H237A at -10.0 Å) penalized off-scale",
        transform=ax1.transAxes,
        fontsize=8.5,
        fontstyle="italic",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#f8fafc", "alpha": 0.9, "ec": "#94a3b8", "lw": 0.6},
        zorder=5,
    )

    ax1.set_xlabel("Evolutionary Fitness Score (ESM-2 $\\Delta$LLR)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Catalytic Geometry Integrity ($-\\Delta$Geom, $\\AA$)", fontsize=11, fontweight="bold")
    ax1.set_title("A. De Novo Mutation Discovery (Pareto Frontier)", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.5)

    if benchmark_variants:
        nature_vars = [
            b for b in benchmark_variants if b.get("variant_name") not in ["S160A", "H237A", "D206A"]
        ]
        if nature_vars:
            x_b = np.array([b.get("activity_50C", 0.0) for b in nature_vars], dtype=np.float32)
            y_b = np.array([b.get("esm2_fitness_score", 0.0) for b in nature_vars], dtype=np.float32)
            names = [b.get("variant_name", "") for b in nature_vars]

            if len(x_b) > 1:
                slope, intercept = np.polyfit(x_b, y_b, 1)
                x_line = np.linspace(min(x_b) - 0.1, max(x_b) + 0.1, 100)
                y_line = slope * x_line + intercept
                ax2.plot(x_line, y_line, color="#3b82f6", linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)

                rho_res = spearmanr(y_b, x_b)
                rho_val = float(rho_res.statistic) if hasattr(rho_res, "statistic") else float(rho_res[0])
                p_val = float(rho_res.pvalue) if hasattr(rho_res, "pvalue") else float(rho_res[1])

                insight_text = (
                    f"Spearman $\\rho = {rho_val:.2f}$ ($p = {p_val:.3e}$)\n"
                    r"$\bf{Additive\ Zero\text{-}Shot\ Limitation:}$" "\n"
                    "Single-point models evaluate substitutions independently.\n"
                    "Engineered salt bridges (e.g. S121E ↔ N233K in FAST-PETase)\n"
                    "receive cumulative penalties despite high combined activity."
                )
                ax2.text(
                    0.48,
                    0.72,
                    insight_text,
                    transform=ax2.transAxes,
                    fontsize=8.5,
                    bbox={"boxstyle": "round,pad=0.4", "fc": "#fef2f2", "alpha": 0.95, "ec": "#ef4444", "lw": 0.8},
                    zorder=7,
                )

            ax2.scatter(
                x_b,
                y_b,
                marker="*",
                color="#eab308",
                edgecolors="#713f12",
                linewidth=1.2,
                s=240,
                zorder=5,
                label="Engineered Variants (Nature 2022)",
            )

            b_offsets = [(0, 10), (0, -18), (10, 10), (-14, -16), (0, 12), (0, -18), (-12, 10), (10, -14)]
            for i, name in enumerate(names):
                offset = b_offsets[i % len(b_offsets)]
                ax2.annotate(
                    name,
                    (x_b[i], y_b[i]),
                    textcoords="offset points",
                    xytext=offset,
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "fc": "white",
                        "alpha": 0.9,
                        "ec": "#94a3b8",
                        "lw": 0.5,
                    },
                    zorder=6,
                )

            ax2.set_xlabel("Experimental Activity at 50°C (Rel. to WT)", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Predicted Evolutionary Fitness (Additive ESM-2 $\\Delta$LLR)", fontsize=11, fontweight="bold")
            ax2.set_title("B. Epistasis Bottleneck: Zero-Shot ESM-2 on Multi-Mutants", fontsize=12, fontweight="bold", pad=12)
            ax2.grid(True, linestyle="--", alpha=0.4)
            ax2.legend(loc="lower left", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(str(save_p), dpi=300)
    plt.close(fig)


def generate_pymol_session(
    pdb_id: str,
    mapped_pdb_path: str | Path,
    selected_candidates: list[dict[str, Any]],
    active_site_residues: list[int],
    output_script_path: str | Path,
    plot_dir: str | Path,
) -> None:
    out_p = Path(output_script_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plot_p = Path(plot_dir)
    plot_p.parent.mkdir(parents=True, exist_ok=True)

    png_cartoon_path = plot_p / f"{pdb_id}_pareto_selected.png"
    png_surface_path = plot_p / f"{pdb_id}_pareto_selected_surface.png"

    commands: list[str] = [
        f"load {mapped_pdb_path!s}, {pdb_id}",
        f"hide everything, {pdb_id}",
        f"show cartoon, {pdb_id}",
        f"spectrum b, blue_white_red, {pdb_id}, minimum=-2.0, maximum=2.0",
    ]

    if active_site_residues:
        res_selection_string = "+".join(str(r) for r in active_site_residues)
        commands.append(f"select active_site, {pdb_id} and resi {res_selection_string}")
        commands.append("show sticks, active_site")
        commands.append("color yellow, active_site")

    for cand in selected_candidates:
        pdb_idx = cand.get("pdb_idx")
        if pdb_idx is None:
            pos_idx = cand.get("pos_idx")
            pdb_idx = pos_idx + 1 if pos_idx is not None else 1

        wt_res = cand.get("wt_aa", "")
        mut_res = cand.get("mut_aa", "")
        mut_name = cand.get("mutation_str", f"{wt_res}{pdb_idx}{mut_res}")
        selection_name = f"mut_{pdb_idx}"

        commands.append(f"select {selection_name}, {pdb_id} and resi {pdb_idx}")
        commands.append(f"show sticks, {selection_name}")
        commands.append(f"color green, {selection_name}")
        commands.append(f'label ({selection_name} and name CA), "{mut_name}"')

    commands.extend([
        "bg_color white",
        "set ray_shadow, 1",
        "set label_color, black",
        "set label_size, 14",
        f"zoom {pdb_id}",
        f"png {png_cartoon_path!s}, width=1600, height=1200, dpi=300, ray=1",
        f"show surface, {pdb_id}",
        "set transparency, 0.35",
        f"png {png_surface_path!s}, width=1600, height=1200, dpi=300, ray=1",
    ])

    with open(str(out_p), "w", encoding="utf-8") as f:
        f.writelines(f"{command}\n" for command in commands)

def generate_pymol_script(
    pdb_id: str,
    mapped_pdb_path: str,
    selected_candidates: list[dict[str, Any]],
    active_site_residues: list[int],
    output_script_path: str,
) -> None:
    out_p = Path(output_script_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    commands: list[str] = [
        f"load {mapped_pdb_path}, {pdb_id}",
        f"hide everything, {pdb_id}",
        f"show cartoon, {pdb_id}",
        f"spectrum b, blue_white_red, {pdb_id}, minimum=-2.0, maximum=2.0",
    ]

    if active_site_residues:
        res_selection_string = "+".join(str(r) for r in active_site_residues)
        commands.append(f"select active_site, {pdb_id} and resi {res_selection_string}")
        commands.append("show sticks, active_site")
        commands.append("color yellow, active_site")

    for candidate in selected_candidates:
        pdb_idx = candidate.get("pdb_idx")
        if pdb_idx is None:
            pos_idx = candidate.get("pos_idx")
            pdb_idx = pos_idx + 1 if pos_idx is not None else candidate.get("wt_idx", 0) + 1

        wt_res = candidate.get("wt_aa") or candidate.get("wt_res", "")
        mut_res = candidate.get("mut_aa") or candidate.get("mut_res", "")
        mut_name = f"{wt_res}{pdb_idx}{mut_res}"
        selection_name = f"mut_{pdb_idx}"

        commands.append(f"select {selection_name}, {pdb_id} and resi {pdb_idx}")
        commands.append(f"show sticks, {selection_name}")
        commands.append(f"color green, {selection_name}")
        commands.append(f'label ({selection_name} and name CA), "{mut_name}"')

    commands.extend([
        "bg_color white",
        "set ray_shadow, 1",
        "set label_color, black",
        f"zoom {pdb_id}",
    ])

    with open(output_script_path, "w", encoding="utf-8") as f:
        f.writelines(f"{command}\n" for command in commands)


def plot_candidate_clusters_pca(
    all_pca_coords: np.ndarray,
    all_cluster_labels: np.ndarray,
    selected_candidates: list[dict[str, Any]],
    save_path: str,
) -> None:
    if all_pca_coords.ndim != 2 or all_pca_coords.shape[1] < 2:
        raise ValueError("all_pca_coords must be a 2D array with at least 2 columns.")
    if len(all_pca_coords) != len(all_cluster_labels):
        raise ValueError("The length of all_pca_coords and all_cluster_labels must match.")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    scatter = ax.scatter(
        all_pca_coords[:, 0],
        all_pca_coords[:, 1],
        c=all_cluster_labels,
        cmap="viridis",
        alpha=0.6,
        s=20,
        edgecolors="none",
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Pareto Rank (1 = Optimal)")

    for candidate in selected_candidates:
        pca_x = candidate.get("pca_x")
        pca_y = candidate.get("pca_y")

        if pca_x is not None and pca_y is not None:
            coords = (pca_x, pca_y)
        else:
            coords = candidate.get("pca_coords")
            if coords is None:
                idx = candidate.get("idx") or candidate.get("index")
                if idx is not None:
                    coords = all_pca_coords[idx]
                else:
                    raise KeyError("Candidate coordinates could not be resolved.")

        pdb_idx = candidate.get("pdb_idx")
        if pdb_idx is None:
            pos_idx = candidate.get("pos_idx")
            pdb_idx = pos_idx + 1 if pos_idx is not None else candidate.get("wt_idx", 0) + 1

        wt_res = candidate.get("wt_aa") or candidate.get("wt_res", "")
        mut_res = candidate.get("mut_aa") or candidate.get("mut_res", "")
        label = f"{wt_res}{pdb_idx}{mut_res}"

        ax.scatter(
            coords[0],
            coords[1],
            marker="*",
            s=160,
            color="red",
            edgecolors="black",
            zorder=5,
        )

        ax.annotate(
            label,
            (coords[0], coords[1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "alpha": 0.85,
                "ec": "gray",
                "lw": 0.5,
            },
            zorder=6,
        )

    ax.set_xlabel("PC1", fontsize=11, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=11, fontweight="bold")
    ax.set_title("Mutation Space PCA and Selected Candidates", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)