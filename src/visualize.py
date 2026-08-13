import os
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def plot_candidate_clusters_pca(
    all_pca_coords: np.ndarray,
    all_cluster_labels: np.ndarray,
    selected_candidates: List[Dict[str, Any]],
    save_path: str
) -> None:
    try:
        if all_pca_coords.ndim != 2 or all_pca_coords.shape[1] < 2:
            raise ValueError(
                "all_pca_coords must be a 2D array with at least 2 columns."
            )
        if len(all_pca_coords) != len(all_cluster_labels):
            raise ValueError(
                "The length of all_pca_coords and all_cluster_labels must match."
            )

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        scatter = ax.scatter(
            all_pca_coords[:, 0],
            all_pca_coords[:, 1],
            c=all_cluster_labels,
            cmap="viridis",
            alpha=0.6,
            s=15,
            edgecolors="none"
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Predicted Stability (dDG)")

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
                        raise KeyError(
                            "Candidate coordinates could not be resolved. "
                            "Must contain 'pca_x'/'pca_y', 'pca_coords', or candidate list index."
                        )

            pdb_idx = candidate.get("pdb_idx")
            if pdb_idx is None:
                pos_idx = candidate.get("pos_idx")
                if pos_idx is not None:
                    pdb_idx = pos_idx + 1
                else:
                    pdb_idx = candidate.get("wt_idx", 0) + 1

            wt_res = candidate.get("wt_aa") or candidate.get("wt_res", "")
            mut_res = candidate.get("mut_aa") or candidate.get("mut_res", "")
            label = f"{wt_res}{pdb_idx}{mut_res}"

            ax.scatter(
                coords[0],
                coords[1],
                marker="*",
                s=150,
                color="red",
                edgecolors="black",
                zorder=5
            )

            ax.annotate(
                label,
                (coords[0], coords[1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    alpha=0.8,
                    ec="gray",
                    lw=0.5
                ),
                zorder=6
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Mutation Space PCA and Selected Candidates")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    except Exception as e:
        raise IOError(f"Failed to generate or save PCA plot: {e}") from e


def generate_pymol_script(
    pdb_id: str,
    mapped_pdb_path: str,
    selected_candidates: List[Dict[str, Any]],
    active_site_residues: List[int],
    output_script_path: str
) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_script_path)), exist_ok=True)

        commands = []
        commands.append(f"load {mapped_pdb_path}, {pdb_id}")
        commands.append(f"hide everything, {pdb_id}")
        commands.append(f"show cartoon, {pdb_id}")
        commands.append(f"spectrum b, blue_white_red, {pdb_id}, minimum=-2.0, maximum=2.0")

        if active_site_residues:
            res_selection_string = "+".join(str(r) for r in active_site_residues)
            commands.append(f"select active_site, {pdb_id} and resi {res_selection_string}")
            commands.append("show sticks, active_site")
            commands.append("color yellow, active_site")

        for candidate in selected_candidates:
            pdb_idx = candidate.get("pdb_idx")
            if pdb_idx is None:
                pos_idx = candidate.get("pos_idx")
                if pos_idx is not None:
                    pdb_idx = pos_idx + 1
                else:
                    pdb_idx = candidate.get("wt_idx", 0) + 1

            wt_res = candidate.get("wt_aa") or candidate.get("wt_res", "")
            mut_res = candidate.get("mut_aa") or candidate.get("mut_res", "")
            mut_name = f"{wt_res}{pdb_idx}{mut_res}"
            selection_name = f"mut_{pdb_idx}"

            commands.append(f"select {selection_name}, {pdb_id} and resi {pdb_idx}")
            commands.append(f"show sticks, {selection_name}")
            commands.append(f"color green, {selection_name}")
            commands.append(f'label ({selection_name} and name CA), "{mut_name}"')

        commands.append("bg_color white")
        commands.append("set ray_shadow, 1")
        commands.append("set label_color, black")
        commands.append(f"zoom {pdb_id}")

        with open(output_script_path, "w", encoding="utf-8") as f:
            for command in commands:
                f.write(f"{command}\n")

    except Exception as e:
        raise IOError(f"Failed to generate PyMOL script at {output_script_path}: {e}") from e