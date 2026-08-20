from typing import Any

import numpy as np

from src.biophysics import CHARGES, HYDROPHOBICITY_KD, VOLUMES


def dominates(cand_a: dict[str, Any], cand_b: dict[str, Any]) -> bool:
    fit_a = cand_a.get("esm2_fitness_score", 0.0)
    fit_b = cand_b.get("esm2_fitness_score", 0.0)
    dist_a = cand_a.get("catalytic_distortion", 0.0)
    dist_b = cand_b.get("catalytic_distortion", 0.0)

    ge_fit = fit_a >= fit_b
    le_dist = dist_a <= dist_b
    gt_fit = fit_a > fit_b
    lt_dist = dist_a < dist_b

    return (ge_fit and le_dist) and (gt_fit or lt_dist)


def compute_pareto_fronts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    n = len(candidates)
    domination_count = [0] * n
    dominated_solutions: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(candidates[p], candidates[q]):
                dominated_solutions[p].append(q)
            elif dominates(candidates[q], candidates[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            candidates[p]["pareto_rank"] = 1
            fronts[0].append(p)

    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front: list[int] = []
        for p in fronts[current_front]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    candidates[q]["pareto_rank"] = current_front + 2
                    next_front.append(q)
        current_front += 1
        fronts.append(next_front)

    return candidates


class CandidateSelector:
    def __init__(self, n_clusters: int = 5, seed: int = 42) -> None:
        if n_clusters < 1:
            raise ValueError("The number of clusters must be at least 1.")
        self.n_clusters = n_clusters
        self.seed = seed

    def extract_mutation_features(
        self, candidates: list[dict[str, Any]], coords_wt: np.ndarray | None = None
    ) -> np.ndarray:
        if not candidates:
            return np.empty((0, 6), dtype=np.float32)

        features = []
        for cand in candidates:
            wt_aa = cand.get("wt_aa", "A")
            mut_aa = cand.get("mut_aa", "A")
            pos_idx = cand.get("pos_idx", 0)

            delta_vol = VOLUMES.get(mut_aa, 100.0) - VOLUMES.get(wt_aa, 100.0)
            delta_hydro = HYDROPHOBICITY_KD.get(mut_aa, 0.0) - HYDROPHOBICITY_KD.get(wt_aa, 0.0)
            delta_charge = CHARGES.get(mut_aa, 0.0) - CHARGES.get(wt_aa, 0.0)

            if coords_wt is not None and 0 <= pos_idx < len(coords_wt):
                coord = coords_wt[pos_idx]
            else:
                coord = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            features.append([
                delta_vol / 100.0,
                delta_hydro / 5.0,
                delta_charge,
                coord[0] / 50.0,
                coord[1] / 50.0,
                coord[2] / 50.0,
            ])
        return np.array(features, dtype=np.float32)

    def select_diverse_top_candidates(
        self, candidates: list[dict[str, Any]], coords_wt: np.ndarray | None = None
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        ranked = compute_pareto_fronts(candidates)

        sorted_candidates = sorted(
            ranked,
            key=lambda x: (
                x.get("pareto_rank", 999),
                -x.get("esm2_fitness_score", 0.0),
            ),
        )

        selected: list[dict[str, Any]] = []
        seen_positions = set()

        for cand in sorted_candidates:
            pos = cand.get("pdb_idx")
            if pos not in seen_positions:
                seen_positions.add(pos)
                selected.append(cand)

            if len(selected) == self.n_clusters:
                break

        if len(selected) < self.n_clusters:
            selected_mut_strs = {c.get("mutation_str") for c in selected}
            for cand in sorted_candidates:
                m_str = cand.get("mutation_str")
                if m_str not in selected_mut_strs:
                    selected.append(cand)
                    selected_mut_strs.add(m_str)
                if len(selected) == self.n_clusters:
                    break

        return selected