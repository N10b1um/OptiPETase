import copy
from typing import List, Dict, Any, Tuple
import numpy as np


class CandidateSelector:
    _VOLUMES = {
        "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
        "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
        "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
        "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0
    }

    _CHARGES = {
        "R": 1.0, "K": 1.0, "H": 0.5, "D": -1.0, "E": -1.0
    }

    _HYDROPHOBICITY = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9,
        "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3,
        "P": -1.6, "H": -3.2, "Q": -3.5, "N": -3.5, "D": -3.5, "E": -3.5,
        "K": -3.9, "R": -4.5
    }

    def __init__(self, n_clusters: int = 5, seed: int = 42) -> None:
        if n_clusters < 1:
            raise ValueError("The number of clusters must be at least 1.")
        self.n_clusters = n_clusters
        self.seed = seed

    def extract_mutation_features(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(candidates, list):
            raise TypeError("Candidates must be provided as a list.")
        if not candidates:
            raise ValueError("The candidates list cannot be empty.")

        features_list = []
        stabilities_list = []

        for idx, candidate in enumerate(candidates):
            for key in ("wt_aa", "mut_aa", "predicted_stability"):
                if key not in candidate:
                    raise ValueError(
                        f"Candidate dictionary at index {idx} is missing required key '{key}'."
                    )

            wt = candidate["wt_aa"]
            mut = candidate["mut_aa"]
            stability = candidate["predicted_stability"]

            if wt not in self._VOLUMES or mut not in self._VOLUMES:
                raise ValueError(
                    f"Invalid single-letter amino acid code for candidate {idx}: wt='{wt}', mut='{mut}'."
                )

            v_delta = self._VOLUMES[mut] - self._VOLUMES[wt]
            h_delta = self._HYDROPHOBICITY.get(mut, 0.0) - self._HYDROPHOBICITY.get(wt, 0.0)
            c_delta = self._CHARGES.get(mut, 0.0) - self._CHARGES.get(wt, 0.0)

            features_list.append([v_delta, h_delta, c_delta, stability])
            stabilities_list.append(stability)

        return np.array(features_list, dtype=np.float64), np.array(stabilities_list, dtype=np.float64)

    def select_diverse_top_candidates(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        if len(candidates) < self.n_clusters:
            raise ValueError(
                f"Number of candidates ({len(candidates)}) cannot be less "
                f"than the target number of clusters ({self.n_clusters})."
            )

        features, stabilities = self.extract_mutation_features(candidates)
        n_samples = features.shape[0]

        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0.0] = 1.0
        features_scaled = (features - mean) / std

        rng = np.random.RandomState(self.seed)
        initial_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = features_scaled[initial_indices]

        labels = np.zeros(n_samples, dtype=np.int32)
        for _ in range(300):
            distances = np.linalg.norm(
                features_scaled[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                axis=2
            )
            new_labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_points = features_scaled[new_labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = features_scaled[rng.choice(n_samples)]

            if np.allclose(centroids, new_centroids):
                labels = new_labels
                centroids = new_centroids
                break
            centroids = new_centroids
            labels = new_labels

        features_centered = features_scaled - np.mean(features_scaled, axis=0)
        _, _, vt = np.linalg.svd(features_centered, full_matrices=False)

        if vt.shape[0] < 2:
            padding = np.zeros((2 - vt.shape[0], vt.shape[1]), dtype=features_centered.dtype)
            vt = np.concatenate([vt, padding], axis=0)

        pca_coords = np.dot(features_centered, vt[:2].T)

        candidates_copied = [copy.deepcopy(c) for c in candidates]
        for idx, candidate in enumerate(candidates_copied):
            candidate["cluster_id"] = int(labels[idx])
            candidate["pca_x"] = float(pca_coords[idx, 0])
            candidate["pca_y"] = float(pca_coords[idx, 1])

        selected_candidates = []
        selected_indices = set()
        
        for k in range(self.n_clusters):
            cluster_candidates = [
                (idx, c) for idx, c in enumerate(candidates_copied) if c["cluster_id"] == k
            ]
            if cluster_candidates:
                best_idx, best_candidate = min(
                    cluster_candidates,
                    key=lambda x: x[1]["predicted_stability"]
                )
                selected_candidates.append(best_candidate)
                selected_indices.add(best_idx)

        if len(selected_candidates) < self.n_clusters:
            remaining = [
                (idx, c) for idx, c in enumerate(candidates_copied) if idx not in selected_indices
            ]
            remaining.sort(key=lambda x: x[1]["predicted_stability"])
            needed = self.n_clusters - len(selected_candidates)
            for i in range(min(needed, len(remaining))):
                idx, candidate = remaining[i]
                selected_candidates.append(candidate)
                selected_indices.add(idx)

        return selected_candidates, pca_coords, labels