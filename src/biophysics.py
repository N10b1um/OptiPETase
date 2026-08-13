from typing import Tuple
import numpy as np


def calculate_local_packing_density(
    coords: np.ndarray, radius: float = 10.0
) -> np.ndarray:
    if not isinstance(coords, np.ndarray):
        raise TypeError("Coordinates must be a NumPy array.")
    
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (L, 3).")
    
    if coords.shape[0] == 0:
        raise ValueError("Coordinates array cannot be empty.")
        
    if not isinstance(radius, (int, float)):
        raise TypeError("Radius must be a float or an integer.")
        
    if radius <= 0.0:
        raise ValueError("Radius must be a positive value.")

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    
    density = np.sum((distances <= radius) & (distances > 1e-5), axis=-1)
    return density


def calculate_surface_aggregation_propensity(
    seq: str, coords: np.ndarray
) -> np.ndarray:
    if not isinstance(seq, str):
        raise TypeError("Sequence must be a string.")
        
    if not isinstance(coords, np.ndarray):
        raise TypeError("Coordinates must be a NumPy array.")

    if len(seq) != coords.shape[0]:
        raise ValueError(
            f"Mismatch between sequence length ({len(seq)}) "
            f"and coordinates length ({coords.shape[0]})."
        )

    aggregation_indices = {
        'A': 0.31, 'C': 1.54, 'D': -3.0, 'E': -2.3, 'F': 1.79,
        'G': -0.13, 'H': -0.13, 'I': 1.8, 'K': -3.4, 'L': 1.7,
        'M': 1.23, 'N': -1.0, 'P': -0.4, 'Q': -0.23, 'R': -2.5,
        'S': -0.26, 'T': -0.05, 'V': 1.22, 'W': 2.25, 'Y': 0.96
    }

    invalid_residues = [aa for aa in seq if aa not in aggregation_indices]
    if invalid_residues:
        raise ValueError(
            f"Sequence contains invalid amino acid characters: {set(invalid_residues)}"
        )

    packing_density = calculate_local_packing_density(coords)
    exposure = 1.0 / (1.0 + packing_density)
    
    propensities = np.array([aggregation_indices[aa] for aa in seq])
    surface_aggregation = exposure * propensities
    
    return surface_aggregation


def validate_mutation_safety(
    wt_seq: str,
    mut_seq: str,
    pos_idx: int,
    coords: np.ndarray,
    shield_mask: np.ndarray,
) -> Tuple[bool, str]:

    if not isinstance(wt_seq, str) or not isinstance(mut_seq, str):
        raise TypeError("Wild-type and mutant sequences must be strings.")
        
    if len(wt_seq) != len(mut_seq):
        raise ValueError("Wild-type and mutant sequences must have equal length.")
        
    if not (0 <= pos_idx < len(wt_seq)):
        raise ValueError(
            f"Mutation position index {pos_idx} is out of bounds for sequence of length {len(wt_seq)}."
        )

    if not isinstance(coords, np.ndarray):
        raise TypeError("Coordinates must be a NumPy array.")
        
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != len(wt_seq):
        raise ValueError(
            f"Coordinates must have shape ({len(wt_seq)}, 3)."
        )

    if not isinstance(shield_mask, np.ndarray):
        raise TypeError("Shield mask must be a NumPy array.")
        
    if shield_mask.dtype != bool:
        raise TypeError("Shield mask must be a boolean NumPy array.")
        
    if shield_mask.shape != (len(wt_seq),):
        raise ValueError(
            f"Shield mask must have shape ({len(wt_seq)},)."
        )

    if shield_mask[pos_idx]:
        return False, "Active site shield violation"

    mut_aa = mut_seq[pos_idx]
    
    valid_amino_acids = {
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
    }
    if mut_aa not in valid_amino_acids:
        raise ValueError(f"Invalid mutant amino acid residue: {mut_aa}")

    packing_density = calculate_local_packing_density(coords)
    mean_density = np.mean(packing_density)
    local_density = packing_density[pos_idx]

    if mut_aa == 'C':
        if local_density < mean_density:
            return False, "Exposed free cysteine risk"

    hydrophobic_residues = {'F', 'L', 'I', 'V', 'W', 'Y'}
    if mut_aa in hydrophobic_residues:
        std_density = np.std(packing_density)
        threshold = mean_density - 0.5 * std_density
        if local_density < threshold:
            return False, "Exposed hydrophobic patch creation"

    return True, "Passed"