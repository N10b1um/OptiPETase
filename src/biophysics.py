import tempfile
from pathlib import Path

import numpy as np
from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, Simulation
from openmm.unit import kelvin, picosecond
from pdbfixer import PDBFixer

from src.structure import ONE_TO_THREE, parse_pdb_all_atom_coordinates

VOLUMES: dict[str, float] = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

HYDROPHOBICITY_KD: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

CHARGES: dict[str, float] = {
    "A": 0.0, "R": 1.0, "N": 0.0, "D": -1.0, "C": 0.0,
    "Q": 0.0, "E": -1.0, "G": 0.0, "H": 0.1, "I": 0.0,
    "L": 0.0, "K": 1.0, "M": 0.0, "F": 0.0, "P": 0.0,
    "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}

AGGREGATION_INDICES: dict[str, float] = {
    "A": 0.31, "C": 1.54, "D": -3.0, "E": -2.3, "F": 1.79,
    "G": -0.13, "H": -0.13, "I": 1.8, "K": -3.4, "L": 1.7,
    "M": 1.23, "N": -1.0, "P": -0.4, "Q": -0.23, "R": -2.5,
    "S": -0.26, "T": -0.05, "V": 1.22, "W": 2.25, "Y": 0.96,
}


def calculate_local_packing_density(coords: np.ndarray, radius: float = 10.0) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError("Coordinates must be a non-empty (N, 3) array.")

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    density = np.sum((distances <= radius) & (distances > 1e-5), axis=-1)
    return density.astype(np.float32)


def calculate_surface_aggregation_propensity(seq: str, coords: np.ndarray) -> np.ndarray:
    if len(seq) != len(coords):
        raise ValueError("Mismatch between sequence length and coordinates length.")

    packing_density = calculate_local_packing_density(coords)
    exposure = 1.0 / (1.0 + packing_density)
    propensities = np.array([AGGREGATION_INDICES[aa] for aa in seq], dtype=np.float32)
    return exposure * propensities


def validate_mutation_safety(
    wt_seq: str,
    mut_seq: str,
    pos_idx: int,
    coords: np.ndarray,
    locked_indices: set[int] | None = None,
    packing_density: np.ndarray | None = None,
    aggregation_threshold: float = 1.0,
    mean_density: float | None = None,
    std_density: float | None = None,
) -> tuple[bool, str]:
    if len(wt_seq) != len(mut_seq) or not (0 <= pos_idx < len(wt_seq)):
        raise ValueError("Invalid sequence length or position index.")

    if locked_indices is not None and pos_idx in locked_indices:
        return False, "Catalytic core residue lock"

    mut_aa = mut_seq[pos_idx]
    if packing_density is None:
        packing_density = calculate_local_packing_density(coords)
    if mean_density is None:
        mean_density = float(np.mean(packing_density))
    if std_density is None:
        std_density = float(np.std(packing_density))

    local_density = packing_density[pos_idx]

    if mut_aa == "C" and local_density < mean_density:
        return False, "Exposed free cysteine risk"

    if mut_aa in {"F", "L", "I", "V", "W", "Y"} and local_density < (mean_density - 0.5 * std_density):
        return False, "Exposed hydrophobic patch creation"

    exposure = 1.0 / (1.0 + local_density)
    delta_aggregation = exposure * (AGGREGATION_INDICES[mut_aa] - AGGREGATION_INDICES[wt_seq[pos_idx]])
    if delta_aggregation > aggregation_threshold:
        return False, "High surface aggregation propensity increase"

    return True, "Passed"


def _calc_dist(
    p1: np.ndarray | None, p2: np.ndarray | None, default: float = 10.0
) -> float:
    if p1 is not None and p2 is not None:
        return float(np.linalg.norm(p1 - p2))
    return default


def _get_atom_coord(coords: dict[tuple[str, int, str], np.ndarray], res_id: int, atom_name: str) -> np.ndarray | None:
    for (_, r_id, a_name), pos in coords.items():
        if r_id == res_id and a_name == atom_name:
            return pos
    return None


def _extract_active_site_distances(
    atom_coords: dict[tuple[str, int, str], np.ndarray],
    chain: str = "A",
    triad_ids: list[int] | None = None,
    oxyanion_ids: list[int] | None = None,
) -> dict[str, float]:
    triad_ids = triad_ids or [160, 206, 237]
    oxyanion_ids = oxyanion_ids or [87, 161]

    ser_id, asp_id, his_id = triad_ids
    oxy1_id, oxy2_id = oxyanion_ids

    og = _get_atom_coord(atom_coords, ser_id, "OG")
    ne2 = _get_atom_coord(atom_coords, his_id, "NE2")
    d1 = _calc_dist(og, ne2)

    nd1 = _get_atom_coord(atom_coords, his_id, "ND1")
    od1 = _get_atom_coord(atom_coords, asp_id, "OD1")
    od2 = _get_atom_coord(atom_coords, asp_id, "OD2")
    d2 = min(_calc_dist(nd1, od1), _calc_dist(nd1, od2))

    ser_o = _get_atom_coord(atom_coords, ser_id, "O")
    d3 = _calc_dist(ser_o, _get_atom_coord(atom_coords, oxy1_id, "N"))
    d4 = _calc_dist(ser_o, _get_atom_coord(atom_coords, oxy2_id, "N"))

    return {"d1": d1, "d2": d2, "d3": d3, "d4": d4}


def _compute_triad_rmsd(
    wt_coords: dict[tuple[str, int, str], np.ndarray],
    mut_coords: dict[tuple[str, int, str], np.ndarray],
    triad_ids: list[int],
    chain: str = "A",
) -> float:
    sq_diffs = [
        np.sum((wt_coords[key] - mut_coords[key]) ** 2)
        for key in wt_coords
        if key[0] == chain and key[1] in triad_ids and key in mut_coords
    ]
    return float(np.sqrt(np.mean(sq_diffs))) if sq_diffs else 0.0


def _parse_mutation_strings(
    mutations: list[str] | dict[str, list[str]], default_chain: str
) -> dict[str, list[str]]:
    if isinstance(mutations, dict):
        return mutations

    formatted = []
    for mut_str in mutations:
        clean_mut = mut_str.strip()
        if len(clean_mut) >= 3:
            wt_3 = ONE_TO_THREE.get(clean_mut[0], "ALA")
            mut_3 = ONE_TO_THREE.get(clean_mut[-1], "ALA")
            pos_num = clean_mut[1:-1]
            formatted.append(f"{wt_3}-{pos_num}-{mut_3}")
    return {default_chain: formatted}


def _is_active_site_mutated(mut_dict: dict[str, list[str]], triad_ids: list[int]) -> bool:
    for mut_list in mut_dict.values():
        for mut in mut_list:
            parts = mut.split("-")
            if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) in triad_ids:
                return True
    return False


def _apply_mutations_to_fixer(fixer: PDBFixer, mut_dict: dict[str, list[str]]) -> None:
    for ch, mut_items in mut_dict.items():
        if not mut_items:
            continue
        chain_residues = {
            res.id: res.name
            for chain in fixer.topology.chains()
            if chain.id == ch
            for res in chain.residues()
        }
        sanitized_muts = []
        for mut in mut_items:
            parts = mut.split("-")
            if len(parts) == 3:
                orig, res_id, new = parts
                actual_orig = chain_residues.get(str(res_id), orig)
                sanitized_muts.append(f"{actual_orig}-{res_id}-{new}")
            else:
                sanitized_muts.append(mut)
        fixer.applyMutations(sanitized_muts, ch)


def _minimize_openmm_structure(
    topology, positions, ff_files: list[str], output_path: Path, max_iterations: int = 200
) -> dict[tuple[str, int, str], np.ndarray]:
    forcefield = ForceField(*ff_files)
    system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picosecond)

    try:
        platform = Platform.getPlatformByName("CUDA")
        properties = {"Precision": "mixed"}
        simulation = Simulation(topology, system, integrator, platform, properties)
    except Exception:  # noqa: BLE001
        simulation = Simulation(topology, system, integrator)

    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(maxIterations=max_iterations)

    state = simulation.context.getState(getPositions=True)
    with open(output_path, "w", encoding="utf-8") as f:
        PDBFile.writeFile(topology, state.getPositions(), f)

    return parse_pdb_all_atom_coordinates(output_path)


def evaluate_mutant_catalytic_geometry(
    wt_pdb_path: str | Path,
    mutations: list[str] | dict[str, list[str]],
    ff_files: list[str] | None = None,
    triad_pdb_ids: list[int] | None = None,
    oxyanion_pdb_ids: list[int] | None = None,
    ph: float = 7.4,
) -> dict[str, float]:
    ff_files = ff_files or ["amber14-all.xml", "implicit/obc2.xml"]
    triad_ids = triad_pdb_ids or [160, 206, 237]
    oxyanion_ids = oxyanion_pdb_ids or [87, 161]

    wt_coords = parse_pdb_all_atom_coordinates(wt_pdb_path)
    chain_id = next(iter(wt_coords.keys()))[0] if wt_coords else "A"

    wt_dists = _extract_active_site_distances(wt_coords, chain_id, triad_ids, oxyanion_ids)
    mut_dict = _parse_mutation_strings(mutations, chain_id)
    is_catalytic_dead = _is_active_site_mutated(mut_dict, triad_ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        fixer = PDBFixer(filename=str(wt_pdb_path))
        fixer.removeHeterogens(keepWater=False)
        _apply_mutations_to_fixer(fixer, mut_dict)
        fixer.findMissingResidues()

        chains = list(fixer.topology.chains())
        for key in list(fixer.missingResidues.keys()):
            chain_idx, res_idx = key
            chain = chains[chain_idx]
            if res_idx == 0 or res_idx == len(list(chain.residues())):
                del fixer.missingResidues[key]

        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)

        min_pdb = Path(tmpdir) / "mutant_min.pdb"
        mut_coords = _minimize_openmm_structure(
            fixer.topology, fixer.positions, ff_files, min_pdb
        )

        mut_dists = _extract_active_site_distances(mut_coords, chain_id, triad_ids, oxyanion_ids)
        triad_rmsd = _compute_triad_rmsd(wt_coords, mut_coords, triad_ids, chain_id)

    delta_d1 = mut_dists["d1"] - wt_dists["d1"]
    delta_d2 = mut_dists["d2"] - wt_dists["d2"]
    catalytic_distortion = float(np.sqrt(delta_d1**2 + delta_d2**2)) + triad_rmsd

    if is_catalytic_dead:
        catalytic_distortion += 10.0

    return {
        "d1_wt": wt_dists["d1"],
        "d2_wt": wt_dists["d2"],
        "d3_wt": wt_dists["d3"],
        "d4_wt": wt_dists["d4"],
        "d1_mut": mut_dists["d1"],
        "d2_mut": mut_dists["d2"],
        "d3_mut": mut_dists["d3"],
        "d4_mut": mut_dists["d4"],
        "triad_rmsd": triad_rmsd,
        "catalytic_distortion": catalytic_distortion,
    }