import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "ASX": "B", "GLX": "Z", "XLE": "J", "XAA": "X",
    "HIE": "H", "HID": "H", "HIP": "H",
    "CYX": "C", "CYM": "C",
    "ASH": "D", "GLH": "E", "LYN": "K", "ARN": "R",
}

ONE_TO_THREE: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

VALID_AMINO_ACIDS = set(THREE_TO_ONE.keys())


def download_pdb(pdb_id: str, output_dir: str | Path = "./pdbs") -> Path:
    pdb_id = pdb_id.strip()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdb_path = output_dir / f"{pdb_id.lower()}.pdb"

    if out_pdb_path.exists():
        return out_pdb_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        urllib.request.urlretrieve(url, out_pdb_path)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error downloading {pdb_id}: {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error downloading {pdb_id}: {e.reason}") from e

    return out_pdb_path


def parse_pdb_ca_coordinates_and_seq(
    pdb_path: str | Path, target_chain: str | None = None
) -> tuple[np.ndarray | None, str, dict[int, int]]:
    path = Path(pdb_path)
    if not path.is_file():
        return None, "", {}

    ca_coords: list[list[float]] = []
    pdb_seq_list: list[str] = []
    pdb_to_idx: dict[int, int] = {}
    seen_residues = set()
    actual_target_chain = target_chain
    prev_res_seq = None
    prev_xyz = None

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("TER"):
                    if target_chain is None:
                        break
                    continue

                if not line.startswith(("ATOM  ", "HETATM")):
                    continue

                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue

                res_name = line[17:20].strip()
                if res_name not in VALID_AMINO_ACIDS:
                    continue

                chain = line[21]
                res_seq_str = line[22:26].strip()
                try:
                    res_seq = int(res_seq_str)
                except ValueError:
                    continue

                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue

                if actual_target_chain is not None and chain != actual_target_chain:
                    if target_chain is None:
                        break
                    continue

                if prev_res_seq is not None and res_seq < prev_res_seq:
                    continue

                if prev_xyz is not None and res_seq == prev_res_seq + 1:
                    dist = np.linalg.norm(np.array([x, y, z]) - np.array(prev_xyz))
                    if dist > 5.0:
                        prev_res_seq = res_seq
                        prev_xyz = [x, y, z]
                        continue

                if actual_target_chain is None:
                    actual_target_chain = chain

                i_code = line[26]
                res_key = (chain, res_seq, i_code)
                if res_key in seen_residues:
                    continue

                one_letter = THREE_TO_ONE.get(res_name, "X")
                seen_residues.add(res_key)
                ca_coords.append([x, y, z])
                pdb_seq_list.append(one_letter)

                if res_seq not in pdb_to_idx:
                    pdb_to_idx[res_seq] = len(pdb_seq_list) - 1

                prev_res_seq = res_seq
                prev_xyz = [x, y, z]

        if not ca_coords:
            return None, "", {}

        coords_arr = np.array(ca_coords, dtype=np.float32)
        pdb_seq = "".join(pdb_seq_list)
        return coords_arr, pdb_seq, pdb_to_idx

    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return None, "", {}


def parse_pdb_all_atom_coordinates(
    pdb_path: str | Path, target_chain: str | None = None
) -> dict[tuple[str, int, str], np.ndarray]:
    path = Path(pdb_path)
    if not path.is_file():
        return {}

    atom_coords: dict[tuple[str, int, str], np.ndarray] = {}
    actual_target_chain = target_chain

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("TER"):
                if target_chain is None:
                    break
                continue

            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            chain = line[21]
            if actual_target_chain is None:
                actual_target_chain = chain

            if actual_target_chain is not None and chain != actual_target_chain:
                if target_chain is None:
                    break
                continue

            atom_name = line[12:16].strip()
            res_seq_str = line[22:26].strip()
            try:
                res_seq = int(res_seq_str)
            except ValueError:
                continue

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue

            key = (chain, res_seq, atom_name)
            if key not in atom_coords:
                atom_coords[key] = np.array([x, y, z], dtype=np.float32)

    return atom_coords


def pairwise_align_wt_pdb(wt_seq: str, pdb_seq: str) -> list[tuple[int, int]]:
    n = len(wt_seq)
    m = len(pdb_seq)

    match_score = 2
    mismatch_penalty = -1
    gap_penalty = -2

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i * gap_penalty
    for j in range(m + 1):
        dp[0, j] = j * gap_penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score_diag = dp[i - 1, j - 1] + (match_score if wt_seq[i - 1] == pdb_seq[j - 1] else mismatch_penalty)
            score_up = dp[i - 1, j] + gap_penalty
            score_left = dp[i, j - 1] + gap_penalty
            dp[i, j] = max(score_diag, score_up, score_left)

    i, j = n, m
    aligned_pairs = []

    while i > 0 or j > 0:
        prev_i, prev_j = i, j
        if i > 0 and j > 0:
            score_diag = dp[i - 1, j - 1] + (match_score if wt_seq[i - 1] == pdb_seq[j - 1] else mismatch_penalty)
            if dp[i, j] == score_diag:
                aligned_pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0:
            score_up = dp[i - 1, j] + gap_penalty
            if dp[i, j] == score_up:
                i -= 1
                continue
        if j > 0:
            score_left = dp[i, j - 1] + gap_penalty
            if dp[i, j] == score_left:
                j -= 1
                continue
        if i == prev_i and j == prev_j:
            break

    aligned_pairs.reverse()
    return aligned_pairs


def interpolate_missing_coordinates(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords_out = coords.copy()
    num_res = len(coords)
    if num_res == 0:
        return coords_out

    if not np.any(mask):
        for i in range(num_res):
            coords_out[i] = np.array([i * 3.8, 0.0, 0.0], dtype=np.float32)
        return coords_out

    resolved_indices = np.where(mask)[0]

    for idx in range(len(resolved_indices) - 1):
        left_idx = resolved_indices[idx]
        right_idx = resolved_indices[idx + 1]
        gap_len = right_idx - left_idx - 1
        if gap_len > 0:
            start_coord = coords_out[left_idx]
            end_coord = coords_out[right_idx]
            for k in range(1, gap_len + 1):
                t = k / (gap_len + 1)
                coords_out[left_idx + k] = (1.0 - t) * start_coord + t * end_coord

    first_resolved = resolved_indices[0]
    if first_resolved > 0:
        if len(resolved_indices) >= 2:
            second_resolved = resolved_indices[1]
            vec = coords_out[first_resolved] - coords_out[second_resolved]
            norm = np.linalg.norm(vec)
            direction = vec / norm if norm > 1e-5 else np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        for i in range(first_resolved - 1, -1, -1):
            coords_out[i] = coords_out[i + 1] + direction * 3.8

    last_resolved = resolved_indices[-1]
    if last_resolved < num_res - 1:
        if len(resolved_indices) >= 2:
            penultimate_resolved = resolved_indices[-2]
            vec = coords_out[last_resolved] - coords_out[penultimate_resolved]
            norm = np.linalg.norm(vec)
            direction = vec / norm if norm > 1e-5 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        for i in range(last_resolved + 1, num_res):
            coords_out[i] = coords_out[i - 1] + direction * 3.8

    return coords_out


def align_and_map_coordinates(wt_seq: str, pdb_path: str | Path) -> np.ndarray | None:
    pdb_coords, pdb_seq, _ = parse_pdb_ca_coordinates_and_seq(pdb_path)
    if pdb_coords is None or not pdb_seq:
        return None

    alignment = pairwise_align_wt_pdb(wt_seq, pdb_seq)
    if not alignment:
        return None

    num_res = len(wt_seq)
    coords_wt = np.zeros((num_res, 3), dtype=np.float32)
    mask = np.zeros(num_res, dtype=bool)

    for wt_idx, pdb_idx in alignment:
        if 0 <= wt_idx < num_res and 0 <= pdb_idx < len(pdb_coords):
            coords_wt[wt_idx] = pdb_coords[pdb_idx]
            mask[wt_idx] = True

    return interpolate_missing_coordinates(coords_wt, mask)

def get_active_site_shield(
    coords: np.ndarray, active_site_indices: list[int], radius: float = 0.0
) -> np.ndarray:
    num_res = len(coords)
    shield_mask = np.zeros(num_res, dtype=bool)
    if num_res == 0 or not active_site_indices:
        return shield_mask

    valid_indices = [idx for idx in active_site_indices if 0 <= idx < num_res]
    if not valid_indices:
        return shield_mask

    if radius <= 0.0:
        shield_mask[valid_indices] = True
        return shield_mask

    active_coords = coords[valid_indices]
    diffs = coords[:, np.newaxis, :] - active_coords[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    min_dists = np.min(dists, axis=-1)
    shield_mask = min_dists <= radius

    return shield_mask

def save_pdb_with_custom_bfactor(
    input_pdb_path: str | Path,
    output_pdb_path: str | Path,
    importance_map: dict[int, float],
    pdb_to_idx: dict[int, int],
    alignment_mapping: list[tuple[int, int]],
) -> None:
    in_path = Path(input_pdb_path)
    out_path = Path(output_pdb_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input PDB file not found: {in_path}")

    pdb_idx_to_wt_idx = {pdb_idx: wt_idx for wt_idx, pdb_idx in alignment_mapping}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as infile, out_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if line.startswith(("ATOM  ", "HETATM")):
                if len(line) < 80:
                    line = line.rstrip("\r\n").ljust(80) + "\n"

                res_seq_str = line[22:26].strip()
                try:
                    res_seq = int(res_seq_str)
                except ValueError:
                    outfile.write(line)
                    continue

                importance_val = 0.0
                pdb_idx = pdb_to_idx.get(res_seq)
                if pdb_idx is None:
                    num_mono = len(pdb_to_idx)
                    if num_mono > 0:
                        min_res = min(pdb_to_idx.keys())
                        res_seq_mapped = (res_seq - min_res) % num_mono + min_res
                        pdb_idx = pdb_to_idx.get(res_seq_mapped)

                if pdb_idx is not None:
                    wt_idx = pdb_idx_to_wt_idx.get(pdb_idx)
                    if wt_idx is not None:
                        importance_val = importance_map.get(wt_idx, 0.0)

                clamped_val = max(-99.99, min(999.99, importance_val))
                bfactor_str = f"{clamped_val:6.2f}"
                if len(bfactor_str) != 6:
                    bfactor_str = "  0.00"

                new_line = line[:60] + bfactor_str + line[66:]
                outfile.write(new_line)
            else:
                outfile.write(line)