import os
import numpy as np
import logging
from pathlib import Path
import torch
import copy
from typing import List, Dict, Any, Tuple

from config import CONFIG

from src.models_inference import (
    ESM2EmbeddingExtractor,
    LigandMPNNRunner
)
from src.biophysics import validate_mutation_safety
from src.selector import CandidateSelector
from src.visualize import (
    plot_candidate_clusters_pca,
    generate_pymol_script
)

from src.structure import (
    download_pdb,
    parse_pdb_ca_coordinates_and_seq,
    align_and_map_coordinates,
    get_active_site_shield,
    save_pdb_with_custom_bfactor,
    pairwise_align_wt_pdb
)

from src.preprocess import repair_pdb, protonate_structure, relax_structure

logger = logging.getLogger(__name__)


def parse_ligandmpnn_designs(fasta_path: Path) -> List[str]:
    if not fasta_path.exists():
        logger.warning(f"LigandMPNN output FASTA file not found at: {fasta_path}")
        return []
    
    sequences = []
    try:
        with open(fasta_path, "r", encoding="utf-8") as f:
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                sequences.append("".join(current_seq))
    except Exception as e:
        logger.error(f"Error parsing LigandMPNN FASTA file: {e}")
        return []
        
    return sequences


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    os.makedirs(CONFIG['PDB_DIR'], exist_ok=True)
    os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)
    os.makedirs(CONFIG['PLOT_DIR'], exist_ok=True)

    logger.info(f"Downloading target PDB: {CONFIG['TARGET_PDB']}")
    pdb_path = download_pdb(CONFIG['TARGET_PDB'], CONFIG['PDB_DIR'])
    
    repaired_pdb_path = Path(CONFIG['PDB_DIR']) / "repaired.pdb"
    logger.info(f"Repairing structure -> {repaired_pdb_path}")
    repaired_pdb_path = repair_pdb(
        input_pdb=pdb_path,
        output_pdb=repaired_pdb_path,
        mutations=CONFIG['MUTATIONS_FIX'],
        keep_water=CONFIG['FIX_KEEP_WATER']
    )
    
    protonated_pdb_path = Path(CONFIG['PDB_DIR']) / "protonated.pdb"
    logger.info(f"Protonating structure -> {protonated_pdb_path}")
    protonated_pdb_path = protonate_structure(
        input_pdb=repaired_pdb_path,
        output_pdb=protonated_pdb_path,
        tmp_pqr=Path(CONFIG['PDB_DIR']) / "tmp_pdb2pqr.pqr",
        ph=CONFIG['PROTONATION_PH'],
        forcefield=CONFIG['PROTONATION_FF']
    )
    if protonated_pdb_path is None or not protonated_pdb_path.exists():
        logger.error("Protonation failed. Falling back to repaired structure.")
        protonated_pdb_path = repaired_pdb_path
        
    relaxed_pdb_path = Path(CONFIG['PDB_DIR']) / "relaxed.pdb"
    logger.info(f"Minimizing energy (relaxation) -> {relaxed_pdb_path}")
    relax_structure(
        input_pdb=protonated_pdb_path,
        output_pdb=relaxed_pdb_path,
        ff_files=CONFIG['FF_FILES_OPENMM']
    )

    logger.info("Parsing C-alpha coordinates and residue mapping from relaxed structure...")
    coords, seq, pdb_to_idx = parse_pdb_ca_coordinates_and_seq(str(relaxed_pdb_path))
    if coords is None or not seq:
        raise ValueError("Failed to parse C-alpha coordinates or sequence from the relaxed structure.")

    idx_to_pdb = {idx: pdb_res for pdb_res, idx in pdb_to_idx.items()}

    logger.info("Generating structural protection shield for active site machinery...")
    active_site_indices = [
        pdb_to_idx[res] for res in CONFIG['ACTIVE_SITE_RESIDUES'] if res in pdb_to_idx
    ]
    if not active_site_indices:
        logger.warning("No active site residues matched the PDB sequence. Shield mask will be empty.")
        
    shield_mask = get_active_site_shield(
        coords=coords,
        active_site_indices=active_site_indices,
        radius=CONFIG['ACTIVE_SITE_RADIUS']
    )

    logger.info("Running LigandMPNN for structure-conditioned sequence generation...")
    mpnn_runner = LigandMPNNRunner(
        model_type="ligand_mpnn",
        number_of_batches=1,
        batch_size=10,
        temperature=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if "device" in mpnn_runner.config:
        del mpnn_runner.config["device"]
        
    try:
        mpnn_runner.generate_candidates(
            input_pdb=relaxed_pdb_path,
            out_dir=CONFIG['RESULTS_DIR']
        )
    except Exception as e:
        logger.warning(f"LigandMPNN execution failed (possibly missing executable/PATH): {e}")

    fasta_path = Path(CONFIG['RESULTS_DIR']) / "seqs" / "relaxed.fa"
    designed_seqs = parse_ligandmpnn_designs(fasta_path)
    
    candidate_mutations = set()
    if len(designed_seqs) > 1:
        for designed_seq in designed_seqs[1:]:
            if len(designed_seq) != len(seq):
                continue
            for i in range(len(seq)):
                if designed_seq[i] != seq[i]:
                    candidate_mutations.add((i, designed_seq[i]))
        logger.info(f"Extracted {len(candidate_mutations)} unique mutations from LigandMPNN designs.")
    else:
        logger.warning("No LigandMPNN designs found. Falling back to single-point scanning mutagenesis.")
        for i in range(len(seq)):
            for mut_aa in ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]:
                if mut_aa != seq[i]:
                    candidate_mutations.add((i, mut_aa))

    logger.info(f"Loading evolutionary language model: {CONFIG['MODEL_NAME']}")
    extractor = ESM2EmbeddingExtractor(
        model_name=CONFIG['MODEL_NAME'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Extracting embeddings and zero-shot mutational log-likelihoods...")
    features_dict = extractor.extract_features(seq)
    zero_shot_scores = features_dict["zero_shot_scores"]

    candidates = []
    for pos_idx, mut_aa in candidate_mutations:
        mut_seq = list(seq)
        mut_seq[pos_idx] = mut_aa
        mut_seq = "".join(mut_seq)
        
        is_safe, reason = validate_mutation_safety(
            wt_seq=seq,
            mut_seq=mut_seq,
            pos_idx=pos_idx,
            coords=coords,
            shield_mask=shield_mask
        )
        if not is_safe:
            continue
            
        score = zero_shot_scores[pos_idx].get(mut_aa, -99.0)
        if score < -4.0:
            continue
            
        candidates.append({
            "wt_aa": seq[pos_idx],
            "mut_aa": mut_aa,
            "pos_idx": pos_idx,
            "pdb_idx": idx_to_pdb[pos_idx],
            "predicted_stability": -score
        })

    if not candidates:
        raise ValueError("No mutational candidates survived the biophysical and evolutionary filters.")
    logger.info(f"Total candidates surviving filters: {len(candidates)}")

    logger.info("Computing sign-corrected evolutionary Importance Map...")
    importance_map = {}
    for i in range(len(seq)):
        scores_at_pos = list(zero_shot_scores[i].values())
        if scores_at_pos:
            importance_map[i] = -np.mean(scores_at_pos)
        else:
            importance_map[i] = 0.0
            
    mapped_pdb_path = Path(CONFIG['RESULTS_DIR']) / "relaxed_importance.pdb"
    logger.info(f"Writing evolutionary importance map into B-factor column -> {mapped_pdb_path}")
    save_pdb_with_custom_bfactor(
        input_pdb_path=str(relaxed_pdb_path),
        output_pdb_path=str(mapped_pdb_path),
        importance_map=importance_map,
        pdb_to_idx=pdb_to_idx,
        alignment_mapping=pairwise_align_wt_pdb(seq, seq)
    )

    logger.info("Grouping candidates into physical-chemical diversity niches via K-Means...")
    selector = CandidateSelector(n_clusters=5, seed=CONFIG['SEED'])
    selected_candidates, pca_coords, labels = selector.select_diverse_top_candidates(candidates)
    
    logger.info("Selected Top Diverse Candidates:")
    for c in selected_candidates:
        logger.info(f"  Mutation: {c['wt_aa']}{c['pdb_idx']}{c['mut_aa']} (Cluster: {c['cluster_id']}, stability dDG: {c['predicted_stability']:.3f})")

    plot_path = Path(CONFIG['PLOT_DIR']) / "mutation_space_pca.png"
    logger.info(f"Plotting mutation space PCA projection -> {plot_path}")
    plot_candidate_clusters_pca(
        all_pca_coords=pca_coords,
        all_cluster_labels=labels,
        selected_candidates=selected_candidates,
        save_path=str(plot_path)
    )
    
    pml_path = Path(CONFIG['RESULTS_DIR']) / "visualize_candidates.pml"
    logger.info(f"Generating PyMOL script -> {pml_path}")
    generate_pymol_script(
        pdb_id=CONFIG['TARGET_PDB'],
        mapped_pdb_path=str(mapped_pdb_path),
        selected_candidates=selected_candidates,
        active_site_residues=CONFIG['ACTIVE_SITE_RESIDUES'],
        output_script_path=str(pml_path)
    )
    
    logger.info("Pipeline executed successfully!")


if __name__ == "__main__":
    main()