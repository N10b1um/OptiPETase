import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from config import CONFIG
from src.benchmark import BenchmarkEvaluator
from src.biophysics import (
    calculate_local_packing_density,
    evaluate_mutant_catalytic_geometry,
    validate_mutation_safety,
)
from src.models_inference import ESM2EmbeddingExtractor, LigandMPNNRunner
from src.pareto import CandidateSelector, compute_pareto_fronts
from src.preprocess import protonate_structure, relax_structure, repair_pdb
from src.structure import (
    download_pdb,
    pairwise_align_wt_pdb,
    parse_pdb_ca_coordinates_and_seq,
    save_pdb_with_custom_bfactor,
)
from src.visualize import (
    generate_pymol_session,
    plot_candidate_clusters_pca,
    plot_pareto_frontier,
)


def run_single_protein_pipeline(
    pdb_id: str,
    active_site_triad: list[int],
    oxyanion_hole: list[int],
) -> dict[str, Any]:
    pdb_dir = Path(CONFIG["PDB_DIR"])
    results_dir = Path(CONFIG["RESULTS_DIR"]) / pdb_id
    plot_dir = results_dir / "plots"
    tables_dir = results_dir / "tables"
    benchmark_dir = Path(CONFIG["BENCHMARK_DIR"])

    pdb_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_pdb = download_pdb(pdb_id, output_dir=pdb_dir)
    repaired_pdb = repair_pdb(
        input_pdb=raw_pdb,
        output_pdb=pdb_dir / f"{pdb_id.lower()}_repaired.pdb",
        mutations=CONFIG["MUTATIONS_FIX"],
        keep_water=CONFIG["FIX_KEEP_WATER"],
    )
    protonated_pdb = protonate_structure(
        input_pdb=repaired_pdb,
        output_pdb=pdb_dir / f"{pdb_id.lower()}_protonated.pdb",
        tmp_pqr=pdb_dir / f"tmp_{pdb_id.lower()}.pqr",
        ph=CONFIG["PROTONATION_PH"],
        forcefield=CONFIG["PROTONATION_FF"],
    )
    relaxed_pdb = relax_structure(
        input_pdb=protonated_pdb,
        output_pdb=pdb_dir / f"{pdb_id.lower()}_relaxed.pdb",
        ff_files=CONFIG["FF_FILES_OPENMM"],
    )

    coords, wt_seq, pdb_to_idx = parse_pdb_ca_coordinates_and_seq(relaxed_pdb)
    if coords is None or not wt_seq:
        raise RuntimeError(f"Failed to parse coordinates for protein {pdb_id}")

    _, raw_seq, raw_pdb_to_idx = parse_pdb_ca_coordinates_and_seq(raw_pdb)
    idx_to_pdb = {idx: pdb_res for pdb_res, idx in pdb_to_idx.items()}

    if raw_seq and raw_pdb_to_idx:
        raw_to_rel = dict(pairwise_align_wt_pdb(raw_seq, wt_seq))
        mapped_triad = [
            idx_to_pdb[raw_to_rel[raw_pdb_to_idx[r]]]
            for r in active_site_triad
            if r in raw_pdb_to_idx and raw_pdb_to_idx[r] in raw_to_rel and raw_to_rel[raw_pdb_to_idx[r]] in idx_to_pdb
        ]
        mapped_oxyanion = [
            idx_to_pdb[raw_to_rel[raw_pdb_to_idx[r]]]
            for r in oxyanion_hole
            if r in raw_pdb_to_idx and raw_pdb_to_idx[r] in raw_to_rel and raw_to_rel[raw_pdb_to_idx[r]] in idx_to_pdb
        ]
    else:
        mapped_triad = [r for r in active_site_triad if r in pdb_to_idx]
        mapped_oxyanion = [r for r in oxyanion_hole if r in pdb_to_idx]

    triad_residues = mapped_triad if mapped_triad else active_site_triad
    oxyanion_residues = mapped_oxyanion if mapped_oxyanion else oxyanion_hole
    triad_indices = [pdb_to_idx[r] for r in triad_residues if r in pdb_to_idx]
    oxyanion_indices = [pdb_to_idx[r] for r in oxyanion_residues if r in pdb_to_idx]

    locked_core_indices = set(triad_indices + oxyanion_indices)
    packing_density = calculate_local_packing_density(coords)

    extractor = ESM2EmbeddingExtractor(model_name=CONFIG["MODEL_NAME"])
    features = extractor.extract_features(wt_seq)
    zero_shot_scores = features["zero_shot_scores"]

    mpnn_runner = LigandMPNNRunner(
        temperature=CONFIG["TEMPERATURE"],
        catalytic_triad_pdb_ids=triad_residues,
    )
    raw_candidates = mpnn_runner.generate_candidates(
        input_pdb=relaxed_pdb,
        wt_seq=wt_seq,
        pdb_to_idx=pdb_to_idx,
        out_dir=results_dir / "ligandmpnn",
    )

    evaluated_candidates = []
    for cand in tqdm(raw_candidates, desc=f"Evaluating {pdb_id} candidates"):
        pos_idx = cand["pos_idx"]
        mut_aa = cand["mut_aa"]

        fitness_score = float(zero_shot_scores.get(pos_idx, {}).get(mut_aa, 0.0))
        if fitness_score < 0.0:
            continue

        mut_seq_list = list(wt_seq)
        mut_seq_list[pos_idx] = mut_aa
        mut_seq = "".join(mut_seq_list)

        is_safe, _ = validate_mutation_safety(
            wt_seq=wt_seq,
            mut_seq=mut_seq,
            pos_idx=pos_idx,
            coords=coords,
            locked_indices=locked_core_indices,
            packing_density=packing_density,
        )
        if not is_safe:
            continue

        geom_eval = evaluate_mutant_catalytic_geometry(
            wt_pdb_path=relaxed_pdb,
            mutations=[cand["mutation_str"]],
            ff_files=CONFIG["FF_FILES_OPENMM"],
            triad_pdb_ids=triad_residues,
            oxyanion_pdb_ids=oxyanion_residues,
            ph=CONFIG["PROTONATION_PH"],
        )

        evaluated_candidates.append({
            **cand,
            "esm2_fitness_score": fitness_score,
            "catalytic_distortion": geom_eval["catalytic_distortion"],
            "triad_rmsd": geom_eval["triad_rmsd"],
            "d1_mut": geom_eval["d1_mut"],
            "d2_mut": geom_eval["d2_mut"],
        })

    ranked_candidates = compute_pareto_fronts(evaluated_candidates)
    rank1_candidates = [c for c in ranked_candidates if c.get("pareto_rank") == 1]

    selector = CandidateSelector(n_clusters=CONFIG["N_CLUSTERS"], seed=CONFIG["SEED"])
    top5_selected = selector.select_diverse_top_candidates(ranked_candidates, coords_wt=coords)

    bench_results: list[dict[str, Any]] = []
    if pdb_id.upper() == "5XJH":
        evaluator = BenchmarkEvaluator(
            benchmark_csv_path=benchmark_dir / "nature_variants_activity.csv",
            wt_pdb_path=relaxed_pdb,
            model_name=CONFIG["MODEL_NAME"],
            results_dir=tables_dir,
        )
        bench_summary = evaluator.run_evaluation()
        bench_results = bench_summary["results"]

    pareto_plot_path = plot_dir / "pareto_frontier.png"
    plot_pareto_frontier(
        all_candidates=ranked_candidates,
        pareto_rank_1=rank1_candidates,
        benchmark_variants=bench_results,
        selected_top5=top5_selected,
        save_path=pareto_plot_path,
    )

    if evaluated_candidates:
        all_features = selector.extract_mutation_features(evaluated_candidates, coords_wt=coords)
        if len(all_features) >= 2:
            pca = PCA(n_components=2, random_state=CONFIG["SEED"])
            pca_coords = pca.fit_transform(all_features)
            for i, cand in enumerate(evaluated_candidates):
                cand["pca_x"] = float(pca_coords[i, 0])
                cand["pca_y"] = float(pca_coords[i, 1])

            labels = np.array([c.get("pareto_rank", 1) for c in evaluated_candidates])
            pca_plot_path = plot_dir / "pca_clusters.png"
            plot_candidate_clusters_pca(
                all_pca_coords=pca_coords,
                all_cluster_labels=labels,
                selected_candidates=top5_selected,
                save_path=str(pca_plot_path),
            )

    raw_means = np.array([
        float(np.mean(list(zero_shot_scores.get(i, {}).values())))
        for i in range(len(wt_seq))
    ], dtype=np.float32)

    mean_val = float(np.mean(raw_means))
    std_val = float(np.std(raw_means)) if np.std(raw_means) > 1e-5 else 1.0
    z_scores = (raw_means - mean_val) / std_val

    importance_map = {i: float(z_scores[i]) for i in range(len(wt_seq))}

    alignment = pairwise_align_wt_pdb(wt_seq, wt_seq)
    mapped_pdb_path = results_dir / f"{pdb_id.lower()}_importance.pdb"
    save_pdb_with_custom_bfactor(
        input_pdb_path=relaxed_pdb,
        output_pdb_path=mapped_pdb_path,
        importance_map=importance_map,
        pdb_to_idx=pdb_to_idx,
        alignment_mapping=alignment,
    )

    pymol_script_path = results_dir / "visualize_candidates.pml"
    generate_pymol_session(
        pdb_id=pdb_id,
        mapped_pdb_path=mapped_pdb_path,
        selected_candidates=top5_selected,
        active_site_residues=triad_residues,
        output_script_path=pymol_script_path,
        plot_dir=plot_dir,
    )

    candidates_json_path = results_dir / "top_candidates.json"
    with open(str(candidates_json_path), "w", encoding="utf-8") as f:
        json.dump(top5_selected, f, indent=2)

    return {
        "pdb_id": pdb_id,
        "selected_candidates": top5_selected,
        "plot_path": str(pareto_plot_path),
        "pymol_script_path": str(pymol_script_path),
        "mapped_pdb_path": str(mapped_pdb_path),
    }


def main() -> None:
    targets = [
        {
            "pdb_id": "5XJH",
            "active_site_triad": [160, 206, 237],
            "oxyanion_hole": [87, 161],
        }
    ]

    for target in targets:
        run_single_protein_pipeline(
            pdb_id=target["pdb_id"],
            active_site_triad=target["active_site_triad"],
            oxyanion_hole=target["oxyanion_hole"],
        )


if __name__ == "__main__":
    main()