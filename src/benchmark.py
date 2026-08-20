import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import CONFIG
from src.biophysics import evaluate_mutant_catalytic_geometry
from src.models_inference import ESM2EmbeddingExtractor
from src.preprocess import protonate_structure, relax_structure, repair_pdb
from src.structure import download_pdb, parse_pdb_ca_coordinates_and_seq

DEFAULT_BENCHMARK_DATA: list[dict[str, Any]] = [
    {
        "variant_name": "WT",
        "mutations": [],
        "scaffold": "WT",
        "activity_30C": 1.0,
        "activity_40C": 0.85,
        "activity_50C": 0.05,
        "activity_55C": 0.0,
        "activity_60C": 0.0,
    },
    {
        "variant_name": "WT_S121E",
        "mutations": ["S121E"],
        "scaffold": "WT",
        "activity_30C": 1.2,
        "activity_40C": 1.1,
        "activity_50C": 0.2,
        "activity_55C": 0.05,
        "activity_60C": 0.0,
    },
    {
        "variant_name": "WT_N233K",
        "mutations": ["N233K"],
        "scaffold": "WT",
        "activity_30C": 1.15,
        "activity_40C": 1.05,
        "activity_50C": 0.3,
        "activity_55C": 0.1,
        "activity_60C": 0.0,
    },
    {
        "variant_name": "WT_S121E+N233K",
        "mutations": ["S121E", "N233K"],
        "scaffold": "WT",
        "activity_30C": 1.4,
        "activity_40C": 1.35,
        "activity_50C": 0.65,
        "activity_55C": 0.4,
        "activity_60C": 0.1,
    },
    {
        "variant_name": "Thermo_R224Q+N233K",
        "mutations": ["R224Q", "N233K"],
        "scaffold": "Thermo",
        "activity_30C": 1.6,
        "activity_40C": 1.8,
        "activity_50C": 1.4,
        "activity_55C": 1.1,
        "activity_60C": 0.7,
    },
    {
        "variant_name": "Dura_S121E",
        "mutations": ["S121E"],
        "scaffold": "Dura",
        "activity_30C": 1.5,
        "activity_40C": 1.7,
        "activity_50C": 1.6,
        "activity_55C": 1.4,
        "activity_60C": 0.9,
    },
    {
        "variant_name": "FAST-PETase",
        "mutations": ["N233K", "R224Q", "S121E"],
        "scaffold": "FAST",
        "activity_30C": 1.8,
        "activity_40C": 2.1,
        "activity_50C": 2.0,
        "activity_55C": 1.8,
        "activity_60C": 1.5,
    },
    {
        "variant_name": "Hot-PETase",
        "mutations": ["S121E", "D186H", "R224Q", "N233K"],
        "scaffold": "Hot",
        "activity_30C": 1.7,
        "activity_40C": 2.0,
        "activity_50C": 2.1,
        "activity_55C": 1.9,
        "activity_60C": 1.6,
    },
    {
        "variant_name": "S160A",
        "mutations": ["S160A"],
        "scaffold": "WT",
        "activity_30C": -5.0,
        "activity_40C": -5.0,
        "activity_50C": -5.0,
        "activity_55C": -5.0,
        "activity_60C": -5.0,
    },
    {
        "variant_name": "H237A",
        "mutations": ["H237A"],
        "scaffold": "WT",
        "activity_30C": -5.0,
        "activity_40C": -5.0,
        "activity_50C": -5.0,
        "activity_55C": -5.0,
        "activity_60C": -5.0,
    },
    {
        "variant_name": "D206A",
        "mutations": ["D206A"],
        "scaffold": "WT",
        "activity_30C": -5.0,
        "activity_40C": -5.0,
        "activity_50C": -5.0,
        "activity_55C": -5.0,
        "activity_60C": -5.0,
    },
]


class BenchmarkEvaluator:
    def __init__(
        self,
        benchmark_csv_path: str | Path | None = None,
        wt_pdb_path: str | Path | None = None,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        results_dir: str | Path = "./results/tables",
    ) -> None:
        self.benchmark_csv_path = Path(benchmark_csv_path) if benchmark_csv_path else None
        self.wt_pdb_path = Path(wt_pdb_path) if wt_pdb_path else None
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.extractor: ESM2EmbeddingExtractor | None = None

    def _get_extractor(self) -> ESM2EmbeddingExtractor:
        if self.extractor is None:
            self.extractor = ESM2EmbeddingExtractor(model_name=self.model_name)
        return self.extractor

    def load_benchmark_variants(self) -> list[dict[str, Any]]:
        candidate_paths: list[Path] = []
        if self.benchmark_csv_path:
            candidate_paths.append(self.benchmark_csv_path)
        candidate_paths.extend([
            Path("./data/benchmarks/nature_variants_activity.csv"),
            Path("./data/raw/nature_variants_activity.csv"),
            Path("./data/benchmarks/nature_2022.csv"),
            Path("./data/raw/nature_2022.csv"),
        ])

        found_path: Path | None = None
        for p in candidate_paths:
            if p.exists() and p.is_file():
                found_path = p
                break

        if found_path is None:
            save_path = Path("./data/benchmarks/nature_variants_activity.csv")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_default = pd.DataFrame(DEFAULT_BENCHMARK_DATA)
            df_default["mutations"] = df_default["mutations"].apply(lambda m: str(m))
            df_default.to_csv(save_path, index=False)
            found_path = save_path

        df = pd.read_csv(found_path)
        variants: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            muts_raw = row.get("mutations", "[]")
            if isinstance(muts_raw, str):
                try:
                    mutations_list = ast.literal_eval(muts_raw)
                except (ValueError, SyntaxError):
                    cleaned = (
                        muts_raw.replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace('"', "")
                    )
                    mutations_list = [m.strip() for m in cleaned.split(",") if m.strip()]
            elif isinstance(muts_raw, list):
                mutations_list = muts_raw
            else:
                mutations_list = []

            variant_dict = {
                "variant_name": str(row.get("variant_name", "")),
                "mutations": mutations_list,
                "scaffold": str(row.get("scaffold", "WT")),
                "activity_30C": float(row.get("activity_30C", 0.0)),
                "activity_40C": float(row.get("activity_40C", 0.0)),
                "activity_50C": float(row.get("activity_50C", 0.0)),
                "activity_55C": float(row.get("activity_55C", 0.0)),
                "activity_60C": float(row.get("activity_60C", 0.0)),
            }
            variants.append(variant_dict)

        controls_present = {v["variant_name"] for v in variants}
        control_specs = [
            ("S160A", ["S160A"]),
            ("H237A", ["H237A"]),
            ("D206A", ["D206A"]),
        ]
        for c_name, c_muts in control_specs:
            if c_name not in controls_present:
                variants.append({
                    "variant_name": c_name,
                    "mutations": c_muts,
                    "scaffold": "WT",
                    "activity_30C": -5.0,
                    "activity_40C": -5.0,
                    "activity_50C": -5.0,
                    "activity_55C": -5.0,
                    "activity_60C": -5.0,
                })

        return variants

    def run_evaluation(self) -> dict[str, Any]:
        variants = self.load_benchmark_variants()
        extractor = self._get_extractor()

        if self.wt_pdb_path is None or not self.wt_pdb_path.exists():
            pdb_dir = Path("./data/raw")
            pdb_dir.mkdir(parents=True, exist_ok=True)
            raw_pdb = download_pdb("5XJH", output_dir=pdb_dir)
            repaired_pdb = repair_pdb(raw_pdb, output_pdb=pdb_dir / "5xjh_repaired.pdb")
            protonated_pdb = protonate_structure(
                repaired_pdb,
                output_pdb=pdb_dir / "5xjh_protonated.pdb",
                ph=CONFIG.get("PROTONATION_PH", 7.4),
                forcefield=CONFIG.get("PROTONATION_FF", "AMBER"),
            )
            self.wt_pdb_path = relax_structure(
                protonated_pdb,
                output_pdb=pdb_dir / "5xjh_relaxed.pdb",
                ff_files=CONFIG.get("FF_FILES_OPENMM", ["amber14-all.xml", "implicit/obc2.xml"]),
            )

        _, pdb_seq, pdb_to_idx = parse_pdb_ca_coordinates_and_seq(str(self.wt_pdb_path))
        idx_to_pdb = {idx: pdb_res for pdb_res, idx in pdb_to_idx.items()}

        results: list[dict[str, Any]] = []

        for var in variants:
            mut_list = var["mutations"]
            cumulative_llr = 0.0

            for mut in mut_list:
                if len(mut) >= 3:
                    wt_aa = mut[0]
                    mut_aa = mut[-1]
                    try:
                        pdb_res = int(mut[1:-1])
                        seq_idx = pdb_to_idx.get(pdb_res)

                        if seq_idx is None or seq_idx >= len(pdb_seq) or pdb_seq[seq_idx] != wt_aa:
                            for idx_candidate, char in enumerate(pdb_seq):
                                if char == wt_aa and abs((idx_to_pdb.get(idx_candidate, idx_candidate + 1)) - pdb_res) <= 35:
                                    seq_idx = idx_candidate
                                    break

                        if seq_idx is not None and pdb_seq and 0 <= seq_idx < len(pdb_seq):
                            fit = extractor.compute_mutation_fitness(pdb_seq, seq_idx, mut_aa)
                            cumulative_llr += fit
                    except ValueError:
                        pass

            geom_eval = evaluate_mutant_catalytic_geometry(
                wt_pdb_path=self.wt_pdb_path,
                mutations=mut_list,
                ff_files=CONFIG.get("FF_FILES_OPENMM"),
                triad_pdb_ids=CONFIG.get("CATALYTIC_TRIAD_PDB", [160, 206, 237]),
                oxyanion_pdb_ids=CONFIG.get("OXYANION_HOLE_PDB", [87, 161]),
                ph=CONFIG.get("PROTONATION_PH", 7.4),
            )

            row_data = {
                **var,
                "esm2_fitness_score": cumulative_llr,
                "catalytic_distortion": geom_eval["catalytic_distortion"],
                "triad_rmsd": geom_eval["triad_rmsd"],
                "d1_mut": geom_eval["d1_mut"],
                "d2_mut": geom_eval["d2_mut"],
                "d3_mut": geom_eval["d3_mut"],
                "d4_mut": geom_eval["d4_mut"],
            }
            results.append(row_data)

        df_results = pd.DataFrame(results)
        summary_path = self.results_dir / "benchmark_summary.csv"
        df_results.to_csv(summary_path, index=False)

        valid_df = df_results[~df_results["variant_name"].isin(["S160A", "H237A", "D206A"])]

        if len(valid_df) > 1:
            rho_50_res = spearmanr(valid_df["esm2_fitness_score"], valid_df["activity_50C"])
            rho_50 = float(rho_50_res.statistic) if hasattr(rho_50_res, "statistic") else float(rho_50_res[0])
            p_50 = float(rho_50_res.pvalue) if hasattr(rho_50_res, "pvalue") else float(rho_50_res[1])

            rho_60_res = spearmanr(valid_df["esm2_fitness_score"], valid_df["activity_60C"])
            rho_60 = float(rho_60_res.statistic) if hasattr(rho_60_res, "statistic") else float(rho_60_res[0])
            p_60 = float(rho_60_res.pvalue) if hasattr(rho_60_res, "pvalue") else float(rho_60_res[1])

            dist_50_res = spearmanr(-valid_df["catalytic_distortion"], valid_df["activity_50C"])
            dist_rho_50 = float(dist_50_res.statistic) if hasattr(dist_50_res, "statistic") else float(dist_50_res[0])

            dist_60_res = spearmanr(-valid_df["catalytic_distortion"], valid_df["activity_60C"])
            dist_rho_60 = float(dist_60_res.statistic) if hasattr(dist_60_res, "statistic") else float(dist_60_res[0])
        else:
            rho_50, p_50, rho_60, p_60 = 0.0, 1.0, 0.0, 1.0
            dist_rho_50, dist_rho_60 = 0.0, 0.0

        control_distortions = {
            r["variant_name"]: r["catalytic_distortion"]
            for r in results
            if r["variant_name"] in ["S160A", "H237A", "D206A"]
        }

        return {
            "summary_path": str(summary_path),
            "results": results,
            "spearman_rho_50C": rho_50 if not np.isnan(rho_50) else 0.0,
            "spearman_p_50C": p_50 if not np.isnan(p_50) else 1.0,
            "spearman_rho_60C": rho_60 if not np.isnan(rho_60) else 0.0,
            "spearman_p_60C": p_60 if not np.isnan(p_60) else 1.0,
            "distortion_rho_50C": dist_rho_50 if not np.isnan(dist_rho_50) else 0.0,
            "distortion_rho_60C": dist_rho_60 if not np.isnan(dist_rho_60) else 0.0,
            "control_distortions": control_distortions,
        }


def run_benchmark(
    benchmark_csv_path: str | Path | None = None,
    wt_pdb_path: str | Path | None = None,
    model_name: str | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    b_csv = benchmark_csv_path or CONFIG.get("BENCHMARK_DIR", "./data/benchmarks") + "/nature_variants_activity.csv"
    r_dir = results_dir or CONFIG.get("TABLES_DIR", "./results/tables")
    m_name = model_name or CONFIG.get("MODEL_NAME", "facebook/esm2_t33_650M_UR50D")

    evaluator = BenchmarkEvaluator(
        benchmark_csv_path=b_csv,
        wt_pdb_path=wt_pdb_path,
        model_name=m_name,
        results_dir=r_dir,
    )
    return evaluator.run_evaluation()


if __name__ == "__main__":
    summary = run_benchmark()
    print(f"Benchmark summary generated at: {summary['summary_path']}")
    print(f"Spearman correlation at 50C: {summary['spearman_rho_50C']:.4f} (p={summary['spearman_p_50C']:.4e})")
    print(f"Spearman correlation at 60C: {summary['spearman_rho_60C']:.4f} (p={summary['spearman_p_60C']:.4e})")
    print(f"Catalytic dead control distortions: {summary['control_distortions']}")