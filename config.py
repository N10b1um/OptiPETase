from typing import Any

CONFIG: dict[str, Any] = {
    "SEED": 42,
    "MODEL_NAME": "facebook/esm2_t33_650M_UR50D",
    "TARGET_PDB": "5XJH",
    "PDB_DIR": "./data/raw",
    "BENCHMARK_DIR": "./data/benchmarks",
    "RESULTS_DIR": "./results",
    "PLOT_DIR": "./results/plots",
    "TABLES_DIR": "./results/tables",
    "CATALYTIC_TRIAD_PDB": [160, 206, 237],
    "OXYANION_HOLE_PDB": [87, 161],
    "PROTONATION_PH": 7.4,
    "PROTONATION_FF": "AMBER",
    "FF_FILES_OPENMM": ["amber14-all.xml", "implicit/obc2.xml"],
    "N_CLUSTERS": 5,
    "TEMPERATURE": 0.1,
    "NUM_BATCHES": 1,
    "BATCH_SIZE": 16,
    "MUTATIONS_FIX": {},
    "FIX_KEEP_WATER": False,
}