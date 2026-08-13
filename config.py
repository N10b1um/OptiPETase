from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "SEED": 42,
    "MODEL_NAME": "facebook/esm2_t33_650M_UR50D",
    "BATCH_SIZE": 1,
    "TARGET_PDB": "8OTA",
    "ACTIVE_SITE_RESIDUES": [165, 210, 242],
    "ACTIVE_SITE_RADIUS": 9.0,
    "PDB_DIR": "./pdbs",
    "RESULTS_DIR": "./results",
    "PLOT_DIR": "./results/plots",
    "MUTATIONS_FIX": {}, #{"A": ["ALA-165-SER"]} todo: add docstring,
    "FIX_KEEP_WATER": False,
    "PROTONATION_PH": 7.4,
    "PROTONATION_FF": "AMBER",
    "FF_FILES_OPENMM": ['amber14-all.xml', 'implicit/obc2.xml']
}