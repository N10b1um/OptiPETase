from pdbfixer import PDBFixer
from pathlib import Path
import subprocess
import sys
from sys import stdout
from openmm.app import *
from openmm import *
from openmm.unit import *
import logging

logger = logging.getLogger(__name__)

def repair_pdb(input_pdb: str | Path, 
                output_pdb: str | Path = "repaired.pdb",
                keep_water: bool = False, 
                mutations: dict = None
                ) -> Path:
    fixer = PDBFixer(filename=str(input_pdb))
    fixer.removeHeterogens(keepWater=keep_water)
    
    if mutations:
        for chain_id, mut_list in mutations.items():
            fixer.applyMutations(mut_list, chain_id)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    with open(str(output_pdb), 'w') as f:
            PDBFile.writeFile(fixer.topology , fixer.positions, f)
            
    return Path(output_pdb)

def protonate_structure(input_pdb: str | Path,
           output_pdb: str | Path = "output.pdb",
           tmp_pqr: str | Path = "tmp_output.pqr",
           ph: float = 7.4,
           forcefield: str = "AMBER"
           ) -> Path | None:
    if not Path(input_pdb).exists():
        raise FileNotFoundError(f"Input .pdb file does not exist: {input_pdb}")

    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    
    command = [
        "pdb2pqr",
        f"--ff={forcefield}",
        f"--with-ph={ph}",
        f"--pdb-output={output_pdb}",
        str(input_pdb),
        str(tmp_pqr)
        ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)

        Path(tmp_pqr).unlink(missing_ok=True)

    except subprocess.CalledProcessError as e:
        print(f"Error running PDB2PQR on {input_pdb}:", file = sys.stderr)
        print(e.stderr , file=sys.stderr)
        return None

    return Path(output_pdb)

def relax_structure(input_pdb: str | Path,
             output_pdb: str | Path = "relaxed.pdb",
             ff_files: list[str] = ['amber14-all.xml', 'implicit/obc2.xml']
             ) -> None:
    if not Path(input_pdb).exists():
        raise FileNotFoundError(f'Input .pdb file does not exist: {input_pdb}')
    
    pdb = PDBFile(str(input_pdb))

    forcefield = ForceField(*ff_files)

    system = forcefield.createSystem(
            pdb.topology, 
            nonbondedMethod=NoCutoff, 
            constraints=HBonds)
    
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002* picosecond) #TODO: put to function's arguments

    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()

    state = simulation.context.getState(getPositions=True)
    with open(str(output_pdb), 'w') as f:
        PDBFile.writeFile(pdb.topology, state.getPositions(), f)
