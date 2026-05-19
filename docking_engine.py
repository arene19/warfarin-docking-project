# docking_engine.py
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
import pandas as pd
from vina import Vina

@dataclass
class DockingResult:
    ligand_name: str
    target_name: str
    best_affinity: float
    all_affinities: list = field(default_factory=list)
    docked_pdbqt: str = ""
    success: bool = False
    error: str = ""
    runtime_seconds: float = 0.0

def run_vina_docking(receptor_pdbqt: str,
                     ligand_pdbqt: str,
                     output_pdbqt: str,
                     box_params: dict,
                     exhaustiveness: int = 64, # Increase to 64 or 128
                     seed: int = 42) -> DockingResult:
    
    ligand_name = Path(ligand_pdbqt).stem
    target_name = Path(receptor_pdbqt).stem

    print(f"  Docking: {ligand_name} -> {target_name}")
    start_time = time.time()

    result = DockingResult(ligand_name=ligand_name, target_name=target_name, 
                           best_affinity=float("inf"), docked_pdbqt=output_pdbqt)

    try:
        v = Vina(sf_name="vina", seed=seed, verbosity=0)
        v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)

        # Ensure these are passed as lists of floats
        v.compute_vina_maps(
            center=[float(x) for x in box_params["center"]],
            box_size=[float(x) for x in box_params["size"]]
        )

        v.dock(exhaustiveness=exhaustiveness, n_poses=20, min_rmsd=1.0)
        energies = v.energies(n_poses=20)

        os.makedirs(os.path.dirname(output_pdbqt), exist_ok=True)
        v.write_poses(output_pdbqt, n_poses=20, overwrite=True)

        affinities = [e[0] for e in energies]
        result.all_affinities = affinities
        result.best_affinity = affinities[0] if affinities else float("inf")
        result.success = True

    except Exception as e:
        result.error = str(e)
        result.success = False
        print(f"  [ERROR] Docking failed: {e}")

    result.runtime_seconds = time.time() - start_time
    return result

def virtual_screening(ligand_pdbqt_dir: str,
                      receptor_pdbqt: str,
                      target_name: str,
                      box_params: dict,
                      output_dir: str = "results/",
                      exhaustiveness: int = 64) -> pd.DataFrame:
    
    os.makedirs(output_dir, exist_ok=True)
    ligand_files = list(Path(ligand_pdbqt_dir).glob("*.pdbqt"))
    
    results = []
    for ligand_path in ligand_files:
        output_pdbqt = os.path.join(output_dir, f"{ligand_path.stem}_{target_name}_docked.pdbqt")
        
        result = run_vina_docking(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=str(ligand_path),
            output_pdbqt=output_pdbqt,
            box_params=box_params,
            exhaustiveness=exhaustiveness
        )
        results.append(asdict(result))

    df = pd.DataFrame(results).sort_values("best_affinity")
    df.to_csv(os.path.join(output_dir, f"{target_name}_screening_results.csv"), index=False)
    return df
