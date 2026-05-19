# docking_engine.py
"""
AutoDock Vina docking engine.
Supports: single ligand, batch docking, virtual screening.
"""

import os
import json
import time
import subprocess
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class DockingResult:
    """Container for a single docking experiment result."""
    ligand_name: str
    target_name: str
    best_affinity: float          # kcal/mol
    all_affinities: list = field(default_factory=list)  # all modes
    rmsd_lb: list = field(default_factory=list)  # RMSD from best (lower bound)
    rmsd_ub: list = field(default_factory=list)  # RMSD from best (upper bound)
    docked_pdbqt: str = ""
    success: bool = False
    error: str = ""
    runtime_seconds: float = 0.0


def run_vina_docking(receptor_pdbqt: str,
                     ligand_pdbqt: str,
                     output_pdbqt: str,
                     box_params: dict,
                     exhaustiveness: int = 32,
                     num_modes: int = 20,
                     energy_range: float = 3.0,
                     seed: int = 42) -> DockingResult:
    """
    Execute AutoDock Vina docking via Python bindings.
    
    Returns DockingResult with binding affinities and poses.
    """
    from vina import Vina
    
    ligand_name = Path(ligand_pdbqt).stem
    target_name = Path(receptor_pdbqt).stem
    
    print(f"\n  Docking: {ligand_name} -> {target_name}")
    start_time = time.time()
    
    result = DockingResult(
        ligand_name=ligand_name,
        target_name=target_name,
        best_affinity=float("inf"),
        docked_pdbqt=output_pdbqt
    )
    
    try:
        # Initialize Vina
        v = Vina(sf_name="vina", seed=seed, verbosity=0)
        
        # Set receptor
        v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)
        
        # Set ligand
        v.set_ligand_from_file(ligand_pdbqt)
        
        # Configure search space
        v.compute_vina_maps(
            center=list(box_params["center"]),
            box_size=list(box_params["size"])
        )
        
        # Run docking
        v.dock(
            exhaustiveness=exhaustiveness,
            n_poses=num_modes,
            min_rmsd=1.0,  # minimum RMSD between poses
            max_evals=0    # 0 = auto
        )
        
        # Get energies
        energies = v.energies(n_poses=num_modes)
        
        # Write docked poses
        os.makedirs(os.path.dirname(output_pdbqt) or ".", exist_ok=True)
        v.write_poses(output_pdbqt, n_poses=num_modes, overwrite=True)
        
        # Parse energies
        affinities = [e[0] for e in energies]  # first column = total energy
        
        result.all_affinities = affinities
        result.best_affinity = affinities[0] if affinities else float("inf")
        result.success = True
        
    except Exception as e:
        result.error = str(e)
        result.success = False
        print(f"  [ERROR] Docking failed: {e}")
    
    result.runtime_seconds = time.time() - start_time
    
    status = "SUCCESS" if result.success else "FAILED"
    print(f"  [{status}] Best affinity: {result.best_affinity:.2f} kcal/mol "
          f"({result.runtime_seconds:.1f}s)")
    
    return result


def parse_vina_log(log_path: str) -> list:
    """
    Parse AutoDock Vina log file to extract all binding modes.
    Returns list of (mode, affinity, rmsd_lb, rmsd_ub).
    """
    modes = []
    
    with open(log_path) as f:
        lines = f.readlines()
    
    parsing = False
    for line in lines:
        if "-----+------------+----------+----------" in line:
            parsing = True
            continue
        
        if parsing and line.strip():
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    mode    = int(parts[0])
                    affin   = float(parts[1])
                    rmsd_lb = float(parts[2])
                    rmsd_ub = float(parts[3])
                    modes.append((mode, affin, rmsd_lb, rmsd_ub))
                except ValueError:
                    parsing = False
    
    return modes


# docking_engine.py

def virtual_screening(ligand_pdbqt_dir: str,
                      receptor_pdbqt: str,
                      target_name: str,
                      box_params: dict,
                      output_dir: str = "results/",
                      exhaustiveness: int = 32,  # Added argument to handle config values
                      max_workers: int = 4) -> pd.DataFrame:
    """
    Run virtual screening: dock all ligands against one receptor.
    Uses parallel execution for efficiency.
    """
    os.makedirs(output_dir, exist_ok=True)

    ligand_files = list(Path(ligand_pdbqt_dir).glob("*.pdbqt"))
    print(f"\n{'='*60}")
    print(f"  Virtual Screening: {len(ligand_files)} ligands -> {target_name}")
    print(f"{'='*60}")

    results = []

    def dock_single(ligand_path):
        output_pdbqt = os.path.join(
            output_dir,
            f"{ligand_path.stem}_{target_name}_docked.pdbqt"
        )
        return run_vina_docking(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=str(ligand_path),
            output_pdbqt=output_pdbqt,
            box_params=box_params,
            exhaustiveness=exhaustiveness  # Passed dynamic config value here
        )

    # Sequential (safer for GPU-limited environments)
    for ligand_path in ligand_files:
        result = dock_single(ligand_path)
        results.append(asdict(result))

    df = pd.DataFrame(results)
    df = df.sort_values("best_affinity").reset_index(drop=True)

    # Save results
    csv_path = os.path.join(output_dir, f"{target_name}_screening_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    return df    
    def dock_single(ligand_path):
        output_pdbqt = os.path.join(
            output_dir, 
            f"{ligand_path.stem}_{target_name}_docked.pdbqt"
        )
        return run_vina_docking(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=str(ligand_path),
            output_pdbqt=output_pdbqt,
            box_params=box_params,
            exhaustiveness=32
        )
    
    # Sequential (safer for GPU-limited environments)
    for ligand_path in ligand_files:
        result = dock_single(ligand_path)
        results.append(asdict(result))
    
    df = pd.DataFrame(results)
    df = df.sort_values("best_affinity").reset_index(drop=True)
    
    # Save results
    csv_path = os.path.join(output_dir, f"{target_name}_screening_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")
    
    return df


if __name__ == "__main__":
    from grid_box import BINDING_BOXES
    
    # Example: single docking
    result = run_vina_docking(
        receptor_pdbqt="pdbqt_receptors/7LCT_chainA_protonated.pdbqt",
        ligand_pdbqt="pdbqt_ligands/Warfarin.pdbqt",
        output_pdbqt="results/Warfarin_VKORC1_docked.pdbqt",
        box_params=BINDING_BOXES["VKORC1"]
    )
    print(f"\nBest binding affinity: {result.best_affinity:.2f} kcal/mol")
    print(f"All poses: {result.all_affinities}")

