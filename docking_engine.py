import os
import time
import shutil
import glob
import json
import subprocess
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, field, asdict
import pandas as pd
from vina import Vina

from pipeline_utils import clean_subprocess_env, resolve_cpu_count

# Per-ligand wall-clock guard (seconds). A docking exceeding this is terminated
# and recorded as a failure instead of stalling the whole screen.
DEFAULT_DOCK_TIMEOUT_S = 900


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


def _serialize_affinities(affinities: list) -> str:
    """JSON list of plain floats for CSV export (avoids np.float64 repr strings)."""
    return json.dumps([float(a) for a in affinities])


def _result_row(result: DockingResult) -> dict:
    row = asdict(result)
    row["all_affinities"] = _serialize_affinities(result.all_affinities)
    return row

def prepare_flexible_receptor(receptor_pdb: str, receptor_pdbqt: str, target_name: str, flex_res_list: list = None) -> tuple:
    """Splits receptor structures into rigid/flexible layers dynamically."""
    if not flex_res_list or not receptor_pdb.endswith(".pdb"):
        return receptor_pdbqt, None

    base_dir = "results/pdbqt_receptors"
    os.makedirs(base_dir, exist_ok=True)

    pdb_name = Path(receptor_pdb).stem
    rigid_output = os.path.join(base_dir, f"{pdb_name}_rigid.pdbqt")
    flex_output = os.path.join(base_dir, f"{pdb_name}_flex.pdbqt")

    if os.path.exists(rigid_output) and os.path.exists(flex_output):
        return rigid_output, flex_output

    print(f"  [FLEX-DOCKING] Building flexible side-chains for {target_name}...")

    flex_flags = " ".join([f"-f {res}" for res in flex_res_list])
    output_basename = os.path.join(base_dir, pdb_name)

    cmd = f"mk_prepare_receptor.py -i {receptor_pdb} -o {output_basename} -p {flex_flags}"

    result = subprocess.run(cmd, shell=True, env=clean_subprocess_env(), capture_output=True, text=True)

    if result.returncode != 0 or not os.path.exists(rigid_output):
        print(f"    [WARNING] mk_prepare_receptor.py execution failed: {result.stderr.strip() or 'Unknown error'}")
        print("    Falling back to standard rigid receptor docking.")
        return receptor_pdbqt, None

    return rigid_output, flex_output


def run_vina_docking(receptor_pdbqt: str,
                     ligand_pdbqt: str,
                     output_pdbqt: str,
                     box_params: dict,
                     exhaustiveness: int = 32,
                     seed: int = 42,
                     target_name: str = "",
                     receptor_pdb: str = "",
                     flex_res_list: list = None,
                     num_modes: int = 20,
                     min_rmsd: float = 1.0,
                     n_cpu: int = None) -> DockingResult:
    """Runs a single AutoDock Vina docking (rigid or flexible side-chains), computing
    the maps in-process. Used for one-off docks and as the fallback path when the
    map-reuse fast path is unavailable.
    """
    ligand_name = Path(ligand_pdbqt).stem
    target_display_name = target_name if target_name else Path(receptor_pdbqt).stem

    print(f"  Docking: {ligand_name} -> {target_display_name}")
    start_time = time.time()

    result = DockingResult(ligand_name=ligand_name, target_name=target_display_name,
                           best_affinity=float("inf"), docked_pdbqt=output_pdbqt)

    try:
        receptor_rigid, receptor_flex = prepare_flexible_receptor(receptor_pdb, receptor_pdbqt, target_name, flex_res_list=flex_res_list)

        # CPU allocation is independent of exhaustiveness (search depth).
        allocated_cpus = resolve_cpu_count(n_cpu)

        v = Vina(sf_name="vina", cpu=allocated_cpus, seed=seed, verbosity=0)

        if receptor_flex and os.path.exists(receptor_rigid) and os.path.exists(receptor_flex):
            v.set_receptor(rigid_pdbqt_filename=receptor_rigid, flex_pdbqt_filename=receptor_flex)
        else:
            v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)

        v.set_ligand_from_file(ligand_pdbqt)

        v.compute_vina_maps(
            center=[float(x) for x in box_params["center"]],
            box_size=[float(x) for x in box_params["size"]]
        )

        v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes, min_rmsd=min_rmsd)
        get_energies = v.energies(n_poses=num_modes)

        os.makedirs(os.path.dirname(output_pdbqt), exist_ok=True)
        v.write_poses(output_pdbqt, n_poses=num_modes, overwrite=True)

        affinities = [float(e[0]) for e in get_energies]
        result.all_affinities = affinities
        result.best_affinity = affinities[0] if affinities else float("inf")
        result.success = bool(affinities)

    except Exception as e:
        result.error = str(e)
        result.success = False
        print(f"  [ERROR] Docking execution failed: {e}")

    result.runtime_seconds = time.time() - start_time
    return result


def _dock_worker(maps_prefix, ligand_pdbqt, output_pdbqt,
                 exhaustiveness, num_modes, min_rmsd, seed, n_cpu, queue):
    """Child-process body: loads precomputed maps (cheap) and docks ONE ligand.

    Running each ligand in its own process gives a reliable wall-clock timeout
    and isolates any C-level crash in the Vina engine from the parent run.
    """
    try:
        v = Vina(sf_name="vina", cpu=n_cpu, seed=seed, verbosity=0)
        v.load_maps(maps_prefix)
        v.set_ligand_from_file(ligand_pdbqt)
        v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes, min_rmsd=min_rmsd)
        energies = v.energies(n_poses=num_modes)
        os.makedirs(os.path.dirname(output_pdbqt), exist_ok=True)
        v.write_poses(output_pdbqt, n_poses=num_modes, overwrite=True)
        queue.put({"affinities": [float(e[0]) for e in energies], "error": ""})
    except Exception as exc:
        queue.put({"affinities": [], "error": repr(exc)})


def _dock_with_timeout(maps_prefix, ligand_pdbqt, output_pdbqt, target_name,
                       exhaustiveness, num_modes, min_rmsd, seed, n_cpu, dock_timeout):
    """Docks one ligand against pre-written maps inside a timeout-guarded subprocess."""
    ligand_name = Path(ligand_pdbqt).stem
    print(f"  Docking: {ligand_name} -> {target_name}")
    start_time = time.time()
    result = DockingResult(ligand_name=ligand_name, target_name=target_name,
                           best_affinity=float("inf"), docked_pdbqt=output_pdbqt)

    # 'spawn' starts a clean interpreter — safest with the Vina C-extension/OpenMP.
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_dock_worker,
        args=(maps_prefix, ligand_pdbqt, output_pdbqt,
              exhaustiveness, num_modes, min_rmsd, seed, n_cpu, queue)
    )
    proc.start()
    proc.join(dock_timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        result.error = f"timeout after {dock_timeout}s"
        print(f"  [TIMEOUT] {ligand_name} exceeded {dock_timeout}s; recorded as failed.")
    else:
        try:
            payload = queue.get_nowait()
        except Exception:
            payload = {"affinities": [], "error": "no result returned from worker"}
        if payload.get("affinities"):
            result.all_affinities = payload["affinities"]
            result.best_affinity = payload["affinities"][0]
            result.success = True
        else:
            result.error = payload.get("error", "unknown error")
            print(f"  [ERROR] Docking failed for {ligand_name}: {result.error}")

    result.runtime_seconds = time.time() - start_time
    return result


def virtual_screening(ligand_pdbqt_dir: str,
                      receptor_pdbqt: str,
                      target_name: str,
                      box_params: dict,
                      output_dir: str = "results/",
                      exhaustiveness: int = 32,
                      receptor_pdb: str = "",
                      flex_res_list: list = None,
                      needed_ligands: list = None,
                      num_modes: int = 20,
                      min_rmsd: float = 1.0,
                      n_cpu: int = None,
                      dock_timeout: int = DEFAULT_DOCK_TIMEOUT_S,
                      seed: int = 42) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    # Filter files to include only the active checked ones from config.yaml
    if needed_ligands:
        ligand_files = []
        for name in needed_ligands:
            path = Path(ligand_pdbqt_dir) / f"{name}.pdbqt"
            if path.exists():
                ligand_files.append(path)
            else:
                print(f"  [WARNING] Expected active ligand file {path} not found. Skipping.")
    else:
        ligand_files = list(Path(ligand_pdbqt_dir).glob("*.pdbqt"))

    allocated_cpus = resolve_cpu_count(n_cpu)
    center = [float(x) for x in box_params["center"]]
    box_size = [float(x) for x in box_params["size"]]

    # Prepare the receptor (rigid, or rigid+flex) ONCE for the whole target.
    receptor_rigid, receptor_flex = prepare_flexible_receptor(
        receptor_pdb, receptor_pdbqt, target_name, flex_res_list=flex_res_list
    )

    # FAST PATH (rigid receptors): compute the affinity grid maps a single time,
    # persist them, and reuse across every ligand. This removes the previous
    # behavior of recomputing identical maps once per ligand.
    maps_prefix = None
    fast_path_ok = False
    if not receptor_flex:
        try:
            maps_dir = os.path.join("results", "vina_maps", target_name)
            os.makedirs(maps_dir, exist_ok=True)
            maps_prefix = os.path.join(maps_dir, target_name)
            # Clear any stale maps from a previous run; Vina.write_maps refuses to
            # overwrite, and a different box/receptor would otherwise be reused.
            for stale in glob.glob(maps_prefix + "*.map") + glob.glob(maps_prefix + "*.fld") + glob.glob(maps_prefix + "*.xyz"):
                try:
                    os.remove(stale)
                except OSError:
                    pass
            print(f"  [MAPS] Computing Vina maps once for {target_name} (reused for all ligands)...")
            v_setup = Vina(sf_name="vina", cpu=allocated_cpus, seed=seed, verbosity=0)
            v_setup.set_receptor(rigid_pdbqt_filename=receptor_rigid)
            # force_even_voxels is required so the grid can be persisted with write_maps().
            v_setup.compute_vina_maps(center=center, box_size=box_size, force_even_voxels=True)
            v_setup.write_maps(maps_prefix)
            del v_setup
            fast_path_ok = True
        except Exception as e:
            print(f"  [MAPS] Map reuse unavailable ({e}); falling back to per-ligand map computation.")
            fast_path_ok = False

    results = []
    for ligand_path in ligand_files:
        output_pdbqt = os.path.join(output_dir, f"{ligand_path.stem}_{target_name}_docked.pdbqt")

        if fast_path_ok:
            result = _dock_with_timeout(
                maps_prefix, str(ligand_path), output_pdbqt, target_name,
                exhaustiveness, num_modes, min_rmsd, seed, allocated_cpus, dock_timeout
            )
        else:
            # Flexible receptors (or map-reuse failure): in-process per-ligand docking.
            result = run_vina_docking(
                receptor_pdbqt=receptor_pdbqt,
                ligand_pdbqt=str(ligand_path),
                output_pdbqt=output_pdbqt,
                box_params=box_params,
                exhaustiveness=exhaustiveness,
                seed=seed,
                target_name=target_name,
                receptor_pdb=receptor_pdb,
                flex_res_list=flex_res_list,
                num_modes=num_modes,
                min_rmsd=min_rmsd,
                n_cpu=allocated_cpus
            )
        results.append(_result_row(result))

    df_new = pd.DataFrame(results).sort_values("best_affinity")
    csv_path = os.path.join(output_dir, f"{target_name}_screening_results.csv")

    # Accumulative logging with a defensive backup and narrow error handling so a
    # transient read failure never silently discards the historical results.
    if os.path.exists(csv_path):
        df_old = None
        try:
            df_old = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            print(f"  [WARNING] Could not read existing {csv_path} ({e}). "
                  f"Preserving it as '{csv_path}.corrupt' and starting a fresh log.")
            try:
                shutil.move(csv_path, csv_path + ".corrupt")
            except OSError:
                pass

        if df_old is not None:
            try:
                shutil.copy2(csv_path, csv_path + ".bak")
            except OSError:
                pass
            df_old = df_old[~df_old["ligand_name"].isin(df_new["ligand_name"])]
            df = pd.concat([df_old, df_new], ignore_index=True).sort_values("best_affinity")
        else:
            df = df_new
    else:
        df = df_new

    df.to_csv(csv_path, index=False)
    return df
