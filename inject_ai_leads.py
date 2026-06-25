import os
import re
import glob
import yaml
import pandas as pd

from config_utils import MASTER_CONFIG, sync_active_config


def find_latest_generation_csv():
    """Find the REINVENT4 generation CSV with the highest integer suffix."""
    candidates = glob.glob("REINVENT4/coumarin_generation_*.csv")
    if not candidates:
        raise FileNotFoundError(
            "No REINVENT4/coumarin_generation_*.csv files found."
        )

    pattern = re.compile(r"coumarin_generation_(\d+)\.csv$")
    numbered = []
    for path in candidates:
        match = pattern.search(os.path.basename(path))
        if match:
            numbered.append((int(match.group(1)), path))

    if not numbered:
        raise FileNotFoundError(
            "Found generation CSVs but none matched 'coumarin_generation_<int>.csv'."
        )

    latest_num, latest_path = max(numbered, key=lambda item: item[0])
    return latest_path, latest_num


def highest_rl_gen_index(ligands):
    """Return the highest integer suffix among existing RL_Gen_* ligand keys."""
    highest = 0
    pattern = re.compile(r"^RL_Gen_(\d+)$")
    for key in ligands:
        match = pattern.match(str(key))
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


def inject_ai_leads():
    # 1. Auto-detect the latest REINVENT generation CSV
    latest_csv, gen_num = find_latest_generation_csv()
    print(f"Reading latest generation (gen {gen_num}): {latest_csv}...")
    df = pd.read_csv(latest_csv)

    # 2. Extract Top 10 based on the GNN's predicted potency
    top_10 = df.sort_values(by="Pred_VKORC1_pXC50", ascending=False).head(10)

    # 3. Load your existing master config (fallback to config.yaml)
    config_path = MASTER_CONFIG
    if not os.path.exists(config_path):
        config_path = "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    if "ligands" not in config or config["ligands"] is None:
        config["ligands"] = {}

    # 4. Determine the next auto-incrementing index for RL_Gen_* names
    start_index = highest_rl_gen_index(config["ligands"]) + 1

    # Decide on the storage format based on existing entries
    use_dict_format = bool(config["ligands"]) and isinstance(
        list(config["ligands"].values())[0], dict
    )

    # 5. Inject the new top 10 leads with auto-incrementing names
    print(
        f"Injecting top 10 AI leads into the configuration "
        f"(starting at RL_Gen_{start_index:02d})..."
    )
    for offset, (_, row) in enumerate(top_10.iterrows()):
        ligand_name = f"RL_Gen_{start_index + offset:02d}"
        smiles = row["SMILES"]

        # Support both standard string format and the dict format used by the Streamlit state
        if use_dict_format:
            config["ligands"][ligand_name] = {"smiles": smiles, "active": True}
        else:
            config["ligands"][ligand_name] = smiles

    # 6. Save the updated config to both files
    with open(MASTER_CONFIG, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    sync_active_config()
    print("\nInjection complete! Master config updated; config.yaml synced (active ligands, flat SMILES).")


if __name__ == "__main__":
    inject_ai_leads()
