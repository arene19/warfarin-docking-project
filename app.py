"""
app.py
Streamlit graphical interface for the Anticoagulant Docking Pipeline.
Features:
- Master Config manager to allow dynamic target/ligand toggling via checkboxes.
- Sidebar Cache Control to force clean OpenMM re-protonation runs on-demand.
- Automatic coordinate preservation for forced-coordinate targets (HSA).
- Separated post-docking suite to generate publication figures and ChimeraX assets on demand.
- Clean file location mappings for generated outputs.
"""

import os
import sys
import time
import yaml
import shutil
import subprocess
import pandas as pd
import streamlit as st
import glob
from pathlib import Path

from receptor_validation import (
    validate_receptor,
    suggest_ref_ligand,
    format_report_text,
)
from enantiomer_tools import (
    generate_rs_enantiomer_pair,
    canonicalize_ligand_smiles,
    format_stereocenters,
    count_chiral_centers,
)

st.set_page_config(
    page_title="Anticoagulant Docking Hub",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────
MASTER_CONFIG_PATH = "config_master.yaml"
CONFIG_PATH = "config.yaml"

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Academic Scientific Dark-Accent Theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ===== GOOGLE FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ===== DESIGN TOKENS ===== */
:root {
    --navy:          #0A2540;
    --navy-mid:      #163D62;
    --navy-light:    #20507E;
    --teal:          #13B5AA;
    --teal-dim:      #0D8C84;
    --teal-glow:     rgba(19,181,170,0.15);
    --gold:          #E8A628;
    --bg-app:        #F0F4F9;
    --bg-card:       #FFFFFF;
    --bg-elevated:   #F8FAFD;
    --border:        #D8E2EE;
    --border-mid:    #B4C5D9;
    --text-primary:  #0A2540;
    --text-secondary:#4A6080;
    --text-muted:    #8699B0;
    --font-serif:    'Source Serif 4', Georgia, serif;
    --font-sans:     'IBM Plex Sans', system-ui, sans-serif;
    --font-mono:     'IBM Plex Mono', 'Courier New', monospace;
    --radius-sm:     6px;
    --radius-md:     10px;
    --radius-lg:     14px;
    --shadow-sm:     0 1px 4px rgba(10,37,64,0.07);
    --shadow-md:     0 4px 16px rgba(10,37,64,0.11);
    --shadow-lg:     0 8px 32px rgba(10,37,64,0.14);
    /* Prevent OS/browser dark-mode from signalling to canvas-rendered widgets */
    color-scheme:    light;
}

/* ===== GLOBAL ===== */
.stApp {
    background-color: var(--bg-app) !important;
    font-family: var(--font-sans) !important;
}
.block-container {
    padding: 1.5rem 2.25rem 4rem 2.25rem !important;
    max-width: 1440px !important;
}
h1, h2, h3, h4 {
    font-family: var(--font-serif) !important;
    color: var(--navy) !important;
    letter-spacing: -0.015em;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #081D33 0%, var(--navy) 45%, var(--navy-mid) 100%) !important;
    border-right: 1px solid rgba(19,181,170,0.18) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown small,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] [data-testid="stToggle"] label,
[data-testid="stSidebar"] [data-testid="stToggle"] label p,
[data-testid="stSidebar"] [data-testid="stToggle"] label span,
[data-testid="stSidebar"] .stToggle label p,
[data-testid="stSidebar"] .stToggle label span,
[data-testid="stSidebar"] .stCheckbox label p,
[data-testid="stSidebar"] .stCheckbox label span {
    color: #E8F4FC !important;
    font-family: var(--font-sans) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption p {
    color: #B8D9EE !important;
    font-size: 0.78rem !important;
}
[data-testid="stSidebar"] .stToggle,
[data-testid="stSidebar"] [data-testid="stToggle"],
[data-testid="stSidebar"] .stCheckbox {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.45rem 0.65rem !important;
}
[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #D6E6F4 !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stNumberInput input {
    background-color: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    color: #40627A !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background-color: var(--teal) !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: #81E6D9 !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, var(--teal-dim), var(--teal)) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.10) !important;
    margin: 0.6rem 0 !important;
}

/* ===== HERO BANNER ===== */
.hero-banner {
    background: linear-gradient(135deg, #081D33 0%, var(--navy) 40%, var(--navy-mid) 80%, #1A527A 100%);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem 1.85rem 2.5rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(19,181,170,0.22);
    box-shadow: var(--shadow-lg);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(19,181,170,0.16) 0%, transparent 68%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -50px; left: 35%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(232,166,40,0.09) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(19,181,170,0.15);
    border: 1px solid rgba(19,181,170,0.4);
    color: #81E6D9;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 0.7rem;
}
.hero-title {
    font-family: var(--font-serif);
    font-size: 2.1rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.025em;
    line-height: 1.18;
}
.hero-subtitle {
    font-family: var(--font-sans);
    font-size: 0.88rem;
    color: rgba(190,212,232,0.88);
    margin: 0;
    line-height: 1.6;
    max-width: 560px;
    font-weight: 300;
}

/* ===== SECTION LABELS ===== */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.64rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--teal-dim);
    display: block;
    margin-bottom: 0.3rem;
}
.section-heading {
    font-family: var(--font-serif);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--navy);
    margin: 0 0 0.2rem 0;
}
.section-desc {
    font-family: var(--font-sans);
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 1rem;
    line-height: 1.55;
}

/* ===== SIDEBAR SECTION LABEL ===== */
.sb-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: rgba(19,181,170,0.75);
    display: block;
    margin: 0 0 0.5rem 0;
}

/* ===== METRIC CARDS ===== */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.3rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-sans) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-serif) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: var(--navy) !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font-sans) !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 0.75rem !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    padding: 0.85rem 1.3rem !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    transition: color 0.18s ease !important;
    margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--teal-dim) !important;
    border-bottom: 3px solid var(--teal) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    padding: 1.75rem !important;
}

/* ===== CONTAINERS WITH BORDER ===== */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius-md) !important;
    border-color: var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
    background: var(--bg-elevated) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"],
[data-testid="stVerticalBlockBorderWrapper"] label,
[data-testid="stVerticalBlockBorderWrapper"] p,
[data-testid="stVerticalBlockBorderWrapper"] span,
[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="select"]:disabled,
[data-testid="stVerticalBlockBorderWrapper"] input:disabled {
    opacity: 0.72 !important;
    background-color: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    -webkit-text-fill-color: var(--text-secondary) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] .stAlert {
    background-color: #E8F4FD !important;
    color: var(--text-primary) !important;
}

/* ===== BUTTONS ===== */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--teal-dim) 0%, var(--teal) 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 10px rgba(13,140,132,0.28) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(13,140,132,0.38) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0px) !important;
}
.stButton > button:not([kind="primary"]) {
    background: transparent !important;
    border: 1.5px solid var(--border-mid) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: var(--radius-sm) !important;
}

/* ===== LAUNCH ZONE ===== */
.launch-zone {
    background: linear-gradient(135deg, #081D33 0%, var(--navy) 55%, var(--navy-mid) 100%);
    border-radius: var(--radius-lg);
    padding: 1.6rem 2rem;
    border: 1px solid rgba(19,181,170,0.22);
    box-shadow: var(--shadow-md);
}
.launch-zone-title {
    font-family: var(--font-serif);
    font-size: 1.15rem;
    font-weight: 600;
    color: #FFFFFF;
    margin: 0 0 0.3rem 0;
}
.launch-zone-desc {
    font-family: var(--font-sans);
    font-size: 0.82rem;
    color: rgba(185,210,232,0.8);
    margin: 0 0 1.1rem 0;
    font-weight: 300;
    line-height: 1.5;
}

/* ===== ALERTS ===== */
.stAlert {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    border-left-width: 4px !important;
}

/* ===== TEXT AREA (console) ===== */
.stTextArea textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    background-color: #0C1B2A !important;
    color: #A2D8B0 !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid rgba(19,181,170,0.22) !important;
    line-height: 1.6 !important;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background: var(--bg-elevated) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}
.streamlit-expanderContent {
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
    background: var(--bg-elevated) !important;
}

/* ===== DIVIDER ===== */
hr {
    border-color: var(--border) !important;
    margin: 1.1rem 0 !important;
}

/* ===== TOOL CARD CONTENT ===== */
.tool-card-title {
    font-family: var(--font-serif);
    font-size: 1.0rem;
    font-weight: 600;
    color: var(--navy);
    margin: 0 0 0.35rem 0;
}
.tool-card-body {
    font-family: var(--font-sans);
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.6;
    margin-bottom: 0.85rem;
    font-weight: 300;
}
.output-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.output-list li {
    font-family: var(--font-sans);
    font-size: 0.79rem;
    color: var(--text-secondary);
    padding: 0.22rem 0;
    display: flex;
    align-items: baseline;
    gap: 6px;
}
.output-list li::before {
    content: '→';
    color: var(--teal);
    font-weight: 700;
    font-size: 0.82rem;
    flex-shrink: 0;
}
.output-list code {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    background: rgba(19,181,170,0.08);
    border: 1px solid rgba(19,181,170,0.18);
    color: var(--teal-dim);
    padding: 1px 5px;
    border-radius: 3px;
}

/* ===== STATUS DOT ===== */
.status-dot-on  { color: #34D399; font-size: 0.78rem; font-family: var(--font-sans); }
.status-dot-off { color: #8699B0; font-size: 0.78rem; font-family: var(--font-sans); }

/* ===== REPORT CARD LABEL ===== */
.report-card-label {
    font-family: var(--font-sans);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin-bottom: 0.55rem;
    display: block;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (ALL LOGIC PRESERVED EXACTLY)
# ─────────────────────────────────────────────────────────────────────
def load_config():
    """
    Loads master config if exists (containing checkbox state).
    Falls back to standard config.yaml on first load, otherwise default template.
    """
    if os.path.exists(MASTER_CONFIG_PATH):
        with open(MASTER_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f), True

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
            return data, False

    # Default fallback template
    return {
        "project_name": "Coumarin_Derivatives_Anticoagulant_Study",
        "docking_params": {"exhaustiveness": 32, "num_modes": 10, "energy_range": 3.0,
                            "min_rmsd": 1.0, "num_conformers": 20, "n_cpu": None, "dock_timeout_s": 900},
        "receptors": {},
        "ligands": {}
    }, False


def run_subtask_with_logs(script_name: str, log_placeholder, extra_args=None) -> int:
    """
    Spawns a python script in a sanitized background thread,
    streaming output in real-time to a Streamlit code container.
    Optional extra_args are appended to the command line.
    """
    working_env = os.environ.copy()
    working_env["PYTHONUNBUFFERED"] = "1"
    working_env.pop("VIRTUAL_ENV", None)
    working_env.pop("PYTHONPATH", None)
    working_env["PATH"] = "/usr/bin:/usr/local/bin:/bin:" + working_env.get("PATH", "")

    command = [sys.executable, "-u", script_name]
    if extra_args:
        command.extend(str(a) for a in extra_args)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=working_env,
        bufsize=1
    )

    log_text = f"Initializing task sequence: {script_name}...\n"
    log_text += f"{'='*60}\n"
    log_placeholder.code(log_text)

    for line in iter(process.stdout.readline, ""):
        log_text += line
        log_placeholder.code(log_text)

    process.stdout.close()
    return_code = process.wait()
    return return_code


# ─────────────────────────────────────────────────────────────────────
# RESULT ARCHIVING ("Start Fresh")
# ─────────────────────────────────────────────────────────────────────
ARCHIVE_ROOT = os.path.join("results", "archive")


def find_archivable_results() -> list:
    """Returns the list of existing accumulating result artifacts that a fresh-start would archive."""
    candidates = sorted(glob.glob("results/docked_poses/*/*_screening_results.csv"))
    candidates += [
        "results/interaction_profile.csv",
        "results/admet_profile.csv",
        "results/docking_summary_report.txt",
        "results/docking_results.csv",
        "docking_results.csv",
    ]
    seen = set()
    existing = []
    for path in candidates:
        if path not in seen and os.path.isfile(path):
            seen.add(path)
            existing.append(path)
    return existing


def archive_previous_results():
    """Moves (never deletes) the accumulating result artifacts into a timestamped archive folder.

    Returns (archive_name, archive_dir, manifest) where manifest is a list of
    {"file", "archived_to", "size_kb"} dicts. Relative paths are preserved inside the archive.
    """
    archive_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    archive_dir = os.path.join(ARCHIVE_ROOT, archive_name)

    manifest = []
    for src in find_archivable_results():
        dest = os.path.join(archive_dir, src)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        size_kb = round(os.path.getsize(src) / 1024.0, 1)
        shutil.move(src, dest)
        manifest.append({"file": src, "archived_to": dest, "size_kb": size_kb})
    return archive_name, archive_dir, manifest


def find_ligand_cache_files(ligand_name: str) -> list[dict]:
    """Locate on-disk prep files, docked poses, and CSV rows for one ligand."""
    ligand_name = str(ligand_name).strip()
    if not ligand_name:
        return []

    entries: list[dict] = []
    seen_paths: set[str] = set()

    def _add(category: str, path: str, action: str = "delete_file") -> None:
        if path in seen_paths or not os.path.isfile(path):
            return
        seen_paths.add(path)
        entries.append({
            "category": category,
            "path": path,
            "action": action,
            "size_kb": round(os.path.getsize(path) / 1024.0, 1),
        })

    for label, path in [
        ("Input PDBQT", f"pdbqt_ligands/{ligand_name}.pdbqt"),
        ("Conformer SDF", f"results/ligands/{ligand_name}.sdf"),
        ("Mirror PDBQT", f"results/pdbqt_ligands/{ligand_name}.pdbqt"),
        ("Legacy SDF", f"ligands/{ligand_name}.sdf"),
        ("Legacy SDF", f"ligands/sdf/{ligand_name}.sdf"),
    ]:
        _add(label, path)

    for pose_path in sorted(glob.glob(f"results/docked_poses/*/{ligand_name}_*_docked.pdbqt")):
        target = Path(pose_path).parent.name
        _add(f"Docked pose ({target})", pose_path)

    for csv_path in sorted(glob.glob("results/docked_poses/*/*_screening_results.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "ligand_name" not in df.columns:
            continue
        if ligand_name not in set(df["ligand_name"].astype(str)):
            continue
        target = Path(csv_path).parent.name
        entries.append({
            "category": f"Screening row ({target})",
            "path": csv_path,
            "action": "remove_csv_row",
            "size_kb": None,
        })

    return entries


def clear_ligand_cache(
    ligand_name: str,
    *,
    clear_input: bool = True,
    clear_docked: bool = True,
    clear_csv_rows: bool = True,
) -> tuple[list[dict], list[str]]:
    """Delete selected cache artifacts for a ligand. Returns (manifest, errors)."""
    ligand_name = str(ligand_name).strip()
    manifest: list[dict] = []
    errors: list[str] = []

    for entry in find_ligand_cache_files(ligand_name):
        action = entry["action"]
        path = entry["path"]

        if action == "delete_file":
            if entry["category"].startswith("Docked pose"):
                if not clear_docked:
                    continue
            elif not clear_input:
                continue
            try:
                os.remove(path)
                manifest.append(entry)
            except OSError as exc:
                errors.append(f"{path}: {exc}")

        elif action == "remove_csv_row":
            if not clear_csv_rows:
                continue
            try:
                df = pd.read_csv(path)
                if ligand_name not in set(df["ligand_name"].astype(str)):
                    continue
                shutil.copy2(path, path + ".bak")
                df = df[df["ligand_name"].astype(str) != ligand_name]
                df.to_csv(path, index=False)
                manifest.append(entry)
            except Exception as exc:
                errors.append(f"{path}: {exc}")

    return manifest, errors


# ─────────────────────────────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────────────────────────────
master_data, is_master = load_config()


# ─────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-eyebrow"> AutoDock Vina · PLIP · ChimeraX · RDKit</div>
    <div class="hero-title">Molecular Docking Hub</div>
    <p class="hero-subtitle">
        Configure receptor targets and compound libraries, execute multi-target virtual screening,
        and analyze docking outcomes — from a single, unified interface.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR — GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sb-label">Project Identity</span>', unsafe_allow_html=True)
    project_name = st.text_input(
        "Project Name",
        value=master_data.get("project_name", "Docking_Study"),
        help="Identifier for this screening run. Used in output filenames and reports."
    )

    st.markdown("---")
    st.markdown('<span class="sb-label">Vina Search Parameters</span>', unsafe_allow_html=True)

    exhaustiveness = st.slider(
        "Search Exhaustiveness",
        min_value=4, max_value=64,
        value=master_data["docking_params"].get("exhaustiveness", 32),
        step=4,
        help="Controls global search depth. Higher values produce more accurate results at the cost of runtime."
    )

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        num_modes = st.number_input(
            "Output Modes",
            min_value=1, max_value=100,
            value=master_data["docking_params"].get("num_modes", 10),
            help="Maximum number of binding poses to output per ligand."
        )
    with col_sb2:
        energy_range = st.number_input(
            "ΔG Range",
            min_value=1.0, max_value=10.0,
            value=master_data["docking_params"].get("energy_range", 3.0),
            step=0.5,
            help="Energy window (kcal/mol) for reporting binding modes."
        )

    col_sb3, col_sb4 = st.columns(2)
    with col_sb3:
        min_rmsd = st.number_input(
            "Pose Min RMSD (Å)",
            min_value=0.0, max_value=5.0,
            value=float(master_data["docking_params"].get("min_rmsd", 1.0)),
            step=0.5,
            help="Minimum RMSD between reported poses. Lower values allow more similar poses."
        )
    with col_sb4:
        num_conformers = st.number_input(
            "Conformers / ligand",
            min_value=1, max_value=200,
            value=int(master_data["docking_params"].get("num_conformers", 20)),
            help="Number of 3D conformers embedded per ligand before docking."
        )

    col_sb5, col_sb6 = st.columns(2)
    with col_sb5:
        # 0 means 'use all available cores'. Stored as null in config when 0.
        _cfg_ncpu = master_data["docking_params"].get("n_cpu")
        n_cpu_ui = st.number_input(
            "CPU cores (0 = all)",
            min_value=0, max_value=256,
            value=int(_cfg_ncpu) if _cfg_ncpu else 0,
            help="CPU cores used per docking. 0 uses all available cores. Independent of exhaustiveness."
        )
    with col_sb6:
        dock_timeout_s = st.number_input(
            "Dock timeout (s)",
            min_value=30, max_value=7200,
            value=int(master_data["docking_params"].get("dock_timeout_s", 900)),
            step=30,
            help="Per-ligand wall-clock limit. A docking exceeding this is terminated and recorded as failed (rigid runs only)."
        )

    st.markdown("---")
    st.markdown('<span class="sb-label">Cache Control</span>', unsafe_allow_html=True)

    force_reprotonation = st.checkbox(
        "Force Re-protonation",
        value=False,
        help="If checked, deletes cached receptor files (.pdb/.pdbqt) before running to force a clean, highly optimized OpenMM protonation run."
    )
    if force_reprotonation:
        st.warning("Cached receptor structures will be cleared on next run.")

    st.markdown("---")
    st.markdown('<span class="sb-label">Docking Mode</span>', unsafe_allow_html=True)

    enable_flexible = st.toggle(
        "Flexible side-chain docking",
        value=True,
        help="When on, docking uses the per-receptor flexible residues defined in the Receptor table. "
             "When off, all receptors dock rigidly — faster, and enables Vina affinity-map reuse across ligands. "
             "Your flexible-residue values stay saved in the table either way."
    )
    if enable_flexible:
        st.caption("Flexible residues from the Receptor table will be used where defined.")
    else:
        st.caption("Rigid docking for all receptors this run (flexible residues ignored).")

    st.markdown("---")
    st.markdown('<span class="sb-label">Interaction Analysis</span>', unsafe_allow_html=True)

    enable_plip = st.toggle(
        "PLIP Interaction Profiling",
        value=False,
        help="When on, runs per-ligand PLIP profiling and the aggregated interaction heatmap after docking (passes --plip to the pipeline). Off by default for faster runs."
    )
    if enable_plip:
        st.caption("PLIP profiling will run after docking and refresh results/interaction_profile.csv.")
    else:
        st.caption("PLIP profiling is off — docking will skip interaction analysis.")

    report_current_only = st.toggle(
        "Summary report: this run only",
        value=True,
        help="When on, the docking summary report and interaction heatmap show only the ligands docked in this run. "
             "When off, they include every ligand ever docked (full history). Raw result CSVs always keep full history either way."
    )

    st.markdown("---")
    st.markdown('<span class="sb-label">Environment Status</span>', unsafe_allow_html=True)

    _status_files = [
        ("config.yaml",           CONFIG_PATH),
        ("config_master.yaml",    MASTER_CONFIG_PATH),
        ("Summary Report",        "results/docking_summary_report.txt"),
        ("ADMET Profile",         "results/admet_profile.csv"),
        ("PLIP Profile",          "results/interaction_profile.csv"),
    ]
    for _label, _path in _status_files:
        _exists = os.path.exists(_path)
        _cls = "status-dot-on" if _exists else "status-dot-off"
        _icon = "●" if _exists else "○"
        st.markdown(
            f'<p class="{_cls}">{_icon}&nbsp; {_label}</p>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────
# DASHBOARD METRICS ROW
# ─────────────────────────────────────────────────────────────────────
_n_rec_total  = len(master_data.get("receptors", {}))
_n_rec_active = sum(
    1 for v in master_data.get("receptors", {}).values()
    if (v.get("active", True) if isinstance(v, dict) else True)
)
_n_lig_total  = len(master_data.get("ligands", {}))
_n_lig_active = sum(
    1 for v in master_data.get("ligands", {}).values()
    if (v.get("active", True) if isinstance(v, dict) else True)
)

_mc1, _mc2, _mc3, _mc4 = st.columns(4)
with _mc1:
    st.metric(
        label="Active Targets",
        value=str(_n_rec_active),
        delta=f"{_n_rec_total} configured"
    )
with _mc2:
    st.metric(
        label="Active Compounds",
        value=str(_n_lig_active),
        delta=f"{_n_lig_total} in library"
    )
with _mc3:
    st.metric(
        label="Exhaustiveness",
        value=str(exhaustiveness),
        delta=f"±{energy_range} kcal window"
    )
with _mc4:
    st.metric(
        label="Output Poses",
        value=str(num_modes),
        delta="per ligand · target pair"
    )

st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# MAIN WORKSPACE TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Target Proteins",
    "Ligand Library",
    "Results & Reports"
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — TARGET PROTEINS
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="section-label">Receptor Configuration</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Target Protein Registry</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Toggle the <b>Active</b> column to include or exclude receptors from '
        'the next docking run without losing parameter values. Add new targets as table rows.</p>',
        unsafe_allow_html=True
    )

    # Map nested dict structure to flat dataframe list
    receptors_list = []
    receptors_dict = master_data.get("receptors", {})

    for name, info in receptors_dict.items():
        receptors_list.append({
            "Active":              info.get("active", True) if is_master else True,
            "Target Name":         name,
            "PDB ID":              info.get("pdb_id", ""),
            "Chain":               info.get("chain", "A"),
            "Native Ligand Res":   info.get("native_ligand_resname", "SWF"),
            "Padding (Å)":         float(info.get("padding", 6.0)),
            "Flexible Residues":   ", ".join(info.get("flexible_residues", []))
        })

    df_receptors = pd.DataFrame(receptors_list) if receptors_list else pd.DataFrame(
        columns=["Active", "Target Name", "PDB ID", "Chain", "Native Ligand Res", "Padding (Å)", "Flexible Residues"]
    )

    edited_receptors = st.data_editor(
        df_receptors,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Active": st.column_config.CheckboxColumn(
                "Active",
                help="Select to include in the docking run",
                default=True,
            )
        },
        key="editor_receptors"
    )

    with st.expander("ℹ️  Grid box coordinates & forced-center logic"):
        st.info(
            "**force_center** and **force_size** parameters stored in `config_master.yaml` "
            "define explicit Vina grid box coordinates, bypassing automatic native-ligand detection. "
            "These values are read from the master config on every save and are never overwritten by this editor. "
            "\n\n"
            "**Flexible residues** should be comma-separated chain:residue pairs — e.g. `A:217, A:276, A:269`."
        )

    st.markdown("---")
    st.markdown('<span class="section-label">Setup Wizard</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Add & Validate Targets</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Use <b>Quick Add</b> to register a new PDB target in one step. '
        'Then run <b>Validate</b> to confirm the native ligand, grid box, flex residues, '
        'reference RMSD, and pocket contacts before a full screening run.</p>',
        unsafe_allow_html=True
    )

    _wiz_add, _wiz_val = st.columns(2, gap="medium")

    with _wiz_add:
        with st.container(border=True):
            st.markdown("**➕ Quick Add Target**")
            _new_name = st.text_input("Target name", placeholder="My_New_Target", key="qa_name")
            _new_pdb = st.text_input("PDB ID", placeholder="6WV3", key="qa_pdb").upper().strip()
            _c1, _c2 = st.columns(2)
            with _c1:
                _new_chain = st.text_input("Chain", value="A", key="qa_chain")
            with _c2:
                _new_pad = st.number_input("Padding (Å)", min_value=2.0, max_value=20.0, value=6.0, key="qa_pad")
            _new_native = st.text_input("Native ligand resname", placeholder="SWF", key="qa_native").upper().strip()
            _new_flex = st.text_input(
                "Flexible residues (optional)",
                placeholder="A:217, A:276",
                key="qa_flex",
                help="Comma-separated chain:residue pairs. Leave blank for rigid docking.",
            )
            if st.button("➕ Add target to registry", use_container_width=True, key="qa_add_btn"):
                if not _new_name or not _new_pdb or not _new_native:
                    st.error("Target name, PDB ID, and native ligand resname are required.")
                else:
                    _flex_parsed = [r.strip() for r in _new_flex.split(",") if r.strip()]
                    _entry = {
                        "pdb_id": _new_pdb,
                        "chain": _new_chain.strip() or "A",
                        "native_ligand_resname": _new_native,
                        "padding": float(_new_pad),
                        "flexible_residues": _flex_parsed,
                        "active": False,
                    }
                    _updated = dict(master_data.get("receptors", {}))
                    _updated[_new_name.strip()] = _entry
                    master_data["receptors"] = _updated
                    master_data["project_name"] = project_name
                    with open(MASTER_CONFIG_PATH, "w") as _f:
                        yaml.safe_dump(master_data, _f, default_flow_style=False)
                    st.success(f"Added **{_new_name.strip()}** to `{MASTER_CONFIG_PATH}` (inactive by default — enable Active in the table above).")
                    st.rerun()

    with _wiz_val:
        with st.container(border=True):
            st.markdown("**Validate Target Setup**")
            _val_targets = list(receptors_dict.keys()) or ["(add a target first)"]
            _val_target = st.selectbox("Target to validate", _val_targets, key="val_target_sel")

            _ref_options = ["Auto-detect reference ligand"]
            for _ln in (master_data.get("ligands") or {}):
                if str(_ln).endswith("_ref"):
                    _ref_options.append(_ln)
            _val_ref_choice = st.selectbox("Reference ligand (RMSD / contacts)", _ref_options, key="val_ref_sel")

            if st.button("Run validation checklist", type="primary", use_container_width=True, key="val_run_btn"):
                if _val_target == "(add a target first)":
                    st.warning("Add a target first.")
                else:
                    _rinfo = dict(receptors_dict.get(_val_target, {}))
                    if isinstance(_rinfo, dict):
                        _rinfo = {k: v for k, v in _rinfo.items() if k != "active"}
                    _ref_name, _ref_smi = None, None
                    if _val_ref_choice != "Auto-detect reference ligand":
                        _ref_name = _val_ref_choice
                        _raw_lig = (master_data.get("ligands") or {}).get(_ref_name, "")
                        _ref_smi = _raw_lig.get("smiles") if isinstance(_raw_lig, dict) else _raw_lig
                    else:
                        _ref_name, _ref_smi = suggest_ref_ligand(_val_target, master_data.get("ligands"))

                    with st.spinner(f"Validating {_val_target}..."):
                        _report = validate_receptor(
                            _val_target, _rinfo,
                            ref_ligand_name=_ref_name,
                            ref_ligand_smiles=_ref_smi,
                        )
                    st.session_state["last_validation"] = _report.to_dict()
                    st.session_state["last_validation_text"] = format_report_text(_report)

            if st.session_state.get("last_validation", {}).get("target_name") == _val_target:
                _rep = st.session_state["last_validation"]
                if _rep.get("overall_pass"):
                    st.success(f"Overall: PASS — {_val_target} is ready for docking.")
                else:
                    st.warning(f"Overall: needs attention — review failed checks below.")

                for _chk in _rep.get("checks", []):
                    _is_optional = str(_chk.get("check_id", "")).startswith("optional_")
                    _icon = "✅" if _chk.get("passed") else ("ℹ️" if _is_optional else "❌")
                    _label = _chk.get("label", "")
                    _detail = _chk.get("detail", "")
                    st.markdown(f"{_icon} **{_label}**")
                    if _detail:
                        st.caption(_detail)

                with st.expander("Full validation log"):
                    st.code(st.session_state.get("last_validation_text", ""))

                _has_rmsd = any(c.get("check_id") == "rmsd" for c in _rep.get("checks", []))
                if not _has_rmsd:
                    st.info(
                        "Tip: dock a reference ligand (name it `Something_ref`) against this target, "
                        "then re-run validation to get RMSD and pocket-contact checks."
                    )


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — LIGAND LIBRARY
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<span class="section-label">Compound Library</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Evaluated Compound Registry</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">Enable or disable individual compounds via the <b>Active</b> checkbox. '
        'SMILES strings are used to generate 3D conformers with RDKit/ETKDG prior to format conversion and docking.</p>',
        unsafe_allow_html=True
    )

    ligands_list = []
    ligands_dict = master_data.get("ligands", {})

    for name, info in ligands_dict.items():
        if isinstance(info, dict):
            smiles = info.get("smiles", "")
            active = info.get("active", True)
        else:
            smiles = str(info)
            active = True

        ligands_list.append({
            "Active":      active,
            "Ligand Name": name,
            "SMILES":      smiles
        })

    df_ligands = pd.DataFrame(ligands_list) if ligands_list else pd.DataFrame(
        columns=["Active", "Ligand Name", "SMILES"]
    )

    edited_ligands = st.data_editor(
        df_ligands,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Active": st.column_config.CheckboxColumn(
                "Active",
                help="Select to include in the docking run",
                default=True,
            )
        },
        key="editor_ligands"
    )

    with st.expander("Clear ligand cache", expanded=False):
        st.markdown(
            '<p class="section-desc">Remove generated files so the next pipeline run rebuilds conformers '
            'and/or re-docks from scratch. Does <b>not</b> remove the ligand from the library table.</p>',
            unsafe_allow_html=True
        )

        with st.container(border=True):
            _cache_names = sorted((master_data.get("ligands") or {}).keys())
            _cache_scope = st.radio(
                "Scope",
                options=["Single ligand", "All ligands in library"],
                horizontal=True,
                key="cache_scope",
            )
            _cache_all = _cache_scope.startswith("All")

            _cache_pick = st.selectbox(
                "Ligand",
                options=_cache_names or ["(no ligands in library)"],
                key="cache_ligand_pick",
                disabled=_cache_all,
                help="Pick one compound, or switch scope to All ligands above.",
            )
            if _cache_all:
                st.info(
                    "Applies the selected cache categories below to **every ligand** in the library "
                    f"({len(_cache_names)} compound(s)). Deletes generated SDF/PDBQT prep files, "
                    "docked pose PDBQTs, and/or screening CSV rows — but **does not** remove ligand "
                    "definitions from the library table. Untick **Screening CSV rows** to keep past "
                    "ΔG results. The next pipeline run will rebuild anything cleared."
                )

            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                _cache_input = st.checkbox(
                    "Input cache (SDF + PDBQT)",
                    value=False,
                    key="cache_clear_input",
                    help="Deletes pdbqt_ligands/, results/ligands/, results/pdbqt_ligands/ files.",
                )
            with _c2:
                _cache_docked = st.checkbox(
                    "Docked poses",
                    value=False,
                    key="cache_clear_docked",
                    help="Deletes results/docked_poses/{target}/{ligand}_*_docked.pdbqt for all targets.",
                )
            with _c3:
                _cache_csv = st.checkbox(
                    "Screening CSV rows",
                    value=False,
                    key="cache_clear_csv",
                    help="Removes this ligand's rows from *_screening_results.csv (creates .bak backup).",
                )

            _cache_any_category = _cache_input or _cache_docked or _cache_csv

            def _cache_entry_matches(entry: dict) -> bool:
                if entry["action"] == "delete_file":
                    if entry["category"].startswith("Docked pose"):
                        return _cache_docked
                    return _cache_input
                if entry["action"] == "remove_csv_row":
                    return _cache_csv
                return False

            _preview_rows: list[dict] = []
            if _cache_names and _cache_any_category:
                _preview_ligands = _cache_names if _cache_all else (
                    [_cache_pick] if _cache_pick in _cache_names else []
                )
                for _pl_name in _preview_ligands:
                    for _entry in find_ligand_cache_files(_pl_name):
                        if not _cache_entry_matches(_entry):
                            continue
                        _row = {
                            "Type": _entry["category"],
                            "Path": _entry["path"],
                            "Size (KB)": _entry["size_kb"] if _entry["size_kb"] is not None else "—",
                            "Action": "Delete file" if _entry["action"] == "delete_file" else "Remove CSV row",
                        }
                        if _cache_all:
                            _row = {"Ligand": _pl_name, **_row}
                        _preview_rows.append(_row)

            if not _cache_any_category:
                st.caption("Tick at least one cache category above to preview what would be cleared.")
            elif _preview_rows:
                _scope_label = f"{len(_cache_names)} ligands" if _cache_all else _cache_pick
                _preview_total = len(_preview_rows)
                _preview_cap = 100
                st.caption(
                    f"{_preview_total} item(s) match the selected options ({_scope_label})"
                    + (f" — showing first {_preview_cap}" if _preview_total > _preview_cap else "")
                    + ":"
                )
                st.dataframe(
                    pd.DataFrame(_preview_rows[:_preview_cap]),
                    width="stretch",
                    hide_index=True,
                )
            elif _cache_names and (_cache_all or _cache_pick in _cache_names):
                st.caption("No cache files match the selected options (already clear).")
            elif not _cache_names:
                st.caption("No ligands in the library yet.")

            _confirm_label = (
                f"I want to clear cache for **all ligands** ({len(_cache_names)} compounds)"
                if _cache_all
                else f"I want to clear cache for **{_cache_pick}**"
            )
            if _cache_all:
                st.info(
                    "Checking the box below confirms you want to delete the ticked cache categories "
                    "for **all** library ligands at once. This cannot be undone (CSV backups are saved as `.bak`)."
                )
            else:
                st.info(
                    "Checking the box below confirms you want to delete the ticked cache categories "
                    f"for **{_cache_pick}** only. Library SMILES entries are kept; only generated files "
                    "and/or screening rows are affected."
                )
            _cache_confirm = st.checkbox(
                _confirm_label,
                value=False,
                key="cache_confirm_all" if _cache_all else "cache_confirm_single",
            )

            _clear_btn_label = (
                "Clear cache for all ligands" if _cache_all else "Clear selected cache"
            )
            if st.button(_clear_btn_label, type="primary", use_container_width=True, key="cache_clear_btn"):
                if not _cache_names:
                    st.warning("No ligands in the library.")
                elif not (_cache_input or _cache_docked or _cache_csv):
                    st.warning("Select at least one cache category to clear.")
                elif not _cache_confirm:
                    st.warning("Check the confirmation box above first.")
                elif not _cache_all and _cache_pick not in _cache_names:
                    st.warning("No ligand selected.")
                else:
                    _targets = _cache_names if _cache_all else [_cache_pick]
                    _manifest: list[dict] = []
                    _errors: list[str] = []
                    for _tgt in _targets:
                        _m, _e = clear_ligand_cache(
                            _tgt,
                            clear_input=_cache_input,
                            clear_docked=_cache_docked,
                            clear_csv_rows=_cache_csv,
                        )
                        for _item in _m:
                            _manifest.append({**_item, "ligand": _tgt})
                        _errors.extend(_e)

                    if _manifest:
                        if _cache_all:
                            st.success(
                                f"Cleared {len(_manifest)} item(s) across **{len(_targets)}** ligands."
                            )
                        else:
                            st.success(f"Cleared {len(_manifest)} item(s) for **{_cache_pick}**.")
                        _cleared_df = pd.DataFrame([
                            {
                                **({"Ligand": m["ligand"]} if _cache_all else {}),
                                "Type": m["category"],
                                "Path": m["path"],
                            }
                            for m in _manifest
                        ])
                        st.dataframe(_cleared_df, width="stretch", hide_index=True)
                    else:
                        st.info("Nothing to clear — cache was already empty for those options.")
                    for _err in _errors:
                        st.error(_err)
                    if _manifest and not _errors:
                        st.rerun()

    with st.expander("ℹ️  SMILES format guide & naming conventions"):
        st.info(
            "Ligand names must be unique and contain no spaces — use underscores (`BENZ_R`, `p_nitro_S`). "
            "SMILES should encode stereochemistry explicitly using `@@` / `@` notation. "
            "For a pair of enantiomers, use matching scaffold names with `_R` / `_S` suffixes to enable "
            "automated stereoselectivity analysis during post-docking reporting."
        )

    with st.expander("Add compounds from SMILES", expanded=False):
        st.markdown(
            '<p class="section-desc">Choose <b>R/S enantiomer pair</b> for chiral scaffolds, or '
            '<b>Single ligand</b> for achiral compounds, racemates, or anything that should not be split into _R/_S.</p>',
            unsafe_allow_html=True
        )

        with st.container(border=True):
            _add_mode = st.radio(
                "Add mode",
                options=["R/S enantiomer pair", "Single ligand (no enantiomers)"],
                horizontal=True,
                key="lig_add_mode",
            )

            _en_col1, _en_col2 = st.columns([1, 2])
            with _en_col1:
                if _add_mode.startswith("R/S"):
                    _en_base = st.text_input(
                        "Base name (no _R/_S suffix)",
                        placeholder="m_nitro",
                        key="en_base_name",
                        help="Example: `m_nitro` → adds `m_nitro_R` and `m_nitro_S`.",
                    )
                else:
                    _en_base = st.text_input(
                        "Ligand name",
                        placeholder="RL_Gen_51",
                        key="single_lig_name",
                        help="Exact name as it will appear in the library (no _R/_S added).",
                    )
                _en_active = st.checkbox("Mark as Active", value=False, key="en_active")
                _en_overwrite = st.checkbox("Overwrite if name already exists", value=False, key="en_overwrite")
            with _en_col2:
                _en_smiles = st.text_area(
                    "SMILES",
                    placeholder="CC(=O)NC(C1=CC(=CC=C1)[N+](=O)[O-])C2=C([O-])C3=CC=CC=C3OC2=O",
                    height=100,
                    key="en_smiles",
                )
                if _en_smiles:
                    _n_chiral = count_chiral_centers(_en_smiles)
                    if _add_mode.startswith("R/S") and _n_chiral == 0:
                        st.info("No chiral centers detected — switch to **Single ligand** mode for this SMILES.")
                    elif not _add_mode.startswith("R/S") and _n_chiral > 0:
                        st.caption(
                            f"{_n_chiral} chiral center(s) detected — will be stored as one entry with the stereo you provide."
                        )

            _en_preview, _en_add = st.columns(2)
            with _en_preview:
                _do_preview = st.button("Preview", use_container_width=True, key="en_preview_btn")
            with _en_add:
                _btn_label = "➕ Generate & add R/S pair" if _add_mode.startswith("R/S") else "➕ Add single ligand"
                _do_add = st.button(_btn_label, type="primary", use_container_width=True, key="en_add_btn")

            if _do_preview or _do_add:
                if not _en_base or not _en_smiles:
                    st.error("Name and SMILES are both required.")
                elif _add_mode.startswith("R/S"):
                    try:
                        _pair = generate_rs_enantiomer_pair(_en_base, _en_smiles)
                        _preview_df = pd.DataFrame([
                            {
                                "Ligand Name": _pair.r_name,
                                "Name suffix": "R",
                                "All stereocenters (CIP)": format_stereocenters(_pair.r_chiral_centers),
                                "SMILES": _pair.r_smiles,
                            },
                            {
                                "Ligand Name": _pair.s_name,
                                "Name suffix": "S",
                                "All stereocenters (CIP)": format_stereocenters(_pair.s_chiral_centers),
                                "SMILES": _pair.s_smiles,
                            },
                        ])
                        st.dataframe(_preview_df, width="stretch", hide_index=True)
                        for _w in _pair.warnings:
                            st.warning(_w)

                        if _do_add:
                            _ligs = dict(master_data.get("ligands", {}))
                            _conflicts = [n for n in (_pair.r_name, _pair.s_name) if n in _ligs]
                            if _conflicts and not _en_overwrite:
                                st.error(
                                    f"Already in library: {', '.join(_conflicts)}. "
                                    "Enable overwrite or choose a different base name."
                                )
                            else:
                                _ligs[_pair.r_name] = {"smiles": _pair.r_smiles, "active": bool(_en_active)}
                                _ligs[_pair.s_name] = {"smiles": _pair.s_smiles, "active": bool(_en_active)}
                                master_data["ligands"] = _ligs
                                master_data["project_name"] = project_name
                                with open(MASTER_CONFIG_PATH, "w") as _f:
                                    yaml.safe_dump(master_data, _f, default_flow_style=False)
                                st.success(
                                    f"Added **{_pair.r_name}** and **{_pair.s_name}** to `{MASTER_CONFIG_PATH}`."
                                )
                                st.rerun()
                    except Exception as _en_err:
                        st.error(f"Could not generate enantiomer pair: {_en_err}")
                else:
                    try:
                        _canon = canonicalize_ligand_smiles(_en_smiles)
                        _single_name = str(_en_base).strip()
                        _preview_df = pd.DataFrame([
                            {
                                "Ligand Name": _single_name,
                                "Chiral centers": count_chiral_centers(_canon),
                                "SMILES": _canon,
                            }
                        ])
                        st.dataframe(_preview_df, width="stretch", hide_index=True)

                        if _do_add:
                            _ligs = dict(master_data.get("ligands", {}))
                            if _single_name in _ligs and not _en_overwrite:
                                st.error(
                                    f"**{_single_name}** already exists. Enable overwrite or pick another name."
                                )
                            else:
                                _ligs[_single_name] = {"smiles": _canon, "active": bool(_en_active)}
                                master_data["ligands"] = _ligs
                                master_data["project_name"] = project_name
                                with open(MASTER_CONFIG_PATH, "w") as _f:
                                    yaml.safe_dump(master_data, _f, default_flow_style=False)
                                st.success(f"Added **{_single_name}** to `{MASTER_CONFIG_PATH}`.")
                                st.rerun()
                    except Exception as _en_err:
                        st.error(f"Could not add ligand: {_en_err}")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS & REPORTS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="section-label">Pipeline Outputs</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Analytical Reports & Results Dashboard</p>', unsafe_allow_html=True)

    # ── Output availability status row ──────────────────────────────
    _report_path  = "results/docking_summary_report.txt"
    _profile_path = "results/interaction_profile.csv"
    _admet_path   = "results/admet_profile.csv"

    _has_report  = os.path.exists(_report_path)
    _has_profile = os.path.exists(_profile_path)
    _has_admet   = os.path.exists(_admet_path)

    _s1, _s2, _s3 = st.columns(3)
    with _s1:
        if _has_report:
            st.success("✅  Docking Summary Report")
        else:
            st.info("Docking Summary Report — pending")
    with _s2:
        if _has_profile:
            st.success("✅  PLIP Interaction Profile")
        else:
            st.info("PLIP Interaction Profile — pending")
    with _s3:
        if _has_admet:
            st.success("✅  ADMET Physicochemical Profile")
        else:
            st.info("ADMET Profile — pending")

    st.markdown("---")

    # ── Primary output panels ────────────────────────────────────────
    _col_rpt, _col_plip = st.columns([1, 1], gap="medium")

    with _col_rpt:
        with st.container(border=True):
            st.markdown('<span class="report-card-label">Docking Summary Report</span>', unsafe_allow_html=True)
            if _has_report:
                with open(_report_path, "r") as _f:
                    st.text_area(
                        "summary_text",
                        value=_f.read(),
                        height=350,
                        label_visibility="collapsed"
                    )
            else:
                st.info("Execute a simulation run to compile a summary report.")

    with _col_plip:
        with st.container(border=True):
            st.markdown('<span class="report-card-label">Structural Interaction Profiles (PLIP)</span>', unsafe_allow_html=True)
            if _has_profile:
                # --- UPDATED: replaced use_container_width with width='stretch' to clear canvas measuring collapse ---
                st.dataframe(pd.read_csv(_profile_path), width="stretch", height=350)
            else:
                st.info("PLIP profiling spreadsheet will appear here once the pipeline has run.")

    # ── ADMET profile (collapsible) ──────────────────────────────────
    if _has_admet:
        with st.expander("ADMET Physicochemical Profile — expand to view full table"):
            _admet_display = pd.read_csv(_admet_path)
            # --- UPDATED: replaced use_container_width with width='stretch' to clear canvas measuring collapse ---
            st.dataframe(_admet_display, width="stretch")

    # ── Post-Docking Analysis Suite ──────────────────────────────────
    st.markdown("---")
    st.markdown('<span class="section-label">On-Demand Tools</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Post-Docking Analysis Suite</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-desc">These tools operate independently of the main pipeline — '
        'trigger them any time to regenerate publication figures or ChimeraX visualization assets '
        'from existing docking results.</p>',
        unsafe_allow_html=True
    )

    # Shared ligand selector — scopes BOTH the figures generator and the ChimeraX renders.
    _available_ligands = sorted((master_data.get("ligands") or {}).keys())

    def _load_affinity_ranking():
        """Returns docked ligand names sorted by best (most negative) affinity."""
        frames = []
        for _csv in glob.glob("results/docked_poses/*/*_screening_results.csv"):
            try:
                _d = pd.read_csv(_csv)
                if {"ligand_name", "best_affinity"}.issubset(_d.columns):
                    frames.append(_d[["ligand_name", "best_affinity"]])
            except Exception:
                pass
        if not frames:
            return []
        _all = pd.concat(frames, ignore_index=True)
        _all["best_affinity"] = pd.to_numeric(_all["best_affinity"], errors="coerce")
        _all = _all.dropna(subset=["best_affinity"])
        _all = _all.sort_values("best_affinity").drop_duplicates("ligand_name")
        return _all["ligand_name"].astype(str).tolist()

    _ranked = _load_affinity_ranking()
    _pickable = sorted(set(_available_ligands) | set(_ranked))

    st.markdown("**Ligands to include** (figures & ChimeraX)")
    _sel_mode = st.radio(
        "Selection mode",
        options=["All available", "Top N by affinity", "Pick specific"],
        horizontal=True,
        label_visibility="collapsed",
        help="All = every ligand. Top N = the strongest binders from your docking results. "
             "Pick specific = start from an empty list and add only the ones you want."
    )

    _selected_ligands = []
    _ligand_args = None  # None => no --ligands filter (include everything the tools find)

    if _sel_mode == "All available":
        _selected_ligands = _pickable
        _ligand_args = None
        st.caption(f"Including all available ligands ({len(_pickable)}).")

    elif _sel_mode == "Top N by affinity":
        if _ranked:
            _n = st.slider("How many top binders?", 1, len(_ranked), min(5, len(_ranked)))
            _selected_ligands = _ranked[:_n]
            _ligand_args = ["--ligands", ",".join(_selected_ligands)]
            st.caption("Top binders: " + ", ".join(_selected_ligands))
        else:
            st.warning("No docked affinities found yet — run docking first, or use another mode.")

    else:  # Pick specific
        _selected_ligands = st.multiselect(
            "Choose ligands",
            options=_pickable,
            default=[],
            placeholder="Start typing to add ligands…",
            label_visibility="collapsed",
        )
        if _selected_ligands:
            _ligand_args = ["--ligands", ",".join(_selected_ligands)]
            st.caption(f"Scoped to {len(_selected_ligands)} of {len(_pickable)} ligand(s).")
        else:
            st.caption("No ligands chosen yet — pick at least one above.")

    # Block the tools when the user is in 'Pick specific' mode but hasn't chosen any.
    _block_run = (_sel_mode == "Pick specific" and not _selected_ligands)

    _suite_col1, _suite_col2 = st.columns(2, gap="medium")

    with _suite_col1:
        with st.container(border=True):
            st.markdown('<p class="tool-card-title">Publication Figures Generator</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="tool-card-body">Renders 5 journal-ready figures: binding affinity heatmap, '
                'stereoselectivity analysis, ADMET properties table, residue-level interaction '
                'fingerprint, and a chemical space SAR map. Styled to match high-impact publication standards.</p>',
                unsafe_allow_html=True
            )

            if st.button("Generate Publication Figures", type="primary", use_container_width=True, disabled=_block_run):
                log_box = st.empty()
                with st.spinner("Rendering publication figures..."):
                    ret_code = run_subtask_with_logs("generate_publication_plots.py", log_box, extra_args=_ligand_args)
                    if ret_code == 0:
                        st.success("✅  Publication figures rendered successfully!")
                    else:
                        st.error("Error during figure generation. Review the log output above.")

            with st.expander("Output files & paths"):
                st.markdown("""
<ul class="output-list">
  <li><code>Publication Figures/publication_heatmap_*.png</code> — Binding Affinity Heatmap</li>
  <li><code>Publication Figures/publication_stereoselectivity_*.png</code> — Stereoselective Analysis</li>
  <li><code>Publication Figures/publication_admet_table_*.png</code> — ADMET Data Table</li>
  <li><code>Publication Figures/publication_residue_interactions_*.png</code> — Residue Fingerprint</li>
  <li><code>Publication Figures/publication_chemical_space.png</code> — Chemical Space (SAR) Map</li>
</ul>
""", unsafe_allow_html=True)

    with _suite_col2:
        with st.container(border=True):
            st.markdown('<p class="tool-card-title">ChimeraX Visualization Generator</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="tool-card-body">Generates ChimeraX session files (.cxs) and stripped '
                'ligand-only coordinate files (.pdb / .pdbqt) for every docked pose. Assets are '
                'named and organized per-target for direct import into ChimeraX or for manual inspection.</p>',
                unsafe_allow_html=True
            )

            if st.button("Build ChimeraX Sessions & PDB Files", type="primary", use_container_width=True, disabled=_block_run):
                log_box = st.empty()
                with st.spinner("Compiling structural visualization assets..."):
                    ret_code = run_subtask_with_logs("pymol_autovis.py", log_box, extra_args=_ligand_args)
                    if ret_code == 0:
                        st.success("✅  ChimeraX sessions and ligand coordinate files generated!")
                    else:
                        st.error("Error during ChimeraX visualization. Review the log output above.")

            with st.expander("Output files & paths"):
                st.markdown("""
<ul class="output-list">
  <li><code>results/visualizations/{Target}/{Compound}_ligand_only.cxs</code> — ChimeraX Session File</li>
  <li><code>results/visualizations/{Target}/{Compound}_ligand_only.png</code> — Ray-Traced Render</li>
  <li><code>results/visualizations/{Target}/{Compound}_ligand_only.pdb</code> — Clean Ligand Coordinates</li>
  <li><code>results/visualizations/{Target}/{Compound}_load.cxc</code> — Universal CXC Load Script</li>
</ul>
""", unsafe_allow_html=True)

    # ── Start Fresh — Archive accumulating results ───────────────────
    st.markdown("---")
    with st.container(border=True):
        st.markdown('<p class="tool-card-title">Start Fresh — Archive Previous Results</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="tool-card-body">Moves the accumulating result files (per-target screening CSVs, '
            'ADMET profile, PLIP interaction profile, summary report, and combined docking results) into a '
            'timestamped archive folder so your next run starts clean. Nothing is deleted — everything stays '
            'recoverable inside the archive.</p>',
            unsafe_allow_html=True
        )

        _archivable = find_archivable_results()
        if _archivable:
            st.caption(
                f"{len(_archivable)} file(s) eligible → will be saved under "
                f"`{ARCHIVE_ROOT}/run_<timestamp>/` (paths preserved)."
            )
            with st.expander("Preview files that will be archived"):
                for _f in _archivable:
                    _kb = round(os.path.getsize(_f) / 1024.0, 1)
                    st.markdown(f"- `{_f}` — {_kb} KB")
        else:
            st.caption("No accumulating result files found — nothing to archive yet.")

        if st.button("Archive Results & Start Fresh", use_container_width=True, disabled=not _archivable):
            _name, _dir, _manifest = archive_previous_results()
            _total_kb = round(sum(m["size_kb"] for m in _manifest), 1)

            st.success(f"✅  Archived {len(_manifest)} file(s) — your next run starts clean.")

            # Little infographic: where it went + under what name + how much
            _m1, _m2, _m3 = st.columns(3)
            _m1.metric("Files archived", len(_manifest))
            _m2.metric("Total size", f"{_total_kb:.1f} KB")
            _m3.metric("Archive name", _name)
            st.info(f"Saved to:  `{_dir}/`")

            _manifest_df = pd.DataFrame([
                {"Original location": m["file"], "Archived copy": m["archived_to"], "Size (KB)": m["size_kb"]}
                for m in _manifest
            ])
            st.dataframe(_manifest_df, width="stretch", hide_index=True)
            st.caption("Restore anytime by moving files back from the archive folder to their original location.")

        # Show where previous archives live, by name
        _existing_archives = sorted(glob.glob(os.path.join(ARCHIVE_ROOT, "run_*")), reverse=True)
        if _existing_archives:
            with st.expander(f"Existing archives ({len(_existing_archives)})"):
                for _arch in _existing_archives:
                    _file_count = len([p for p in glob.glob(os.path.join(_arch, "**", "*"), recursive=True) if os.path.isfile(p)])
                    st.markdown(f"- `{_arch}/` — **{os.path.basename(_arch)}** · {_file_count} file(s)")


# ─────────────────────────────────────────────────────────────────────
# EXECUTION ZONE
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")

st.markdown("""
<div class="launch-zone">
    <p class="launch-zone-title">Save Settings &amp; Execute Pipeline</p>
    <p class="launch-zone-desc">
        Writes both the master config (with checkbox states) and the active runtime config,
        then launches the full docking pipeline in an isolated background process with live console output.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

if st.button("Save Settings and Start Docking Pipeline", use_container_width=True, type="primary"):

    # Optional Cache Clearing block to force fresh OpenMM protonation runs
    if force_reprotonation:
        st.info("Clearing cached receptor structures to force clean re-protonation...")
        folders_to_clear = ["pdbqt_receptors", "proteins/protonated"]
        for folder in folders_to_clear:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        st.warning(f"Could not delete cached file {file_path}: {e}")

    # 1. Parse table selections and build MASTER and ACTIVE dicts
    master_receptors = {}
    active_receptors = {}
    for _, row in edited_receptors.iterrows():
        name = row.get("Target Name")
        if not name or pd.isna(name):
            continue
        name_str = str(name).strip()
        flex_res = [r.strip() for r in str(row.get("Flexible Residues", "")).split(",") if r.strip()]

        # --- FIXED: Lookup original parameters to preserve coordinates (force_center, force_size) ---
        original_info = master_data.get("receptors", {}).get(name_str, {})

        receptor_entry = {
            "pdb_id":               str(row.get("PDB ID", "")).strip(),
            "chain":                str(row.get("Chain", "A")).strip(),
            "native_ligand_resname":str(row.get("Native Ligand Res", "SWF")).strip(),
            "padding":              float(row.get("Padding (Å)", 6.0)),
            "flexible_residues":    flex_res,
            "active":               bool(row.get("Active", True))
        }

        # Merge manual search coordinates (like force_center/force_size) from original configuration!
        for key in ["force_center", "force_size", "base_pdb_id"]:
            if key in original_info:
                receptor_entry[key] = original_info[key]

        master_receptors[name_str] = receptor_entry

        if receptor_entry["active"]:
            active_entry = receptor_entry.copy()
            active_entry.pop("active")
            # Global flexible-docking toggle: when off, force rigid docking for the
            # run while keeping the residue list saved in the master config/table.
            if not enable_flexible:
                active_entry["flexible_residues"] = []
            active_receptors[name_str] = active_entry

    master_ligands = {}
    active_ligands = {}
    for _, row in edited_ligands.iterrows():
        name   = row.get("Ligand Name")
        smiles = row.get("SMILES")
        if not name or pd.isna(name) or not smiles or pd.isna(smiles):
            continue
        name_str   = str(name).strip()
        smiles_str = str(smiles).strip()
        active_bool = bool(row.get("Active", True))

        master_ligands[name_str] = {
            "smiles": smiles_str,
            "active": active_bool
        }

        if active_bool:
            active_ligands[name_str] = smiles_str

    # 0 cores in the UI means "use all available" -> stored as null in config.
    n_cpu_cfg = int(n_cpu_ui) if int(n_cpu_ui) > 0 else None
    docking_params_cfg = {
        "exhaustiveness": int(exhaustiveness),
        "num_modes":      int(num_modes),
        "energy_range":   float(energy_range),
        "min_rmsd":       float(min_rmsd),
        "num_conformers": int(num_conformers),
        "n_cpu":          n_cpu_cfg,
        "dock_timeout_s": int(dock_timeout_s)
    }

    # 2. Compile Master configuration & Standard configuration
    master_config = {
        "project_name": str(project_name),
        "docking_params": dict(docking_params_cfg),
        "receptors": master_receptors,
        "ligands":   master_ligands
    }

    active_config = {
        "project_name": str(project_name),
        "docking_params": dict(docking_params_cfg),
        "receptors": active_receptors,
        "ligands":   active_ligands
    }

    # 3. Write Master configuration and standard config to files
    with open(MASTER_CONFIG_PATH, "w") as f:
        yaml.safe_dump(master_config, f, default_flow_style=False)

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(active_config, f, default_flow_style=False)

    st.info("Config files updated. Launching isolated background docking thread...")

    log_placeholder = st.empty()
    log_text = "Starting docking pipeline simulation...\n"

    # 4. Pass the exact, working system environment directly to keep openbabel/plip fully functional
    working_env = os.environ.copy()
    working_env["PYTHONUNBUFFERED"] = "1"
    working_env.pop("VIRTUAL_ENV", None)
    working_env.pop("PYTHONPATH", None)
    working_env["PATH"] = "/usr/bin:/usr/local/bin:/bin:" + working_env.get("PATH", "")

    pipeline_cmd = [sys.executable, "-u", "main_pipeline.py"]
    if enable_plip:
        pipeline_cmd.append("--plip")
        st.info("PLIP interaction profiling enabled for this run.")
    if not report_current_only:
        pipeline_cmd.append("--report-all")
        st.info("Summary report will include the full docked history.")

    process = subprocess.Popen(
        pipeline_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=working_env,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ""):
        log_text += line
        log_placeholder.code(log_text)

    process.stdout.close()
    return_code = process.wait()

    if return_code == 0:
        st.success("✅  Pipeline completed successfully! Navigate to the **Results & Reports** tab to view outputs.")
        st.balloons()
        st.rerun()
    else:
        st.error("Pipeline crashed during execution. Review the terminal output above to diagnose the error.")