        # CGenFF / ParamChem notes — RL_Gen_22

        | Field | Value |
        |-------|-------|
        | SMILES source | `config_master.yaml` → `RL_Gen_22_isoA` |
        | CGenFF auto risk | **high** |
        | Resname in PDB | LIG |
        | Chain | B |

        ## SMILES
        ```
        O=C(NCC(F)(F)F)O[C@H]1CC[C@@](CNC(=O)c2ccc(F)cc2Cl)(c2ccccc2)CC1
        ```

        ## Submission checklist
        1. Upload `ligand.sdf` (or `ligand.mol2`) to [ParamChem](https://cgenff.umaryland.edu/) or CHARMM-GUI Ligand Reader.
        2. Set charge method consistent with pH 7.4 (coumarin refs: enolate).
        3. Download `.str` / `.rtf` + `.prm` (or GROMACS `.itp` from CHARMM-GUI).
        4. Rename ligand residue to **LIG** to match protein–ligand complex PDB.
        5. Merge `#include "ligand.itp"` into system topology after `#include "toppar_water_ions.str"` equivalent.

        ## Ligand-specific notes
        - **HIGH RISK**: fluorinated benzamide, carbamate, multi-ring scaffold.
- Use RL_Gen_22_isoA SMILES (explicit @) — matches flat-dock MODEL 1 embed.
- Lipinski MW violation (487 Da) in ADMET profile; verify topology before long production.
- ParamChem fallback likely if CHARMM-GUI auto CGenFF penalizes torsions.


        ## ParamChem fallback (if auto CGenFF fails)
        1. Draw or paste SMILES in ParamChem; run CGenFF.
        2. Inspect penalty scores — accept < 50 for production; reparameterize if higher.
        3. Export GROMACS format via CHARMM-GUI Ligand Reader & Modeler using the ParamChem mol2.
