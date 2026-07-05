        # CGenFF / ParamChem notes — RL_Gen_45

        | Field | Value |
        |-------|-------|
        | SMILES source | `config_master.yaml` → `RL_Gen_45` |
        | CGenFF auto risk | **high** |
        | Resname in PDB | LIG |
        | Chain | B |

        ## SMILES
        ```
        CCOc1ccccc1CC(=O)OC1OC2OC3(C)CCC4C(C)CCC(C1C)C24OO3
        ```

        ## Submission checklist
        1. Upload `ligand.sdf` (or `ligand.mol2`) to [ParamChem](https://cgenff.umaryland.edu/) or CHARMM-GUI Ligand Reader.
        2. Set charge method consistent with pH 7.4 (coumarin refs: enolate).
        3. Download `.str` / `.rtf` + `.prm` (or GROMACS `.itp` from CHARMM-GUI).
        4. Rename ligand residue to **LIG** to match protein–ligand complex PDB.
        5. Merge `#include "ligand.itp"` into system topology after `#include "toppar_water_ions.str"` equivalent.

        ## Ligand-specific notes
        - **HIGH RISK**: spirocyclic / bridged peroxide-like scaffold; unusual for CGenFF.
- Flat-dock MODEL 1 pose; ParamChem manual submission strongly recommended.
- Inspect penalty scores carefully; consider 20 ns pilot only until ligand RMSD stable.


        ## ParamChem fallback (if auto CGenFF fails)
        1. Draw or paste SMILES in ParamChem; run CGenFF.
        2. Inspect penalty scores — accept < 50 for production; reparameterize if higher.
        3. Export GROMACS format via CHARMM-GUI Ligand Reader & Modeler using the ParamChem mol2.
