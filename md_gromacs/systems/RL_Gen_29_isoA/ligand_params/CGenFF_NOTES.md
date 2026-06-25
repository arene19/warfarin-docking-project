        # CGenFF / ParamChem notes — RL_Gen_29_isoA

        | Field | Value |
        |-------|-------|
        | SMILES source | `config_master.yaml` → `RL_Gen_29_isoA` |
        | CGenFF auto risk | **high** |
        | Resname in PDB | LIG |
        | Chain | B |

        ## SMILES
        ```
        O=C(O)/C=C(\C[C@](O)(Cn1cncn1)C(F)(F)F)c1ccc2ccccc2c1
        ```

        ## Submission checklist
        1. Upload `ligand.sdf` (or `ligand.mol2`) to [ParamChem](https://cgenff.umaryland.edu/) or CHARMM-GUI Ligand Reader.
        2. Set charge method consistent with pH 7.4 (coumarin refs: enolate).
        3. Download `.str` / `.rtf` + `.prm` (or GROMACS `.itp` from CHARMM-GUI).
        4. Rename ligand residue to **LIG** to match protein–ligand complex PDB.
        5. Merge `#include "ligand.itp"` into system topology after `#include "toppar_water_ions.str"` equivalent.

        ## Ligand-specific notes
        - **HIGH RISK**: triazole, gem-difluoro, extended aromatic scaffold.
- Use isoA SMILES only (explicit @ stereochemistry).
- ParamChem manual submission recommended if CHARMM-GUI ligand reader fails.


        ## ParamChem fallback (if auto CGenFF fails)
        1. Draw or paste SMILES in ParamChem; run CGenFF.
        2. Inspect penalty scores — accept < 50 for production; reparameterize if higher.
        3. Export GROMACS format via CHARMM-GUI Ligand Reader & Modeler using the ParamChem mol2.
