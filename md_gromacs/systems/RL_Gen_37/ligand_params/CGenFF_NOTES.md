        # CGenFF / ParamChem notes — RL_Gen_37

        | Field | Value |
        |-------|-------|
        | SMILES source | `config_master.yaml` → `RL_Gen_37_isoA` |
        | CGenFF auto risk | **high** |
        | Resname in PDB | LIG |
        | Chain | B |

        ## SMILES
        ```
        O=C1N(CC2CC2)CCC12CCN([C@H](c1ccccc1)c1ccc(F)cc1)CC2
        ```

        ## Submission checklist
        1. Upload `ligand.sdf` (or `ligand.mol2`) to [ParamChem](https://cgenff.umaryland.edu/) or CHARMM-GUI Ligand Reader.
        2. Set charge method consistent with pH 7.4 (coumarin refs: enolate).
        3. Download `.str` / `.rtf` + `.prm` (or GROMACS `.itp` from CHARMM-GUI).
        4. Rename ligand residue to **LIG** to match protein–ligand complex PDB.
        5. Merge `#include "ligand.itp"` into system topology after `#include "toppar_water_ions.str"` equivalent.

        ## Ligand-specific notes
        - **HIGH RISK** for automated CGenFF: spiro/piperidinone, cyclopropyl, benzhydryl-like center.
- Use RL_Gen_37_isoA SMILES (CIP R) — matches md_poses flat-dock MODEL 1 embed.
- **ParamChem fallback likely required**: upload ligand.sdf, download mol2+str, merge manually.
- If CGenFF penalizes many torsions, consider shortening production until topology validated.


        ## ParamChem fallback (if auto CGenFF fails)
        1. Draw or paste SMILES in ParamChem; run CGenFF.
        2. Inspect penalty scores — accept < 50 for production; reparameterize if higher.
        3. Export GROMACS format via CHARMM-GUI Ligand Reader & Modeler using the ParamChem mol2.
