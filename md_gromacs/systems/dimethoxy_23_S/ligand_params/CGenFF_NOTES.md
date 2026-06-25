        # CGenFF / ParamChem notes — dimethoxy_23_S

        | Field | Value |
        |-------|-------|
        | SMILES source | `config.yaml` → `dimethoxy_23_S` |
        | CGenFF auto risk | **medium** |
        | Resname in PDB | LIG |
        | Chain | B |

        ## SMILES
        ```
        CC(=O)N[C@@H](c1c(OC)c(OC)ccc1)C2=C([O-])C3=CC=CC=C3OC2=O
        ```

        ## Submission checklist
        1. Upload `ligand.sdf` (or `ligand.mol2`) to [ParamChem](https://cgenff.umaryland.edu/) or CHARMM-GUI Ligand Reader.
        2. Set charge method consistent with pH 7.4 (coumarin refs: enolate).
        3. Download `.str` / `.rtf` + `.prm` (or GROMACS `.itp` from CHARMM-GUI).
        4. Rename ligand residue to **LIG** to match protein–ligand complex PDB.
        5. Merge `#include "ligand.itp"` into system topology after `#include "toppar_water_ions.str"` equivalent.

        ## Ligand-specific notes
        - Enolate + dimethoxy aniline; stereocenter S.
- Watch ether and amide torsions in ParamChem output.


        ## ParamChem fallback (if auto CGenFF fails)
        1. Draw or paste SMILES in ParamChem; run CGenFF.
        2. Inspect penalty scores — accept < 50 for production; reparameterize if higher.
        3. Export GROMACS format via CHARMM-GUI Ligand Reader & Modeler using the ParamChem mol2.
