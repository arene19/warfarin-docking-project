"""Generate explicit R/S enantiomer SMILES pairs for the ligand library."""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


@dataclass
class EnantiomerPair:
    base_name: str
    r_name: str
    s_name: str
    r_smiles: str
    s_smiles: str
    r_chiral_centers: list[tuple[int, str]] = field(default_factory=list)
    s_chiral_centers: list[tuple[int, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def chiral_centers(self) -> list[tuple[int, str]]:
        """Backward-compatible alias for the R enantiomer centers."""
        return self.r_chiral_centers


def format_stereocenters(centers: list[tuple[int, str]]) -> str:
    """Human-readable list of all stereocenters, e.g. 'atom 4:R, atom 11:S'."""
    if not centers:
        return "none"
    return ", ".join(f"atom {idx}:{lab}" for idx, lab in sorted(centers))


def count_chiral_centers(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return 0
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return len(
        Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    )


def canonicalize_ligand_smiles(smiles: str) -> str:
    """Parse and return canonical isomeric SMILES for a single library entry."""
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError("Could not parse SMILES.")
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _invert_tetrahedral_stereo(mol: Chem.Mol) -> Chem.Mol:
    """Return the enantiomer by inverting all tetrahedral atom tags."""
    dup = Chem.Mol(mol)
    for atom in dup.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(dup, cleanIt=True, force=True)
    return dup


def _cip_labels(mol: Chem.Mol) -> list[tuple[int, str]]:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return [
        (idx, label)
        for idx, label in Chem.FindMolChiralCenters(
            mol, includeUnassigned=False, useLegacyImplementation=False
        )
        if label in ("R", "S")
    ]


def _primary_cip(mol: Chem.Mol) -> str | None:
    labels = _cip_labels(mol)
    if not labels:
        return None
    labels.sort(key=lambda x: x[0])
    return labels[0][1]


def _canon_smiles(mol: Chem.Mol) -> str:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _strip_suffix(name: str) -> str:
    for suffix in ("_R", "_S"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _first_defined_isomer(mol: Chem.Mol) -> Chem.Mol:
    """Return a fully assigned stereoisomer, using enumeration when needed."""
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    undefined = [idx for idx, lab in centers if lab == "?"]
    if not undefined:
        return Chem.Mol(mol)

    opts = StereoEnumerationOptions(onlyUnassigned=True, tryEmbedding=False, maxIsomers=64)
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    if not isomers:
        raise ValueError("Could not assign stereochemistry to input.")
    return isomers[0]


def generate_rs_enantiomer_pair(base_name: str, smiles: str) -> EnantiomerPair:
    """
    Build an R/S enantiomer pair from a SMILES string.

    The input may be stereo-undefined or either enantiomer. Output names follow
    the project convention: {base}_R and {base}_S, assigned by CIP rules on the
    lowest-index stereocenter (matches existing library entries like BENZ_R).
    """
    base_name = _strip_suffix(base_name.strip())
    if not base_name:
        raise ValueError("Base name is required.")
    if not smiles or not str(smiles).strip():
        raise ValueError("SMILES is required.")

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError("Could not parse SMILES.")

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    chiral_idxs = [idx for idx, _ in centers]
    if not chiral_idxs:
        raise ValueError(
            "No chiral centers detected — use **Single ligand** mode for achiral compounds, "
            "or the ligand table above."
        )

    warnings: list[str] = []
    if len(chiral_idxs) > 1:
        warnings.append(
            f"{len(chiral_idxs)} chiral centers detected. The _R / _S suffix still follows "
            f"the lowest-index center, but the preview shows CIP labels at **all** centers."
        )

    mol_a = _first_defined_isomer(mol)
    mol_b = _invert_tetrahedral_stereo(mol_a)
    if _canon_smiles(mol_a) == _canon_smiles(mol_b):
        raise ValueError("Input appears achiral or meso — R/S pair would be identical.")

    r_mol, s_mol = mol_a, mol_b
    r_cip = _primary_cip(r_mol)
    s_cip = _primary_cip(s_mol)
    if r_cip == "S" and s_cip == "R":
        r_mol, s_mol = s_mol, r_mol

    r_smiles = _canon_smiles(r_mol)
    s_smiles = _canon_smiles(s_mol)
    if r_smiles == s_smiles:
        raise ValueError("Generated R and S SMILES are identical.")

    return EnantiomerPair(
        base_name=base_name,
        r_name=f"{base_name}_R",
        s_name=f"{base_name}_S",
        r_smiles=r_smiles,
        s_smiles=s_smiles,
        r_chiral_centers=_cip_labels(r_mol),
        s_chiral_centers=_cip_labels(s_mol),
        warnings=warnings,
    )
