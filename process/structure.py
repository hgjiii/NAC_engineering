import warnings
import io
import numpy as np

from Bio import BiopythonWarning
from Bio.PDB import PDBParser

warnings.simplefilter("ignore", BiopythonWarning)

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
STANDARD_AA = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
}
EXCLUDE_HETATM = {"HOH", "WAT", "MG", "ZN", "CA", "NA", "CL", "SO4", "PO4"}


def load_models_from_pdb(filepath):

    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    models = []
    current_lines = []
    in_model = False

    with open(filepath) as f:
        for line in f:
            if line.startswith("MODEL"):
                in_model = True
                current_lines = [line]
            elif line.startswith("ENDMDL"):
                current_lines.append(line)
                block = "".join(current_lines)
                try:
                    s = parser.get_structure("m", io.StringIO(block))
                    model = list(s.get_models())[0]
                    models.append(model)
                except Exception as e:
                    print(f"Warning: failed to parse model {len(models) + 1} — {e}")
                current_lines = []
                in_model = False
            elif in_model:
                current_lines.append(line)

    return models


def get_coord(atom):
    return np.array(atom.get_vector().get_array())


def get_ligand_heavy_coords(model):
    """
    Return heavy atom coordinates for all HETATM residues (excluding water/ions).
    Returns a list of dicts: [{"chain", "resid", "resname", "coords", "atom_names"}, ...]
    """
    ligands = []
    for chain in model.get_chains():
        for res in chain.get_residues():
            if not res.id[0].startswith("H_"):
                continue
            resname = res.get_resname().strip()
            if resname in EXCLUDE_HETATM:
                continue
            heavy = [a for a in res.get_atoms()
                     if not a.get_name().startswith("H") and a.element not in (None, "H")]
            ligands.append({
                "chain":      chain.id,
                "resid":      res.id[1],
                "resname":    resname,
                "coords":     np.array([get_coord(a) for a in heavy]),
                "atoms": [a.get_name() for a in heavy],
            })
    if not ligands:
        raise ValueError("No ligand found. Check EXCLUDE_HETATM list.")
    return ligands


def filter_ligand_atoms(ligands, atoms):
    """
    Description:
        Keep only atoms whose name is in atom_names from each ligand dict,
        returning new dicts with coords (and atom_names) restricted accordingly.

    Args:
        ligands: List of ligand dicts from get_ligand_heavy_coords (must
            include the "atoms" field).
        atom_names: Iterable of atom names to keep (case-sensitive).

    Returns:
        New list of ligand dicts with coords filtered to the selected subset.
        Ligands containing none of the requested atoms are dropped.
    """
    target = set(atoms)
    out = []
    for lig in ligands:
        mask = [n in target for n in lig["atoms"]]
        if not any(mask):
            continue
        idx = np.array(mask)
        out.append({
            "chain": lig["chain"],
            "resid": lig["resid"],
            "resname": lig["resname"],
            "coords": lig["coords"][idx],
            "atoms": [n for n, m in zip(lig["atoms"], mask) if m],
        })
    return out

    
def min_dist_to_ligand(coord, lig_coords):
    """Return the minimum distance from a single coordinate to any ligand heavy atom."""
    return np.sqrt(((lig_coords - coord) ** 2).sum(axis=1)).min()


def find_key_residues(models, ligands_list, cutoff_1=4.0, cutoff_2=None):
    """
    Description:
        Identify key active-site residues within cutoff A of any ligand heavy atom,
        across one or more models. With multiple models, results are merged by
        (chain, resid), keeping the smallest min_sc_dist.

    Args:
        models: A single Bio.PDB Model or a list of Models.
        ligands_list: Ligand dicts from get_ligand_heavy_coords for a single Model,
            or a list of such ligand-dict lists, one per Model in models.
        cutoff_1: Primary sidechain-to-ligand distance cutoff (A).
        cutoff_2: Optional looser cutoff (A) used together with the CB-vs-CA rule.

    Returns:
        List of key-residue dicts sorted by ascending resid. Each dict contains
        chain, resid, resname, min_sc_dist, ca_dist, cb_dist, source_models.
    """
    # Normalize to list form so the rest of the function is uniform
    if not isinstance(models, list):
        models = [models]
        ligands_list = [ligands_list]
    if len(models) != len(ligands_list):
        raise ValueError("models and ligands_list must have the same length.")

    merged = {}
    for midx, (model, ligands) in enumerate(zip(models, ligands_list)):
        all_lig_coords = np.vstack([lig["coords"] for lig in ligands])
        lig_ids = {(lig["chain"], lig["resid"]) for lig in ligands}
        for chain in model.get_chains():
            for res in chain.get_residues():
                resname = res.get_resname().strip()
                if resname not in STANDARD_AA or resname == "GLY":
                    continue
                if (chain.id, res.id[1]) in lig_ids:
                    continue
                atom_map = {a.get_name(): a for a in res.get_atoms()}
                if "CA" not in atom_map or "CB" not in atom_map:
                    continue
                sc_coords = np.array([
                    get_coord(a) for name, a in atom_map.items()
                    if name not in BACKBONE_ATOMS
                    and not name.startswith("H")
                    and a.element not in (None, "H")
                ])
                if len(sc_coords) == 0:
                    continue
                min_sc_dist = np.sqrt(
                    ((sc_coords[:, np.newaxis, :] - all_lig_coords[np.newaxis, :, :]) ** 2)
                    .sum(axis=2)
                ).min()
                ca_dist = min_dist_to_ligand(get_coord(atom_map["CA"]), all_lig_coords)
                cb_dist = min_dist_to_ligand(get_coord(atom_map["CB"]), all_lig_coords)

                # Apply criteria
                if cutoff_2 is None:
                    if min_sc_dist > cutoff_1:
                        continue
                else:
                    if min_sc_dist > cutoff_2:
                        continue
                    if min_sc_dist > cutoff_1 and cb_dist >= ca_dist:
                        continue

                key = (chain.id, res.id[1])
                entry = {
                    "chain": chain.id,
                    "resid": res.id[1],
                    "resname": resname,
                    "min_sc_dist": round(float(min_sc_dist), 3),
                    "ca_dist": round(float(ca_dist), 3),
                    "cb_dist": round(float(cb_dist), 3),
                }
                if key not in merged or entry["min_sc_dist"] < merged[key]["min_sc_dist"]:
                    merged[key] = entry

    return sorted(merged.values(), key=lambda x: x["resid"])

    
def filter_ligand_atoms(ligands, atoms):
    """
    Description:
        Keep only atoms whose name is in atom_names from each ligand dict,
        returning new dicts with coords (and atom_names) restricted accordingly.

    Args:
        ligands: List of ligand dicts from get_ligand_heavy_coords (must
            include the "atoms" field).
        atom_names: Iterable of atom names to keep (case-sensitive).

    Returns:
        New list of ligand dicts with coords filtered to the selected subset.
        Ligands containing none of the requested atoms are dropped.
    """
    target = set(atoms)
    out = []
    for lig in ligands:
        mask = [n in target for n in lig["atoms"]]
        if not any(mask):
            continue
        idx = np.array(mask)
        out.append({
            "chain": lig["chain"],
            "resid": lig["resid"],
            "resname": lig["resname"],
            "coords": lig["coords"][idx],
            "atoms": [n for n, m in zip(lig["atoms"], mask) if m],
        })
    return out


def load_prmsd(csv_path):
    """
    Description:
        Read the per-model pRMSD column from a PLACER info CSV.

    Args:
        csv_path: Path to the PLACER ".csv" info file with a 'prmsd' column.

    Returns:
        List of per-model pRMSD values (floats), in model order.
    """
    prmsd_list = []
    with open(csv_path) as f:
        header = next(f).split(",")
        idx = header.index("prmsd")
        for line in f:
            prmsd_list.append(float(line.split(",")[idx]))
    return prmsd_list