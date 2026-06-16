import numpy as np

from placer.process.structure import get_coord

# Standard residue chi-atom definitions (residue-level convention, IUPAC 2003)
CHI_ATOMS = {
    "ARG": [["N","CA","CB","CG"], ["CA","CB","CG","CD"]],
    "ASN": [["N","CA","CB","CG"], ["CA","CB","CG","OD1"]],
    "ASP": [["N","CA","CB","CG"], ["CA","CB","CG","OD1"]],
    "CYS": [["N","CA","CB","SG"]],
    "GLN": [["N","CA","CB","CG"], ["CA","CB","CG","CD"]],
    "GLU": [["N","CA","CB","CG"], ["CA","CB","CG","CD"]],
    "HIS": [["N","CA","CB","CG"], ["CA","CB","CG","ND1"]],
    "ILE": [["N","CA","CB","CG1"], ["CA","CB","CG1","CD1"]],
    "LEU": [["N","CA","CB","CG"], ["CA","CB","CG","CD1"]],
    "LYS": [["N","CA","CB","CG"], ["CA","CB","CG","CD"]],
    "MET": [["N","CA","CB","CG"], ["CA","CB","CG","SD"]],
    "PHE": [["N","CA","CB","CG"], ["CA","CB","CG","CD1"]],
    "SER": [["N","CA","CB","OG"]],
    "THR": [["N","CA","CB","OG1"]],
    "TRP": [["N","CA","CB","CG"], ["CA","CB","CG","CD1"]],
    "TYR": [["N","CA","CB","CG"], ["CA","CB","CG","CD1"]],
    "VAL": [["N","CA","CB","CG1"]],
}


def compute_dihedral(p1, p2, p3, p4):
    """
    Description:
        Compute the signed dihedral angle (degrees) defined by four points.

    Args:
        p1, p2, p3, p4: 3D coordinates as (3,) arrays.

    Returns:
        Dihedral angle in degrees, range (-180, 180].
    """
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.degrees(np.arctan2(y, x)))


def classify_rotamer_bin(angle):
    """
    Description:
        Classify a chi angle into g+/t/g- using IUPAC convention.

    Args:
        angle: Dihedral angle in degrees, range (-180, 180].

    Returns:
        One of "g+", "t", "g-".
    """
    if 0.0 <= angle < 120.0:
        return "g+"
    if angle >= 120.0 or angle < -120.0:
        return "t"
    return "g-"


def get_chi_angles(model, chain_id, resid):
    """
    Description:
        Compute chi1 (and chi2 if defined) for one residue in one model.

    Args:
        model: Bio.PDB Model.
        chain_id: Chain identifier.
        resid: Residue sequence number.

    Returns:
        Tuple (chi1, chi2). Either may be None if the residue has no such
        chi or if any required atom is missing.
    """
    try:
        res = model[chain_id][(" ", resid, " ")]
    except KeyError:
        return None, None
    resname = res.get_resname().strip()
    if resname not in CHI_ATOMS:
        return None, None

    atom_map = {a.get_name(): a for a in res.get_atoms()}
    angles = []
    for atom_names in CHI_ATOMS[resname]:
        if not all(n in atom_map for n in atom_names):
            angles.append(None)
            continue
        coords = [get_coord(atom_map[n]) for n in atom_names]
        angles.append(compute_dihedral(*coords))
    while len(angles) < 2:
        angles.append(None)
    return angles[0], angles[1]


def analyze_rotamer_clusters(models, key_residues):
    """
    Description:
        For each key residue, count rotamer-bin occupancies across all models
        using chi1 (and chi2 when defined).

    Args:
        models: List of Bio.PDB Models (PLACER ensemble).
        key_residues: List of residue dicts with chain/resid/resname.

    Returns:
        List of dicts, one per residue, each with chain, resid, resname,
        n_chi (1 or 2), counts (label -> count), occupancy (label -> fraction),
        dominant_label, dominant_occupancy.
    """
    n_total = len(models)
    out = []
    for kr in key_residues:
        resname = kr["resname"]
        if resname not in CHI_ATOMS:
            continue   # ALA / GLY / PRO etc., no chi analysis
        n_chi = len(CHI_ATOMS[resname])

        counts = {}
        for m in models:
            chi1, chi2 = get_chi_angles(m, kr["chain"], kr["resid"])
            if chi1 is None:
                continue
            label = classify_rotamer_bin(chi1)
            if n_chi == 2 and chi2 is not None:
                label = (label, classify_rotamer_bin(chi2))
            counts[label] = counts.get(label, 0) + 1

        if not counts:
            continue
        occupancy = {k: v / n_total for k, v in counts.items()}
        dom_label = max(counts, key=counts.get)
        out.append({
            "chain": kr["chain"],
            "resid": kr["resid"],
            "resname": resname,
            "n_chi": n_chi,
            "counts": counts,
            "occupancy": occupancy,
            "dominant_label": dom_label,
            "dominant_occupancy": occupancy[dom_label],
        })
    return out
