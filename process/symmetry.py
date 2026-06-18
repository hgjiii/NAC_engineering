import numpy as np

from process import structure


ADIPIC_ACID_SYM = [("C1","C2"), ("C2","C1"),("C3","C5"), ("C5","C3"),("C4","C6"), 
                   ("C6","C4"),("O1","O3"), ("O3","O1"),("O2","O4"), ("O4","O2")]
 
# ADIPIC_ACID_SYM = [("C4","C6"), ("C5","C3"),("C1","C2"), ("C2","C1"),("C3","C5"), 
#                    ("C6","C4"),("O1","O3"), ("O3","O1"),("O2","O4"), ("O4","O2")]

def pairwise_rmsd_matrix(coords_ensemble, models=None):
    """
    Description:
        Compute the all-vs-all heavy-atom RMSD matrix of a ligand ensemble by
        direct coordinate subtraction, with no superposition.

    Args:
        coords_ensemble: ndarray of shape (n_models, n_atoms, 3) sharing a
            common backbone frame.
        models: Optional list of Bio.PDB Models (only used for APA symmetry).

    Returns:
        Symmetric RMSD matrix of shape (n_models, n_models) in Angstrom,
        with zero diagonal.
    """
    n, na, _ = coords_ensemble.shape

    # ---- TEMPORARY: APA end-flip symmetry. Remove this block when generalized. ----
    perms = [np.arange(na)]
    if models is not None:
        names, idx = [], 0
        for lig in structure.get_ligand_heavy_coords(models[0]):
            for chain in models[0].get_chains():
                for res in chain.get_residues():
                    if res.get_resname().strip() == lig["resname"] and res.id[1] == lig["resid"]:
                        for a in res.get_atoms():
                            if a.get_name().startswith("H") or a.element in (None, "H"):
                                continue
                            names.append((lig["resname"], a.get_name(), idx))
                            idx += 1
        apa_idx = {n: i for rn, n, i in names if rn == "APA"}
        if apa_idx:
            flip = np.arange(na)
            for s, d in ADIPIC_ACID_SYM:
                if s in apa_idx and d in apa_idx:
                    flip[apa_idx[s]] = apa_idx[d]
            perms.append(flip)

    rmsd = np.zeros((n, n))
    for i in range(n):
        best = None
        for p in perms:
            diff = coords_ensemble - coords_ensemble[i][p]
            r = np.sqrt((diff ** 2).sum(axis=(1, 2)) / na)
            best = r if best is None else np.minimum(best, r)
        rmsd[i] = best
    return rmsd


def select_apa_reactive_carboxyl(ligands, atp_atom_names):
    """
    Description:
        Pick APA atom names of the carboxyl group closer to the ATP reference
        atoms, resolving APA's two-fold symmetry per model.

    Args:
        ligands: Ligand dict list from get_ligand_heavy_coords for one model.
        atp_atom_names: ATP atom names defining the reactive anchor.

    Returns:
        Set of APA atom names: either {"C4", "O1", "O2"} or {"C6", "O3", "O4"}.
    """
    apa = next((l for l in ligands if l["resname"] == "APA"), None)
    atp = next((l for l in ligands if l["resname"] == "ATP"), None)
    if apa is None or atp is None:
        raise ValueError("Both APA and ATP must be present.")

    target = set(atp_atom_names)
    atp_mask = np.array([n in target for n in atp["atoms"]])
    if atp_mask.sum() == 0:
        raise ValueError(f"ATP anchor atoms {target} not found.")
    atp_center = atp["coords"][atp_mask].mean(axis=0)

    groups = [{"C4", "O1", "O2"}, {"C6", "O3", "O4"}]
    dists = []
    for g in groups:
        mask = np.array([n in g for n in apa["atoms"]])
        if mask.sum() == 0:
            dists.append(np.inf)
            continue
        center = apa["coords"][mask].mean(axis=0)
        dists.append(np.linalg.norm(center - atp_center))

    return groups[int(np.argmin(dists))]


def resolve_symmetric_atom(ligand, atom_name, partner_xyz, sym_pairs=None):
    """
    Description:
        Pick the symmetry-equivalent atom of a ligand closer to partner_xyz.

    Args:
        ligand: Ligand dict from structure.get_ligand_heavy_coords.
        atom_name: Requested ligand atom name.
        partner_xyz: Reference 3D point.
        sym_pairs: List of (atom_name, atom_name) swap tuples; None disables.

    Returns:
        Effective atom name closer to partner_xyz, or input atom_name unchanged.
    """
    if sym_pairs is None:
        return atom_name
    candidates = {atom_name} | {b for a, b in sym_pairs if a == atom_name}
    candidates &= set(ligand["atoms"])
    if len(candidates) <= 1:
        return atom_name
    best_name, best_d = atom_name, np.inf
    for n in candidates:
        xyz = ligand["coords"][ligand["atoms"].index(n)]
        d = np.linalg.norm(xyz - partner_xyz)
        if d < best_d:
            best_d, best_name = d, n
    return best_name