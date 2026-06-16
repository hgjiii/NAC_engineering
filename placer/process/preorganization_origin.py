import warnings
import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBParser

from placer.process.structure import BACKBONE_ATOMS, get_coord

warnings.simplefilter("ignore", BiopythonWarning)

# def get_sidechain_coords(model, key_residues):
#     """
#     Extract sidechain heavy atom coords for key residues from a model.
#     Returns dict: {(chain, resid): np.ndarray of shape (n_atoms, 3)}
#     Atoms are ordered by atom name for consistent comparison across models.
#     """
#     coords = {}
#     for kr in key_residues:
#         try:
#             res = model[kr["chain"]][(" ", kr["resid"], " ")]
#         except KeyError:
#             continue  # residue missing in this model — skip

#         atoms = sorted(
#             [a for a in res.get_atoms()
#              if a.get_name() not in BACKBONE_ATOMS
#              and not a.get_name().startswith("H")
#              and a.element not in (None, "H")],
#             key=lambda a: a.get_name()  # sort by name for cross-model consistency
#         )
#         if atoms:
#             coords[(kr["chain"], kr["resid"])] = np.array([get_coord(a) for a in atoms])
#     return coords

def get_sidechain_coords(model, key_residues):
    """
    Extract sidechain heavy atom coords for key residues from a model.
    Returns dict: {(chain, resid): np.ndarray of shape (n_atoms, 3)}
    Missing or Gly-mutated residues are stored as empty array (shape (0, 3))
    to preserve key consistency across models.
    """
    coords = {}
    for kr in key_residues:
        key = (kr["chain"], kr["resid"])
        try:
            res = model[kr["chain"]][(" ", kr["resid"], " ")]
        except KeyError:
            coords[key] = np.empty((0, 3))  # residue missing — placeholder
            continue

        atoms = sorted(
            [a for a in res.get_atoms()
             if a.get_name() not in BACKBONE_ATOMS
             and not a.get_name().startswith("H")
             and a.element not in (None, "H")],
            key=lambda a: a.get_name()
        )
        # empty if Gly mutation (no sidechain) — placeholder to keep key
        coords[key] = np.array([get_coord(a) for a in atoms]) if atoms else np.empty((0, 3))

    return coords
    
def compute_rmsd(coords_a, coords_b):
    """
    Compute RMSD between sidechain coords of two models across all key residues.
    Only residues present in both models are included.
    """
    diffs = []
    for key in coords_a:
        if key not in coords_b:
            continue  # residue missing in one model — skip
        ca, cb = coords_a[key], coords_b[key]
        if ca.shape != cb.shape:
            continue  # atom count mismatch — skip
        diffs.append(((ca - cb) ** 2).sum(axis=1))  # squared dist per atom → (n_atoms,)

    if not diffs:
        return np.nan
    return float(np.sqrt(np.concatenate(diffs).mean()))  # RMSD over all atoms
    

def analyze_preorganization(models, key_residues, ref_idx = 0, verbose=False):
    n = len(models)

    # Step 1: extract sidechain coords for all models
    all_coords = [get_sidechain_coords(m, key_residues) for m in models]

    # Step 2: compute n x n pairwise RMSD matrix
    # symmetric matrix — compute upper triangle only, then mirror
    rmsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = compute_rmsd(all_coords[i], all_coords[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    # Step 3: reference model = model 0 (lowest pRMSD)
    ref_rmsds = rmsd_matrix[ref_idx][np.arange(n) != ref_idx]

    if verbose:
        print(f"=== Reference Model (Lowest pRMSD) ===")
        print(f"  Model index                            : {ref_idx + 1}")
        print(f"  Global mean pairwise RMSD (All pairs)  : {rmsd_matrix[np.triu_indices(n, 1)].mean():.3f} Å")
        print()
        print(f"=== Reference-model RMSD Statistics ===")
        print(f"  Mean   : {ref_rmsds.mean():.3f} Å")
        print(f"  Std    : {ref_rmsds.std():.3f} Å")
        print(f"  Median : {np.median(ref_rmsds):.3f} Å")
        print(f"  Min    : {ref_rmsds.min():.3f} Å")
        print(f"  Max    : {ref_rmsds.max():.3f} Å")
        print()
    return {
        "ref_idx": ref_idx,
        "ref_coords": all_coords[ref_idx],
        "rmsd_matrix": rmsd_matrix,
        "ref_rmsds": ref_rmsds,
    }