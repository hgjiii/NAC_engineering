import warnings
import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBParser

from placer.process.structure import BACKBONE_ATOMS, get_coord
from placer.process.rotamer import (
    CHI_ATOMS, classify_rotamer_bin, get_chi_angles,
)

warnings.simplefilter("ignore", BiopythonWarning)


# =============================================================================
# RMSD-based preorganization
# =============================================================================

def get_sidechain_coords(model, key_residues):
    """
    Description:
        Extract sidechain heavy-atom coords for key residues from one model.
        Missing or Gly-mutated residues are stored as empty arrays to preserve
        key consistency across models.

    Args:
        model: Bio.PDB Model.
        key_residues: List of residue dicts with chain/resid/resname.

    Returns:
        Dict mapping (chain, resid) to ndarray of shape (n_atoms, 3); shape
        (0, 3) for missing residues or residues with no sidechain.
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
        coords[key] = np.array([get_coord(a) for a in atoms]) if atoms else np.empty((0, 3))

    return coords


def compute_rmsd(coords_a, coords_b):
    """
    Description:
        Compute RMSD between sidechain coords of two models across all shared
        key residues. Residues missing in either model or with mismatched atom
        count are skipped.

    Args:
        coords_a: Dict from get_sidechain_coords for model A.
        coords_b: Dict from get_sidechain_coords for model B.

    Returns:
        RMSD in Angstrom over all included atoms; np.nan if no comparable atoms.
    """
    diffs = []
    for key in coords_a:
        if key not in coords_b:
            continue
        ca, cb = coords_a[key], coords_b[key]
        if ca.shape != cb.shape:
            continue
        diffs.append(((ca - cb) ** 2).sum(axis=1))

    if not diffs:
        return np.nan
    return float(np.sqrt(np.concatenate(diffs).mean()))


def get_residue_rotamer_labels(models, kr):
    """
    Description:
        Collect rotamer-bin labels for one residue across all models.

    Args:
        models: List of Bio.PDB Models.
        kr: Key-residue dict with chain/resid/resname.

    Returns:
        Tuple (labels, k). labels has length equal to the number of models for
        which a label could be assigned (missing data skipped). k is 3 for
        chi1-only residues, 9 for chi1+chi2 residues, and 0 if the residue
        type has no chi (ALA/GLY/PRO).
    """
    resname = kr["resname"]
    if resname not in CHI_ATOMS:
        return [], 0
    n_chi = len(CHI_ATOMS[resname])
    k = 3 if n_chi == 1 else 9

    labels = []
    for m in models:
        chi1, chi2 = get_chi_angles(m, kr["chain"], kr["resid"])
        if chi1 is None:
            continue
        if n_chi == 1:
            labels.append(classify_rotamer_bin(chi1))
        else:
            if chi2 is None:
                continue
            labels.append((classify_rotamer_bin(chi1), classify_rotamer_bin(chi2)))
    return labels, k


def compute_rotamer_score(labels, k):
    """
    Description:
        Compute single-residue rotamer scores from a list of bin labels:
        dominant occupancy and normalized Shannon entropy.

    Args:
        labels: List of rotamer-bin labels for one residue across models.
        k: Bin cardinality (3 or 9).

    Returns:
        Dict with dominant_occupancy and entropy_norm; None if labels is empty.
    """
    if not labels or k == 0:
        return None
    n = len(labels)
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    occupancy = {lab: c / n for lab, c in counts.items()}
    dom_occ = max(occupancy.values())

    H = -sum(p * np.log(p) for p in occupancy.values() if p > 0)
    H_norm = H / np.log(k) if k > 1 else 0.0
    return {
        "dominant_occupancy": dom_occ,
        "entropy_norm": H_norm,
    }


def analyze_preorganization_rmsd(models, key_residues, ref_idx=0, verbose=False):
    """
    Description:
        RMSD-based preorganization metric. Build the all-vs-all sidechain
        RMSD matrix over the key-residue set and report the reference-model
        row as the preorganization indicator.

    Args:
        models: List of Bio.PDB Models (PLACER ensemble for one state).
        key_residues: List of residue dicts (chain/resid/resname).
        ref_idx: Index of the reference model used to extract ref_rmsds.
        verbose: Print summary if True.

    Returns:
        Dict with ref_idx, ref_coords, rmsd_matrix, ref_rmsds.
    """
    n = len(models)

    # Step 1: extract sidechain coords for all models
    all_coords = [get_sidechain_coords(m, key_residues) for m in models]

    # Step 2: compute n x n pairwise RMSD matrix
    rmsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = compute_rmsd(all_coords[i], all_coords[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    # Step 3: reference-model RMSDs
    ref_rmsds = rmsd_matrix[ref_idx][np.arange(n) != ref_idx]

    if verbose:
        print(f"=== Reference Model (RMSD-based) ===")
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


def analyze_preorganization_rotamer(models, key_residues, lock_threshold=0.7, verbose=False):
    """
    Description:
        Rotamer-based preorganization metric. Per-residue rotamer-bin
        occupancies and entropy aggregated into state-level scores over the
        given key-residue set.

    Args:
        models: List of Bio.PDB Models (PLACER ensemble for one state).
        key_residues: List of residue dicts (chain/resid/resname).
        lock_threshold: Dominant-occupancy cutoff used to count "locked" residues.
        verbose: Print summary if True.

    Returns:
        Dict with entropy_norm (1D ndarray over residues), dominant_occupancy
        (1D ndarray over residues), and fraction_locked (scalar).
    """
    # Step 1: collect per-residue rotamer scores
    entropy_norm = []
    dominant_occupancy = []
    for kr in key_residues:
        labels, k = get_residue_rotamer_labels(models, kr)
        score = compute_rotamer_score(labels, k)
        if score is None:
            continue
        entropy_norm.append(score["entropy_norm"])
        dominant_occupancy.append(score["dominant_occupancy"])
    entropy_norm = np.array(entropy_norm)
    dominant_occupancy = np.array(dominant_occupancy)

    # Step 2: aggregate scores over the residue set
    if len(dominant_occupancy) > 0:
        fraction_locked = float((dominant_occupancy >= lock_threshold).sum() / len(dominant_occupancy))
    else:
        fraction_locked = np.nan

    if verbose:
        print(f"=== Reference Set (Rotamer-based) ===")
        print(f"  Residues analyzed                      : {len(entropy_norm)} / {len(key_residues)}")
        print(f"  Lock threshold (dominant occupancy)    : {lock_threshold:.2f}")
        print()
        print(f"=== Reference-set Rotamer Statistics ===")
        if len(entropy_norm) > 0:
            print(f"  Mean   : {entropy_norm.mean():.3f} (entropy_norm), {dominant_occupancy.mean():.3f} (dom_occ)")
            print(f"  Std    : {entropy_norm.std():.3f} (entropy_norm), {dominant_occupancy.std():.3f} (dom_occ)")
            print(f"  Median : {np.median(entropy_norm):.3f} (entropy_norm), {np.median(dominant_occupancy):.3f} (dom_occ)")
            print(f"  Min    : {entropy_norm.min():.3f} (entropy_norm), {dominant_occupancy.min():.3f} (dom_occ)")
            print(f"  Max    : {entropy_norm.max():.3f} (entropy_norm), {dominant_occupancy.max():.3f} (dom_occ)")
            print(f"  Fraction locked : {fraction_locked:.3f}")
        else:
            print(f"  No analyzable residues.")
        print()
    return {
        "entropy_norm": entropy_norm,
        "dominant_occupancy": dominant_occupancy,
        "fraction_locked": fraction_locked,
    }