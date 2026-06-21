import numpy as np

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from process import structure
from process import symmetry

def build_ligand_ensemble(models):
    """
    Description:
        Stack each model's ligand heavy-atom coords into one composite-pose
        tensor, assuming a shared PLACER backbone frame (no alignment).

    Args:
        models: List of Bio.PDB Models, one per PLACER conformation.

    Returns:
        Tuple (ligand_info, coords_ensemble). coords_ensemble has shape
        (n_models, n_total_atoms, 3); ligand_info is a list of dicts with
        chain/resid/resname for each ligand in model 0 (concatenation order).
    """
    ref = structure.get_ligand_heavy_coords(models[0])
    ref_counts = [l["coords"].shape[0] for l in ref]
    ref_xyz = np.vstack([l["coords"] for l in ref])
    ref_info = [{"chain": l["chain"], "resid": l["resid"], "resname": l["resname"]}
                for l in ref]

    ensemble = [ref_xyz]
    for i, model in enumerate(models[1:], start=1):
        ligs = structure.get_ligand_heavy_coords(model)
        counts = [l["coords"].shape[0] for l in ligs]
        if counts != ref_counts:
            raise ValueError(f"Model {i}: ligand atom counts {counts} differ from "
                             f"reference {ref_counts}.")
        ensemble.append(np.vstack([l["coords"] for l in ligs]))
    return ref_info, np.stack(ensemble, axis=0)


def pairwise_rmsd_matrix(coords_ensemble):
    """
    Description:
        Compute the all-vs-all heavy-atom RMSD matrix of a ligand ensemble by
        direct coordinate subtraction, with no superposition.

    Args:
        coords_ensemble: ndarray of shape (n_models, n_atoms, 3) sharing a
            common backbone frame.

    Returns:
        Symmetric RMSD matrix of shape (n_models, n_models) in Angstrom,
        with zero diagonal.
    """
    n, na, _ = coords_ensemble.shape
    rmsd = np.zeros((n, n))
    for i in range(n):
        diff = coords_ensemble - coords_ensemble[i]
        rmsd[i] = np.sqrt((diff ** 2).sum(axis=(1, 2)) / na)
    return rmsd


def cluster_poses(rmsd_mat, cutoff=2.0, linkage_method="average"):
    """
    Description:
        Agglomerative hierarchical clustering on a precomputed RMSD matrix,
        cut at a fixed distance threshold to produce flat clusters.

    Args:
        rmsd_mat: Symmetric pairwise RMSD matrix (n, n) in Angstrom.
        cutoff: Distance threshold; pairs whose merge height is <= cutoff
            are placed in the same flat cluster.
        linkage_method: scipy linkage method ("average", "complete", ...).

    Returns:
        ndarray of length n with 1-indexed integer cluster labels.
    """
    condensed = squareform(rmsd_mat, checks=False)
    Z = linkage(condensed, method=linkage_method)
    return fcluster(Z, t=cutoff, criterion="distance")


def summarize_clusters(labels, rmsd_mat):
    """
    Description:
        For each cluster, compute size, occupancy, medoid frame (real member
        with minimum mean RMSD to the others), and maximum intra-cluster RMSD.

    Args:
        labels: Cluster labels from cluster_poses.
        rmsd_mat: Same pairwise RMSD matrix used during clustering.

    Returns:
        List of dicts sorted by descending size, each with keys:
            cluster_id, size, occupancy, medoid_idx, poses,
            medoid_mean_rmsd, max_intra_rmsd.
    """
    n_total = len(labels)
    out = []
    for cid in sorted(set(labels)):
        poses = np.where(labels == cid)[0]
        sub = rmsd_mat[np.ix_(poses, poses)]
        mean_dists = sub.mean(axis=1)
        medoid_local = int(np.argmin(mean_dists))
        out.append({
            "cluster_id": int(cid),
            "size": int(len(poses)),
            "occupancy": len(poses) / n_total,
            "medoid_idx": int(poses[medoid_local]),
            "poses": poses.tolist(),
            "medoid_mean_rmsd": float(mean_dists[medoid_local]),
            "max_intra_rmsd": float(sub.max()),
        })
    out.sort(key=lambda x: -x["size"])
    return out


def analyze_ligand_pose_clusters(models, rmsd_cutoff=2.0, linkage_method="average", symmetric = False):
    """
    Description:
        Cluster ligand poses in a PLACER multi-model PDB by pairwise heavy-atom
        RMSD and report per-cluster occupancy and medoid frames.

    Args:
        models: List of Bio.PDB Models (PLACER output).
        rmsd_cutoff: RMSD threshold (A) for flat clustering.
        linkage_method: scipy linkage method.

    Returns:
        Dict with keys: ligand_info, coords_ensemble, rmsd_matrix, labels, summary.
    """
    lig_info, coords_ens = build_ligand_ensemble(models)

    if symmetric:
        rmsd_mat = symmetry.pairwise_rmsd_matrix(coords_ens, models) # Temporal line to reflect symmetry of adipic acid for clustering ligand
    else:
        rmsd_mat = pairwise_rmsd_matrix(coords_ens) # This line is the original function, but temporarily, change this line to reflect symmetry of adipic acid for clustering ligand pose
    labels = cluster_poses(rmsd_mat, cutoff=rmsd_cutoff, linkage_method=linkage_method)
    summary = summarize_clusters(labels, rmsd_mat)
    return {
        "ligand_info":     lig_info,
        "coords_ensemble": coords_ens,
        "rmsd_matrix":     rmsd_mat,
        "labels":          labels,
        "summary":         summary,
    }