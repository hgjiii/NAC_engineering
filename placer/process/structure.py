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



    
################################################################################################################################################################
# cluster.py 
################################################################################################################################################################

# def build_ligand_ensemble(models):
#     """
#     Description:
#         Concatenate heavy-atom coordinates of all ligands per model into a
#         single tensor, treating multi-ligand systems as one composite pose.
#         Assumes a shared backbone frame (PLACER output), so no alignment.

#     Args:
#         models: List of Bio.PDB Models, one per PLACER conformation.

#     Returns:
#         Tuple (ligand_info, coords_ensemble). coords_ensemble has shape
#         (n_models, n_total_atoms, 3); ligand_info is a list of dicts with
#         chain/resid/resname for each ligand in model 0 (concatenation order).
#     """
#     ref = get_ligand_heavy_coords(models[0])
#     ref_counts = [l["coords"].shape[0] for l in ref]
#     ref_xyz = np.vstack([l["coords"] for l in ref])
#     ref_info = [{"chain": l["chain"], "resid": l["resid"], "resname": l["resname"]}
#                 for l in ref]

#     ensemble = [ref_xyz]
#     for i, model in enumerate(models[1:], start=1):
#         ligs = get_ligand_heavy_coords(model)
#         counts = [l["coords"].shape[0] for l in ligs]
#         if counts != ref_counts:
#             raise ValueError(f"Model {i}: ligand atom counts {counts} differ from "
#                              f"reference {ref_counts}.")
#         ensemble.append(np.vstack([l["coords"] for l in ligs]))
#     return ref_info, np.stack(ensemble, axis=0)

###########################ORIGINAL##############################################################################################################################################
###########################ORIGINAL##############################################################################################################################################
###########################ORIGINAL##############################################################################################################################################

# def pairwise_rmsd_matrix(coords_ensemble):
#     """
#     Description:
#         Compute the all-vs-all heavy-atom RMSD matrix of a ligand ensemble by
#         direct coordinate subtraction, with no superposition.

#     Args:
#         coords_ensemble: ndarray of shape (n_models, n_atoms, 3) sharing a
#             common backbone frame.

#     Returns:
#         Symmetric RMSD matrix of shape (n_models, n_models) in Angstrom,
#         with zero diagonal.
#     """
#     n, na, _ = coords_ensemble.shape
#     rmsd = np.zeros((n, n))
#     for i in range(n):
#         diff = coords_ensemble - coords_ensemble[i]
#         rmsd[i] = np.sqrt((diff ** 2).sum(axis=(1, 2)) / na)
#     return rmsd
###########################ORIGINAL##############################################################################################################################################
###########################ORIGINAL##############################################################################################################################################
###########################ORIGINAL##############################################################################################################################################

####################################TEMPORARY######################################################################################################################################
####################################TEMPORARY######################################################################################################################################
####################################TEMPORARY######################################################################################################################################

# def pairwise_rmsd_matrix(coords_ensemble, models=None):
#     """
#     Description:
#         Compute the all-vs-all heavy-atom RMSD matrix of a ligand ensemble by
#         direct coordinate subtraction, with no superposition.

#     Args:
#         coords_ensemble: ndarray of shape (n_models, n_atoms, 3) sharing a
#             common backbone frame.
#         models: Optional list of Bio.PDB Models (only used for APA symmetry).

#     Returns:
#         Symmetric RMSD matrix of shape (n_models, n_models) in Angstrom,
#         with zero diagonal.
#     """
#     n, na, _ = coords_ensemble.shape

#     # ---- TEMPORARY: APA end-flip symmetry. Remove this block when generalized. ----
#     perms = [np.arange(na)]
#     if models is not None:
#         names, idx = [], 0
#         for lig in get_ligand_heavy_coords(models[0]):
#             for chain in models[0].get_chains():
#                 for res in chain.get_residues():
#                     if res.get_resname().strip() == lig["resname"] and res.id[1] == lig["resid"]:
#                         for a in res.get_atoms():
#                             if a.get_name().startswith("H") or a.element in (None, "H"):
#                                 continue
#                             names.append((lig["resname"], a.get_name(), idx))
#                             idx += 1
#         apa_idx = {n: i for rn, n, i in names if rn == "APA"}
#         if apa_idx:
#             flip = np.arange(na)
#             for s, d in [("C1","C2"), ("C2","C1"),
#                          ("C3","C5"), ("C5","C3"),
#                          ("C4","C6"), ("C6","C4"),
#                          ("O1","O3"), ("O3","O1"),
#                          ("O2","O4"), ("O4","O2")]:
#                 if s in apa_idx and d in apa_idx:
#                     flip[apa_idx[s]] = apa_idx[d]
#             perms.append(flip)
#     # ---- END TEMPORARY ----

#     rmsd = np.zeros((n, n))
#     for i in range(n):
#         best = None
#         for p in perms:
#             diff = coords_ensemble - coords_ensemble[i][p]
#             r = np.sqrt((diff ** 2).sum(axis=(1, 2)) / na)
#             best = r if best is None else np.minimum(best, r)
#         rmsd[i] = best
#     return rmsd

# ####################################TEMPORARY######################################################################################################################################
# ####################################TEMPORARY######################################################################################################################################
# ####################################TEMPORARY######################################################################################################################################


# def cluster_poses(rmsd_mat, cutoff=2.0, linkage_method="average"):
#     """
#     Description:
#         Agglomerative hierarchical clustering on a precomputed RMSD matrix,
#         cut at a fixed distance threshold to produce flat clusters.

#     Args:
#         rmsd_mat: Symmetric pairwise RMSD matrix (n, n) in Angstrom.
#         cutoff: Distance threshold; pairs whose merge height is <= cutoff
#             are placed in the same flat cluster.
#         linkage_method: scipy linkage method ("average", "complete", ...).

#     Returns:
#         ndarray of length n with 1-indexed integer cluster labels.
#     """
#     condensed = squareform(rmsd_mat, checks=False)
#     Z = linkage(condensed, method=linkage_method)
#     return fcluster(Z, t=cutoff, criterion="distance")




# def summarize_clusters(labels, rmsd_mat):
#     """
#     Description:
#         For each cluster, compute size, occupancy, medoid frame (real member
#         with minimum mean RMSD to the others), and maximum intra-cluster RMSD.

#     Args:
#         labels: Cluster labels from cluster_poses.
#         rmsd_mat: Same pairwise RMSD matrix used during clustering.

#     Returns:
#         List of dicts sorted by descending size, each with keys:
#             cluster_id, size, occupancy, medoid_idx, poses,
#             medoid_mean_rmsd, max_intra_rmsd.
#     """
#     n_total = len(labels)
#     out = []
#     for cid in sorted(set(labels)):
#         poses = np.where(labels == cid)[0]
#         sub = rmsd_mat[np.ix_(poses, poses)]
#         mean_dists = sub.mean(axis=1)
#         medoid_local = int(np.argmin(mean_dists))
#         out.append({
#             "cluster_id": int(cid),
#             "size": int(len(poses)),
#             "occupancy": len(poses) / n_total,
#             "medoid_idx": int(poses[medoid_local]),
#             "poses": poses.tolist(),
#             "medoid_mean_rmsd": float(mean_dists[medoid_local]),
#             "max_intra_rmsd": float(sub.max()),
#         })
#     out.sort(key=lambda x: -x["size"])
#     return out

# def analyze_ligand_pose_clusters(models, rmsd_cutoff=2.0, linkage_method="average", symmetric = False):
#     """
#     Description:
#         Cluster ligand poses in a PLACER multi-model PDB by pairwise heavy-atom
#         RMSD and report per-cluster occupancy and medoid frames.

#     Args:
#         models: List of Bio.PDB Models (PLACER output).
#         rmsd_cutoff: RMSD threshold (A) for flat clustering.
#         linkage_method: scipy linkage method.

#     Returns:
#         Dict with keys: ligand_info, coords_ensemble, rmsd_matrix, labels, summary.
#     """
#     lig_info, coords_ens = build_ligand_ensemble(models)

#     if symmetric:
#         rmsd_mat = symmetry.pairwise_rmsd_matrix(coords_ens, models) # Temporal line to reflect symmetry of adipic acid for clustering ligand
#     else:
#         rmsd_mat = pairwise_rmsd_matrix(coords_ens) # This line is the original function, but temporarily, change this line to reflect symmetry of adipic acid for clustering ligand pose
#     labels = cluster_poses(rmsd_mat, cutoff=rmsd_cutoff, linkage_method=linkage_method)
#     summary = summarize_clusters(labels, rmsd_mat)
#     return {
#         "ligand_info":     lig_info,
#         "coords_ensemble": coords_ens,
#         "rmsd_matrix":     rmsd_mat,
#         "labels":          labels,
#         "summary":         summary,
#     }