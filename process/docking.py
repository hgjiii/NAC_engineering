import os

import numpy as np

from process import structure


def parse_pdbqt(filepath, format="general"):
    """
    Description:
        Parse a pdbqt file into atom coordinates (and Vina affinities).

    Args:
        filepath: Path to the pdbqt file.
        format: 'general' for a single conformer (input ligand), returns coords
            only as an ndarray (n_atoms, 3); 'vina' for a multi-conformer Vina
            output, returns a list of {"affinity", "coords"} dicts.

    Returns:
        ndarray (n_atoms, 3) when format='general'; list of
        {"affinity": float, "coords": ndarray (n_atoms, 3)} when format='vina'.
    """
    conformers = []
    current_coords = []
    current_affinity = None

    with open(filepath) as f:
        for line in f:
            if line.startswith("MODEL"):
                current_coords = []
                current_affinity = None
            elif line.startswith("REMARK VINA RESULT"):
                current_affinity = float(line.split()[3])
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_coords.append([x, y, z])
            elif line.startswith("ENDMDL"):
                if current_coords:
                    conformers.append({
                        "affinity": current_affinity,
                        "coords":   np.array(current_coords),
                    })

    # general: no MODEL/ENDMDL blocks
    if not conformers and current_coords:
        conformers.append({
            "affinity": None,
            "coords":   np.array(current_coords),
        })

    if format == "general":
        return conformers[0]["coords"]  # np.ndarray (n_atoms, 3)
    elif format == "vina":
        return conformers               # list of {"affinity", "coords"}
    else:
        raise ValueError(f"Unknown format: '{format}'. Use 'general' or 'vina'.")


def get_reference_ligand_coords_from_multimodel(pdb_path, model_idx, ligand_resname="ADI"):
    """
    Description:
        Extract ligand heavy-atom coords from a specific model in a
        multi-model PDB via Bio.PDB.

    Args:
        pdb_path: Path to the multi-model PDB.
        model_idx: 1-based model index.
        ligand_resname: Three-letter residue name of the target ligand.

    Returns:
        np.ndarray of shape (n_atoms, 3); empty array if not found.
    """
    models = structure.load_models_from_pdb(pdb_path)
    model = models[model_idx - 1]   # 1-based -> 0-based
    coords = []
    for res in model.get_residues():
        if res.get_resname().strip() != ligand_resname:
            continue
        for atom in res.get_atoms():
            if atom.element in (None, "H") or atom.get_name().startswith("H"):
                continue
            coords.append(atom.get_coord())
    return np.array(coords)


def select_pose_by_reference(pdbqt_path, ref_coords):
    """
    Description:
        Pick the docking pose closest in heavy-atom RMSD to a reference.

    Args:
        pdbqt_path: Path to vina-generated multi-pose pdbqt file.
        ref_coords: Reference coords (n_atoms, 3), order must match ligand.

    Returns:
        Dict with "affinity", "coords", "rank" (0-based), and "rmsd_to_ref";
        None if no poses or atom count mismatch.
    """
    poses = parse_pdbqt(pdbqt_path, format="vina")
    if not poses or poses[0]["coords"].shape != ref_coords.shape:
        return None
    rmsds = [np.sqrt(((p["coords"] - ref_coords) ** 2).sum() / ref_coords.shape[0])
             for p in poses]
    best = int(np.argmin(rmsds))
    return {
        "affinity": poses[best]["affinity"],
        "coords": poses[best]["coords"],
        "rank": best,
        "rmsd_to_ref": float(rmsds[best]),
    }


def pairwise_rmsd(coords_list):
    """
    Description:
        Mean of pairwise heavy-atom RMSDs between coords, no superposition.

    Args:
        coords_list: List of np.ndarray (n_atoms, 3), all same shape.

    Returns:
        Mean pairwise RMSD; np.nan if fewer than 2 entries.
    """
    n = len(coords_list)
    if n < 2:
        return np.nan
    rmsds = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords_list[i] - coords_list[j]
            rmsds.append(np.sqrt((diff ** 2).sum() / diff.shape[0]))
    return float(np.mean(rmsds))


def aggregate_docking_for_entry(entry, docking_dir, ref_root, ligand_resname="ADI",
                                rmsd_threshold=None):
    """
    Description:
        Aggregate one entry's docking results matched to its PLACER multi-model
        reference coords.

    Args:
        entry: Entry id (subfolder name under docking_dir; ref PDB is
            <ref_root>/<entry>.relax_model.pdb).
        docking_dir: Root of docking outputs (entry subfolders).
        ref_root: Root containing multi-model PLACER PDBs
            (one per entry, e.g. carA_<UID>.relax_model.pdb).
        ligand_resname: Ligand resname.
        rmsd_threshold: Optional cutoff for matched-pose RMSD.

    Returns:
        Aggregated stats dict for this entry, or None if the entry folder /
        reference PDB is missing or no pose matched.
    """
    entry_dir = os.path.join(docking_dir, entry)
    ref_pdb = os.path.join(ref_root, f"{entry}.relax_model.pdb")
    if not (os.path.isdir(entry_dir) and os.path.exists(ref_pdb)):
        return None

    per_model = []
    for fname in sorted(os.listdir(entry_dir)):
        if not (fname.startswith("ligand_") and fname.endswith(".pdbqt")):
            continue
        base = fname.replace("ligand_", "").replace(".pdbqt", "")
        model_idx = int(base.split("_")[-1])

        ref_coords = get_reference_ligand_coords_from_multimodel(
            ref_pdb, model_idx, ligand_resname
        )
        if ref_coords.shape[0] == 0:
            continue

        result = select_pose_by_reference(os.path.join(entry_dir, fname), ref_coords)
        if result is None:
            continue
        if rmsd_threshold is not None and result["rmsd_to_ref"] > rmsd_threshold:
            continue
        per_model.append({"model": base, **result})

    if not per_model:
        return None

    n_total = sum(1 for f in os.listdir(entry_dir)
                  if f.startswith("ligand_") and f.endswith(".pdbqt"))

    return {
        "affinity_mean": float(np.mean([m["affinity"] for m in per_model])),
        "affinity_std": float(np.std([m["affinity"] for m in per_model])),
        "rank_mean": float(np.mean([m["rank"] for m in per_model])),
        "rmsd_to_ref_mean": float(np.mean([m["rmsd_to_ref"] for m in per_model])),
        "pose_rmsd_mean": pairwise_rmsd([m["coords"] for m in per_model]),
        "n_models": len(per_model),
        "n_total_models": n_total,
        "per_model": per_model,
    }
