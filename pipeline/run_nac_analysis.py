import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import pickle
import argparse

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from Bio import BiopythonWarning
warnings.simplefilter("ignore", BiopythonWarning)

from process import structure, cluster
from process import preorganization as po
from process.config import load_config
from process.interaction import build_catalytic_key_res, compute_pair_distances
from process.nac import extract_entry_id, fill_interaction_pairs, compute_nac


# Columns written to the output CSV (scalars only; array-like fields excluded)

OUTPUT_COLUMNS = [
    "entry", "nac_fraction_holo", "nac_holo_idxs",
    "n_confident_holo", "prmsd_mean", "prmsd_std",
    "nac_fraction_apo", "nac_apo_idxs", "n_confident_apo",
]

APO_COLUMNS = ["nac_fraction_apo", "nac_apo_idxs", "n_confident_apo"]

def process_entry(pdb_path, csv_path, apo_pdb_path, apo_csv_path, cat_entry, cfg, params):
    """
    Description:
        Run the full holo NAC analysis for one PLACER entry and return its
        result dict (scalar and array-like fields).

    Args:
        pdb_path: Path to the multi-model PLACER PDB.
        csv_path: Path to the matching PLACER info CSV (pRMSD).
        apo_pdb_path: Path to the apo PLACER PDB, or None to skip apo.
        apo_csv_path: Path to the apo info CSV, or None to skip apo.
        cat_entry: Per-entry catalytic mapping dict.
        cfg: Loaded enzyme config dict.
        params: Dict of algorithm parameters.

    Returns:
        Result dict for this entry, or None if the entry is skipped.
    """
    models = structure.load_models_from_pdb(pdb_path)
    prmsd_list = structure.load_prmsd(csv_path)

    # Major ligand pose cluster -> key residues near the ligand
    lig_cluster = cluster.analyze_ligand_pose_clusters(
        models, rmsd_cutoff=params["cutoff_rmsd"], symmetric=cfg["symmetry_pairs"] is not None
    )
    major_pose_idx = lig_cluster["summary"][0]["poses"]
    ref_models = [models[i] for i in major_pose_idx]
    lig_coords = [structure.get_ligand_heavy_coords(m) for m in ref_models]
    key_res_H1 = structure.find_key_residues(
        ref_models, lig_coords, params["cutoff_close"], params["cutoff_orient"]
    )

    # Catalytic residues -> fill interaction pair template
    key_res_cat = build_catalytic_key_res(models[0], cat_entry, cfg["catalytic"]["cat_keys"],
                                          cfg["catalytic"]["chain_id"])
    resname_to_resid = {kr["resname"]: kr["resid"] for kr in key_res_cat}
    try:
        pairs = fill_interaction_pairs(cfg["interaction_pairs"], resname_to_resid)
    except KeyError:
        return None   # missing catalytic residue

    # Atom-atom distances -> NAC computation
    pair_distances = compute_pair_distances(models, pairs, sym_pairs=cfg["symmetry_pairs"])

    buffer = np.array(prmsd_list)
    conf_mask = buffer <= params["max_buffer"]
    effective_cutoffs = {label: base + buffer for label, base in cfg["contact_cutoffs"].items()}
    nac = compute_nac(pair_distances, effective_cutoffs, conf_mask)

    # Rotamer-based preorganization
    result = po.analyze_preorganization_rotamer(models, key_res_H1,
                                                lock_threshold=params["lock_threshold"])
    result["entropy_norm_mean"] = float(np.mean(result["entropy_norm"]))
    result["entropy_norm_std"] = float(np.std(result["entropy_norm"]))
    result["dom_occ_mean"] = float(np.mean(result["dominant_occupancy"]))
    result["dom_occ_std"] = float(np.std(result["dominant_occupancy"]))

    # NAC fields
    result["pair_distances"] = pair_distances
    result["nac_fractions"] = nac["fracs"]
    result["nac_fraction_holo"] = nac["frac_nac"]
    result["nac_holo_idxs"] = nac["nac_holo_idxs"]
    result["n_confident_holo"] = nac["n_confident_holo"]

    # pRMSD statistics
    result["prmsd_mean"] = float(np.mean(prmsd_list))
    result["prmsd_std"] = float(np.std(prmsd_list))

    # Apo preorganization matching
    if apo_pdb_path is not None and os.path.exists(apo_pdb_path) and os.path.exists(apo_csv_path):
        apo_models = structure.load_models_from_pdb(apo_pdb_path)
        apo_prmsd_list = structure.load_prmsd(apo_csv_path)
        apo = po.analyze_preorganization_cat(
            models, apo_models, key_res_cat, nac["nac_holo_idxs"],
            apo_prmsd_list, max_buffer=params["max_buffer"]
        )
        result["nac_apo_idxs"] = apo["nac_apo_idxs"]
        result["nac_fraction_apo"] = apo["nac_fraction_apo"]
        result["n_confident_apo"] = apo["n_confident_apo"]
    else:
        result["nac_apo_idxs"] = None
        result["nac_fraction_apo"] = None
        result["n_confident_apo"] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="PLACER-based NAC analysis (holo).")
    parser.add_argument("--holo_dir", required=True, help="Directory of PLACER holo PDBs.")
    parser.add_argument("--apo_dir", default=None, help="Directory of PLACER apo PDBs.")

    parser.add_argument("--catres", required=True, help="Catalytic residue mapping pickle.")
    parser.add_argument("--config", required=True, help="Enzyme config JSON.")
    parser.add_argument("--output", required=True, help="Output CSV path.")

    parser.add_argument("--cutoff_close", type=float, default=5.0)
    parser.add_argument("--cutoff_orient", type=float, default=6.0)
    parser.add_argument("--cutoff_rmsd", type=float, default=2.0)
    parser.add_argument("--lock_threshold", type=float, default=0.7)
    parser.add_argument("--max_buffer", type=float, default=2.0)

    parser.add_argument("--min_conf", type=int, default=0,
                        help="Keep only entries with n_confident_models >= this "
                             "(applied to holo, and to apo when --apo_dir is given).")

    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(args.catres, "rb") as f:
        cat_mapping = pickle.load(f)
    lookup = {member: key for key in cat_mapping for member in key.split(";")}

    params = {
        "cutoff_close": args.cutoff_close,
        "cutoff_orient": args.cutoff_orient,
        "cutoff_rmsd": args.cutoff_rmsd,
        "lock_threshold": args.lock_threshold,
        "max_buffer": args.max_buffer,
    }

    pdb_names = sorted(f for f in os.listdir(args.holo_dir) if f.endswith(".pdb"))

    all_results = {}
    for file in tqdm(pdb_names):
        entry = extract_entry_id(file)
        pdb_path = os.path.join(args.holo_dir, file)
        csv_path = os.path.join(args.holo_dir, os.path.splitext(file)[0].replace("_model", "") + ".csv")

        apo_pdb_path, apo_csv_path = None, None
        if args.apo_dir is not None:
            apo_pdb_path = os.path.join(args.apo_dir, file)
            apo_csv_path = os.path.join(args.apo_dir, os.path.splitext(file)[0].replace("_model", "") + ".csv")

        if entry not in lookup:
            continue
        cat_entry = cat_mapping[lookup[entry]]

        result = process_entry(pdb_path, csv_path, apo_pdb_path, apo_csv_path, cat_entry, cfg, params)
        if result is None:
            continue

        result["entry"] = entry
        all_results[entry] = result

    # Write scalar columns to CSV
    df = pd.DataFrame.from_dict(all_results, orient="index")


    def to_1based_str(idxs):
        if idxs is None:
            return None
        return ";".join(str(i + 1) for i in idxs)

    df["nac_holo_idxs"] = df["nac_holo_idxs"].apply(to_1based_str)
    df["nac_apo_idxs"] = df["nac_apo_idxs"].apply(to_1based_str)

    # Confidence filter: holo always; apo only when apo_dir given
    if args.min_conf > 0:
        mask = df["n_confident_holo"] >= args.min_conf
        if args.apo_dir is not None:
            mask &= df["n_confident_apo"].fillna(-1) >= args.min_conf
        df = df[mask]

    # Drop apo columns when apo_dir was not given
    if args.apo_dir is not None:
        columns = OUTPUT_COLUMNS
    else:
        columns = [c for c in OUTPUT_COLUMNS if c not in APO_COLUMNS]

    df[columns].to_csv(args.output, index=False)

if __name__ == "__main__":
    main()