import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from Bio import BiopythonWarning
from Bio.Seq import Seq
warnings.simplefilter("ignore", BiopythonWarning)

from process import docking
from process import search


# NCBI Entrez contact email (required by efetch).
ENTREZ_EMAIL = "ghdrms206@gmail.com"


def check_translation(aa_seq, dna_seq):
    """
    Description:
        Standard-codon-table sanity check: translate the CDS DNA and test
        whether it matches the protein sequence (trailing stop codon ignored).

    Args:
        aa_seq: Protein sequence string, or None.
        dna_seq: CDS DNA sequence string, or None.

    Returns:
        True if the DNA translates to aa_seq under the standard codon table,
        otherwise False (including when either sequence is missing).
    """
    if not aa_seq or not dna_seq:
        return False
    try:
        trans = str(Seq(dna_seq).translate())
    except Exception:
        return False
    if trans.endswith("*"):
        trans = trans[:-1]
    return aa_seq == trans


def main():
    parser = argparse.ArgumentParser(
        description="Docking-based candidate analysis: aggregate Vina affinities "
                    "matched to PLACER references and enrich NAC candidates with "
                    "taxonomy/sequence/DNA metadata.")
    parser.add_argument("--docking_dir", required=True,
                        help="Root of docking outputs (entry subfolders).")
    parser.add_argument("--ref_root", required=True,
                        help="Root of multi-model PLACER reference PDBs.")
    parser.add_argument("--candidates", required=True,
                        help="NAC candidates table (tab-separated) with an 'entry' column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--ligand_resname", required=True,
                        help="Three-letter ligand residue name (e.g. ADI).")

    parser.add_argument("--rmsd_threshold", type=float, default=3.0,
                        help="Matched-pose RMSD cutoff (A); pass to relax/disable.")
    parser.add_argument("--sort", action="store_true",
                        help="If set, sort the output by affinity_mean (ascending).")

    args = parser.parse_args()

    search.configure_entrez(ENTREZ_EMAIL)

    # Load NAC candidates
    df_final = pd.read_csv(args.candidates)

    # Per entry: docking aggregation + metadata search + sanity check (one pass)
    add = {"affinity_mean": [], "pose_rmsd_mean": [], "taxonomy": [],
           "sequence": [], "source_dna": [], "sanity_check": []}
    for _, row in tqdm(df_final.iterrows(), total=len(df_final)):
        uniprot_id = row["entry"]

        res = docking.aggregate_docking_for_entry(
            uniprot_id, args.docking_dir, args.ref_root,
            ligand_resname=args.ligand_resname, rmsd_threshold=args.rmsd_threshold,
        )
        if res is None:
            aff, rmsd = np.nan, np.nan
            dock_log = "docking=none".ljust(31)
        else:
            aff = res["affinity_mean"]
            rmsd = res["pose_rmsd_mean"]
            dock_log = (f"aff={aff:>6.2f}  rmsd={res['rmsd_to_ref_mean']:>4.2f}  "
                        f"n={res['n_models']}/{res['n_total_models']}").ljust(31)

        tax = search.get_taxonomy(uniprot_id)
        seq = search.get_sequence(uniprot_id)
        dna_rec = search.get_dna_from_uniprot(uniprot_id)
        dna_seq = dna_rec["dna"] if dna_rec else None
        tax_name = tax["scientific_name"] if tax else None
        sanity = check_translation(seq, dna_seq)

        add["affinity_mean"].append(aff)
        add["pose_rmsd_mean"].append(rmsd)
        add["taxonomy"].append(tax_name)
        add["sequence"].append(seq)
        add["source_dna"].append(dna_seq)
        add["sanity_check"].append(sanity)

        tqdm.write(
            f"  {uniprot_id:<25s} {dock_log}  "
            f"tax={'loaded' if tax_name else 'failed'}  "
            f"seq={'loaded' if seq else 'failed'}  "
            f"dna={'loaded' if dna_seq else 'failed'}  "
            f"sanity={'pass' if sanity else 'fail'}"
        )

    df_final = df_final.assign(**add)
    if args.sort:
        df_final = df_final.sort_values(by="affinity_mean")

    # sanity_check as the last column
    cols = [c for c in df_final.columns if c != "sanity_check"] + ["sanity_check"]
    df_final = df_final[cols]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    df_final.to_csv(args.output, index=False)
    print(f"Wrote {len(df_final)} rows -> {args.output}")


if __name__ == "__main__":
    main()
