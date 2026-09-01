import os
import time
import argparse

import pyrosetta
from pyrosetta import Pose
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

ONE_LETTER_AA = ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
                 'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T']

ONE_TO_THREE = {
    'G': 'GLY', 'A': 'ALA', 'L': 'LEU', 'M': 'MET', 'F': 'PHE', 'W': 'TRP',
    'K': 'LYS', 'Q': 'GLN', 'E': 'GLU', 'S': 'SER', 'P': 'PRO', 'V': 'VAL',
    'I': 'ILE', 'C': 'CYS', 'Y': 'TYR', 'H': 'HIS', 'R': 'ARG', 'N': 'ASN',
    'D': 'ASP', 'T': 'THR'
}


def load_ligand_params_to_pose(params, pose):
    """Attach ligand params to a pose's residue type set.

    Args:
        params: List of params file paths.
        pose: Pose to attach the residue types to.
    """
    if len(params) != 0 and params[0] != "":
        params = pyrosetta.Vector1(params)
        res_set = pose.conformation().modifiable_residue_type_set_for_conf()
        res_set.read_files_for_base_residue_types(params)
        pose.conformation().reset_residue_type_set_for_conf(res_set)


def get_ligand_coords(pose):
    """Collect heavy-atom xyz of every ligand residue in the pose.

    Arg:
        pose: Input pose.

    Return:
        List of xyzVector for all ligand heavy atoms.
    """
    ligand_atoms = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_ligand():
            for a in range(1, res.natoms() + 1):
                if res.atom_type(a).is_heavyatom():
                    ligand_atoms.append(res.xyz(a))
    return ligand_atoms


def min_distance(atom_xyz, ligand_coords):
    """Minimum distance from a single atom to any ligand atom.

    Args:
        atom_xyz: xyzVector of the query atom.
        ligand_coords: List of ligand heavy-atom xyzVectors.

    Return:
        Minimum distance, or None if atom_xyz is None.
    """
    if atom_xyz is None:
        return None
    return min((atom_xyz - l).norm() for l in ligand_coords)


def determine_design_residues(pose, cut1=6.0, cut2=8.0):
    """Pick protein residues near the ligand as mutation targets.

    Args:
        pose: Complex pose with ligand present.
        cut1: CA distance cutoff for inclusion.
        cut2: Extended CA cutoff, used only if CB is closer than CA.

    Return:
        List of residue indices (pose numbering) to mutate.
    """
    ligand_atoms = get_ligand_coords(pose)
    if not ligand_atoms:
        raise ValueError("No ligand residue found in the pose; "
                         "cannot determine mutation targets.")

    designable = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue

        d_ca = min_distance(res.xyz("CA"), ligand_atoms)

        has_cb = res.has("CB")
        d_cb = min_distance(res.xyz("CB"), ligand_atoms) if has_cb else None

        if d_ca <= cut1:
            designable.append(i)
        elif d_ca <= cut2 and has_cb and d_cb < d_ca:
            designable.append(i)

    return designable


def mutate_residue(pose, posi, amino):
    """Return a copy of the pose with a single residue substituted.

    Args:
        pose: Input pose (left untouched).
        posi: Residue index (pose numbering) to mutate.
        amino: Target one-letter amino acid code.

    Return:
        New pose carrying the point mutation.
    """
    mutant = Pose()
    mutant.assign(pose)

    mutate = MutateResidue()
    mutate.set_target(posi)
    mutate.set_res_name(ONE_TO_THREE[amino])
    mutate.apply(mutant)

    return mutant


def main():
    parser = argparse.ArgumentParser(
        description="Single saturation mutagenesis of ligand-proximal residues."
    )
    parser.add_argument("-i", "--input_pdb", type=str, required=True,
                        help="Path to the input PDB file.")
    parser.add_argument("-o", "--output_dir", type=str, default="outputs/mutants",
                        help="Directory to write mutant PDB files.")
    parser.add_argument("-pr", "--params", type=str, default=None, nargs="+",
                        help="Ligand params file path(s) (optional, multiple allowed).")
    parser.add_argument("-c1", "--cut1", type=float, default=6.0,
                        help="CA distance cutoff for mutation targets.")
    parser.add_argument("-c2", "--cut2", type=float, default=8.0,
                        help="Extended CA cutoff used when CB is closer than CA.")
    args = parser.parse_args()

    pyrosetta.init()
    os.makedirs(args.output_dir, exist_ok=True)

    pose = Pose()
    load_ligand_params_to_pose(args.params if args.params else [""], pose)
    pyrosetta.io.pose_from_file(pose, args.input_pdb)
    info = pose.pdb_info()

    name = os.path.splitext(os.path.basename(args.input_pdb))[0]
    positions = determine_design_residues(pose, args.cut1, args.cut2)
    # print(f"Mutation target residues ({len(positions)}): {positions}")
    print(f"Mutation target residues ({len(positions)}): "
        f"{[f'{info.number(p)}{info.chain(p)}' for p in positions]}")


    targets = []
    for posi in positions:
        wt = pose.residue(posi).name1()
        if wt not in ONE_LETTER_AA:
            print(f"Skipping position {posi}: non-canonical residue "
                  f"({pose.residue(posi).name()})")
            continue
        targets.append((posi, wt))

    total = len(targets)
    count = 0
    start_time = time.time()

    for idx, (posi, wt) in enumerate(targets, start=1):
        cycle_start = time.time()
        for amino in ONE_LETTER_AA:
            if amino == wt:
                continue

            mutant = mutate_residue(pose, posi, amino)
            # out_path = os.path.join(args.output_dir, f"{name}_{wt}{posi}{amino}.pdb")
            out_path = os.path.join(args.output_dir,
                        f"{name}_{wt}{info.number(posi)}{amino}.pdb")

            mutant.dump_pdb(out_path)
            count += 1

        elapsed = time.time() - start_time
        cycle_time = time.time() - cycle_start
        eta = (elapsed / idx) * (total - idx)
        print(f"  [{idx}/{total}] Mutating position {info.number(posi)} (wildtype: {wt}): "
              f"{cycle_time:.1f} s  | Elapsed: {elapsed / 60:.1f} min  "
              f"| Remaining: {eta / 60:.1f} min")

    total_time = time.time() - start_time
    print(f"\nWrote {count} mutant PDB files to {args.output_dir} "
          f"(total {total_time / 60:.1f} min)")


if __name__ == "__main__":
    main()
