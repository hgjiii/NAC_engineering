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


def build_init_options(args):
    """Assemble the pyrosetta.init() flags from the parsed arguments.

    Patches have to be handed to Rosetta at init time: the pose-level route
    load_ligand_params_to_pose() uses reads base residue types only, so a
    patch passed that way is silently ignored.

    Args:
        args: Parsed command line.

    Return:
        Flag string for pyrosetta.init(extra_options=...).

    Raise:
        SystemExit if a named patch file does not exist, which Rosetta would
        otherwise skip quietly and only fail on much later.
    """
    flags = []

    if args.patches:
        missing = [p for p in args.patches if not os.path.isfile(p)]
        if missing:
            raise SystemExit(f"Patch file(s) not found: {missing}")
        flags.append("-extra_patch_fa " + " ".join(args.patches))

    if args.load_PDB_components is not None:
        flags.append(f"-load_PDB_components {args.load_PDB_components}")

    return " ".join(flags)


def parse_exclusions(tokens):
    """Parse -x/--exclude tokens into residues to leave unmutated.

    A token is a PDB residue number, optionally suffixed with its chain --
    the same form this script prints its target list in, so entries can be
    copied straight out of the log.

    Args:
        tokens: Raw CLI tokens, e.g. ["296", "479A"].

    Return:
        Set of (number, chain) pairs, chain None when unspecified.

    Raise:
        ValueError on a token that is not a residue number.
    """
    parsed = set()
    for token in tokens:
        token = token.strip().upper()
        number, chain = (token[:-1], token[-1]) if token[-1:].isalpha() \
            else (token, None)
        if not number.lstrip("-").isdigit():
            raise ValueError(f"Invalid --exclude token '{token}': expected a "
                             "residue number, optionally followed by a chain "
                             "id (e.g. 296 or 296A).")
        parsed.add((int(number), chain))
    return parsed


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


def parse_ligand_specs(specs):
    """Parse -l/--ligands tokens into {residue_name: atom_names or None}.

    A token is either a bare residue name (whole ligand) or
    "NAME@ATOM1,ATOM2,..." (only that moiety). Names are upper-cased.
    Repeating a name merges its atoms; a bare name wins over a moiety.

    Args:
        specs: List of raw CLI tokens, e.g. ["ADI", "AMP@N1,C2"].

    Return:
        Dict mapping residue name to a set of atom names, or None for
        "use every heavy atom of this ligand".
    """
    parsed = {}
    for spec in specs:
        name, sep, atoms = spec.partition("@")
        name = name.strip().upper()
        if not name:
            raise ValueError(f"Invalid ligand spec '{spec}': missing residue name.")

        if not sep:
            parsed[name] = None
            continue

        atom_names = {a.strip().upper() for a in atoms.split(",") if a.strip()}
        if not atom_names:
            raise ValueError(f"Invalid ligand spec '{spec}': "
                             "'@' given with no atom names.")
        if name not in parsed:
            parsed[name] = atom_names
        elif parsed[name] is not None:
            parsed[name] |= atom_names

    return parsed


def select_ligand_atoms(pose, ligand_specs=None):
    """Pick the ligand heavy atoms that define the mutation site.

    Args:
        pose: Input pose.
        ligand_specs: Dict from parse_ligand_specs, or None to use every
            heavy atom of every ligand residue in the pose.

    Return:
        List of (residue index, [atom indices]) for the selected ligands.
    """
    selection = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_ligand():
            continue

        name = res.name3().strip().upper()
        if ligand_specs is not None and name not in ligand_specs:
            continue
        wanted = ligand_specs[name] if ligand_specs is not None else None

        picked = []
        for a in range(1, res.natoms() + 1):
            if not res.atom_type(a).is_heavyatom():
                continue
            if wanted is not None and res.atom_name(a).strip().upper() not in wanted:
                continue
            picked.append(a)

        selection.append((i, picked))

    return selection


def check_ligand_selection(pose, selection, ligand_specs):
    """Fail loudly on ligand or atom names that matched nothing.

    Args:
        pose: Input pose.
        selection: Output of select_ligand_atoms.
        ligand_specs: Output of parse_ligand_specs, or None.

    Raise:
        ValueError listing what is actually available in the pose.
    """
    present = sorted({pose.residue(i).name3().strip().upper()
                      for i in range(1, pose.total_residue() + 1)
                      if pose.residue(i).is_ligand()})

    if not selection:
        if ligand_specs:
            raise ValueError(
                f"None of the requested ligands {sorted(ligand_specs)} were found. "
                f"Ligand residues present in the pose: {present or '(none)'}")
        raise ValueError("No ligand residue found in the pose; "
                         "cannot determine mutation targets.")

    if not ligand_specs:
        return

    matched = {pose.residue(i).name3().strip().upper() for i, _ in selection}
    missing = sorted(set(ligand_specs) - matched)
    if missing:
        raise ValueError(f"Requested ligand(s) {missing} not found. "
                         f"Ligand residues present in the pose: {present}")

    for i, picked in selection:
        res = pose.residue(i)
        wanted = ligand_specs[res.name3().strip().upper()]
        if wanted is None:
            continue
        found = {res.atom_name(a).strip().upper() for a in picked}
        unknown = sorted(wanted - found)
        if unknown:
            available = [res.atom_name(a).strip()
                         for a in range(1, res.natoms() + 1)
                         if res.atom_type(a).is_heavyatom()]
            raise ValueError(
                f"Atom(s) {unknown} not found in ligand "
                f"{res.name3().strip()} (residue {i}). "
                f"Heavy atoms available: {available}")


def get_ligand_coords(pose, selection):
    """Collect xyz of the selected ligand atoms.

    Args:
        pose: Input pose.
        selection: Output of select_ligand_atoms.

    Return:
        List of xyzVector defining the site.
    """
    return [pose.residue(i).xyz(a) for i, picked in selection for a in picked]


def report_ligand_selection(pose, selection, info):
    """Print which ligand atoms define the mutation site.

    Args:
        pose: Input pose.
        selection: Output of select_ligand_atoms.
        info: pose.pdb_info() for chain/number labels.
    """
    print(f"Ligand atoms defining the site ({len(selection)} residue(s)):")
    for i, picked in selection:
        res = pose.residue(i)
        n_heavy = sum(1 for a in range(1, res.natoms() + 1)
                      if res.atom_type(a).is_heavyatom())
        names = [res.atom_name(a).strip() for a in picked]
        detail = "all heavy atoms" if len(picked) == n_heavy else ", ".join(names)
        print(f"  {res.name3().strip()} {info.chain(i)}{info.number(i)}: "
              f"{len(picked)}/{n_heavy} atoms ({detail})")


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


def determine_design_residues(pose, ligand_atoms, cut1=6.0, cut2=8.0):
    """Pick protein residues near the selected ligand atoms as mutation targets.

    Args:
        pose: Complex pose with ligand present.
        ligand_atoms: Ligand heavy-atom xyzVectors defining the site.
        cut1: CA distance cutoff for inclusion.
        cut2: Extended CA cutoff, used only if CB is closer than CA.

    Return:
        List of residue indices (pose numbering) to mutate.
    """
    if not ligand_atoms:
        raise ValueError("No ligand atoms given; "
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
    parser.add_argument("-pc", "--patches", type=str, default=None, nargs="+",
                        help="Rosetta patch file(s) defining modified residues "
                             "(optional, multiple allowed).")
    parser.add_argument("-lc", "--load_PDB_components", type=str, default=None,
                        choices=["true", "false"],
                        help="Rosetta's -load_PDB_components. Leave unset to "
                             "keep Rosetta's default (true). Set false so an "
                             "unrecognised residue name fails loudly instead "
                             "of being replaced from the PDB component "
                             "dictionary.")
    parser.add_argument("-x", "--exclude", type=str, default=None, nargs="+",
                        help="Residue(s) to leave unmutated, by PDB number or "
                             "number plus chain (e.g. -x 296 479A). Use it for "
                             "chemically modified residues, whose modification "
                             "MutateResidue would drop without warning, and for "
                             "catalytic positions.")
    parser.add_argument("-l", "--ligands", type=str, default=None, nargs="+",
                        help="Ligand(s) defining the mutation site, space separated. "
                             "Use 'NAME' for the whole ligand or 'NAME@AT1,AT2,...' "
                             "to use only that moiety "
                             "(e.g. -l ADI AMP, or -l ADI@C1,O1,C2 AMP@N1,C2). "
                             "Default: every ligand residue found in the pose.")
    parser.add_argument("-c1", "--cut1", type=float, default=6.0,
                        help="CA distance cutoff for mutation targets.")
    parser.add_argument("-c2", "--cut2", type=float, default=8.0,
                        help="Extended CA cutoff used when CB is closer than CA.")
    args = parser.parse_args()

    excluded = parse_exclusions(args.exclude) if args.exclude else set()

    pyrosetta.init(extra_options=build_init_options(args))
    os.makedirs(args.output_dir, exist_ok=True)

    pose = Pose()
    load_ligand_params_to_pose(args.params if args.params else [""], pose)
    pyrosetta.io.pose_from_file(pose, args.input_pdb)
    info = pose.pdb_info()

    name = os.path.splitext(os.path.basename(args.input_pdb))[0]

    ligand_specs = parse_ligand_specs(args.ligands) if args.ligands else None
    selection = select_ligand_atoms(pose, ligand_specs)
    check_ligand_selection(pose, selection, ligand_specs)
    report_ligand_selection(pose, selection, info)
    ligand_atoms = get_ligand_coords(pose, selection)

    positions = determine_design_residues(pose, ligand_atoms,
                                          args.cut1, args.cut2)
    # print(f"Mutation target residues ({len(positions)}): {positions}")
    print(f"Mutation target residues ({len(positions)}): "
        f"{[f'{info.number(p)}{info.chain(p)}' for p in positions]}")


    targets = []
    for posi in positions:
        number, chain = info.number(posi), info.chain(posi)
        if (number, chain) in excluded or (number, None) in excluded:
            print(f"Excluding position {number}{chain}: "
                  f"{pose.residue(posi).name()}")
            continue
        wt = pose.residue(posi).name1()
        if wt not in ONE_LETTER_AA:
            print(f"Skipping position {number}{chain}: non-canonical residue "
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
