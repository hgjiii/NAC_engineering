"""FastRelax a single PDB, ligands included.

Relaxes the whole structure with the starting coordinates restrained. Use
run_fastrelax.py instead when a directory needs relaxing in bulk, or when the
output feeds PLACER and therefore has to come out without hydrogens.

Usage:
    python pipeline/relax_protein.py -i model.pdb
    python pipeline/relax_protein.py -i model.pdb -pr LIG.params
    python pipeline/relax_protein.py -i model.pdb -pr LIG.params,COF.params -o outdir
    python pipeline/relax_protein.py -i model.pdb -pr LIG.params -pc xhb.txt
"""

import os
import argparse

import pyrosetta
from pyrosetta import Pose, pose_from_pdb, get_fa_scorefxn, rosetta


def split_paths(value):
    """Normalise a path argument into a list.

    Accepts what either script style hands over: a comma separated string as
    -pr uses, or the list argparse builds from a space separated -pc. Commas
    inside a list entry are split too, so both spellings work everywhere.

    Args:
        value: String, list of strings, or None.

    Return:
        List of non-empty path strings.
    """
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    return [p.strip() for item in items for p in item.split(",") if p.strip()]


def init_rosetta(params, patches=None, load_PDB_components=None):
    """Initialise Rosetta with every ligand params and patch file in one go.

    All of them must reach a single init(): each call resets the flags, so
    initialising once per file would leave only the last one registered.

    Patches in particular have to be handed over at init time, exactly as
    run_fastrelax.py does it. A modified residue such as XHB is a patched
    base type, and the pose-level residue type route reads base types only,
    so a patch supplied that way is silently ignored.

    Args:
        params: List of ligand params file paths, possibly empty.
        patches: List of Rosetta patch files defining modified residues,
            possibly empty or None.
        load_PDB_components: "true"/"false" for Rosetta's flag of the same
            name, or None to keep Rosetta's default.

    Raise:
        SystemExit if a named params or patch file does not exist, which
        Rosetta would otherwise skip quietly and only fail on much later.
    """
    patches = patches or []

    missing = [p for p in params if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Params file(s) not found: {missing}")

    missing = [p for p in patches if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Patch file(s) not found: {missing}")

    flags = []
    if params:
        flags.append("-extra_res_fa " + " ".join(params))
    if patches:
        flags.append("-extra_patch_fa " + " ".join(patches))
    if load_PDB_components is not None:
        flags.append(f"-load_PDB_components {load_PDB_components}")

    pyrosetta.init(extra_options=" ".join(flags))


def relax_pdb(input_pdb, output_dir=None):
    """FastRelax one PDB and write <name>.relax.pdb.

    Args:
        input_pdb: Path to the structure to relax.
        output_dir: Directory to write into, or None to write beside the input.

    Return:
        Path of the file written.
    """
    name = os.path.basename(input_pdb)
    if name.endswith(".pdb"):
        name = name[:-len(".pdb")]

    out_dir = output_dir or os.path.dirname(os.path.abspath(input_pdb))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.relax.pdb")

    pose = pose_from_pdb(input_pdb)
    print(pose)

    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(get_fa_scorefxn())
    relax.constrain_relax_to_start_coords(True)
    relax.apply(pose)

    pose.dump_pdb(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="FastRelax a single PDB, ligands included.")
    parser.add_argument("-i", "--input_pdb", type=str, required=True,
                        help="PDB file to relax.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Directory for the output "
                             "(default: beside the input PDB).")
    parser.add_argument("-pr", "--params", type=str, default=None,
                        help="Ligand params file(s), comma separated.")
    parser.add_argument("-pc", "--patches", type=str, default=None, nargs="+",
                        help="Rosetta patch file(s) defining modified "
                             "residues, e.g. a covalently modified CYS. "
                             "Space or comma separated.")
    parser.add_argument("-lc", "--load_PDB_components", type=str, default=None,
                        choices=["true", "false"],
                        help="Rosetta's -load_PDB_components. Set false so an "
                             "unrecognised residue fails loudly instead of "
                             "being replaced from the PDB component dictionary.")
    args = parser.parse_args()

    params = split_paths(args.params)
    patches = split_paths(args.patches)
    init_rosetta(params, patches, args.load_PDB_components)

    out_path = relax_pdb(args.input_pdb, args.output_dir)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
