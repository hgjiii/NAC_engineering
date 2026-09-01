import sys, os
import re
import time
import warnings
warnings.filterwarnings("ignore")
import argparse

import numpy as np

# DIR = os.getcwd()
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f"{DIR}/PLACER")
import PLACER

## Initializing PLACER model with default checkpoint
placer = PLACER.PLACER()

## Canonical amino acids used for saturation mutagenesis
AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
                "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def structure_name(pdb_path: str):
    """Name prefix used for the PLACER outputs of a structure."""
    return os.path.basename(pdb_path).replace(".pdb" if ".pdb" in pdb_path else ".cif", "")


def build_placer_input(pdb_path: str, lig_path: dict, name: str = None):
    """Build a PLACERinput object, configured identically for every run.

    Args:
        pdb_path: Path to the input PDB/CIF.
        lig_path: {"LIG": "path/to/ligand.mol2", ...} or None (apo prediction).
        name: Output name prefix. Defaults to the structure file name.

    Return:
        Configured PLACERinput object.
    """
    assert isinstance(lig_path, dict) or lig_path == None, 'The argument "lig_path" must be a dictionary type and have the format of {"LIG": "path/to/ligand", ...} (or be a None type)'

    pl_inp = PLACER.PLACERinput()
    pl_inp.pdb(pdb_path)
    pl_inp.name(name if name is not None else structure_name(pdb_path))
    # pl_inp.ligand_reference({"AMP": amp_name, "SIN": sucfile})
    if lig_path is None:
        pl_inp.exclude_sm(True)
    else:
        pl_inp.ignore_ligand_hydrogens(True)
        # pl_inp.predict_ligand([("Z", "LG0", 1)]) # This is the specific line for only CAR
        pl_inp.ligand_reference(lig_path)

    return pl_inp


def run_placer(pdb_path: str, lig_path: dict, output_dir: str, iterations: int = 50,
               mutation: tuple = None, name: str = None):
    """Run PLACER once on a structure, optionally mutating one position at inference.

    Args:
        pdb_path: Path to the input PDB/CIF.
        lig_path: {"LIG": "path/to/ligand.mol2", ...} or None.
        output_dir: Directory the models/scores are written to.
        iterations: Number of PLACER samples.
        mutation: (chain, resno, name3) applied through PLACER's internal mutate(),
            i.e. the backbone stays as in the input and only the side chains of the
            crop are rebuilt. None runs the wildtype.
        name: Output name prefix. Defaults to the structure file name.
    """
    try:
        pl_inp = build_placer_input(pdb_path, lig_path, name=name)
        if mutation is not None:
            chain, resno, name3 = mutation
            pl_inp.mutate({(chain, int(resno)): name3})

        outputs = placer.run(pl_inp, iterations)

        os.makedirs(output_dir, exist_ok=True)
        PLACER.protocol.dump_output(outputs, f"{output_dir}/{pl_inp.name()}", rerank="prmsd")

    except AssertionError as e:
        print("AssertionError: ", e)
        pass


def get_crop_residues(pdb_path: str, lig_path: dict, radius: float = None):
    """Collect the protein residues that survive PLACER's internal crop.

    Mirrors the preprocessing that PLACER.run() does before inference
    (parse structure -> parse fixed ligands -> build crop), so the returned
    residues are exactly the ones whose side chains PLACER rebuilds, i.e. the
    active site pocket around the ligand(s).

    Args:
        pdb_path: Path to the input PDB/CIF.
        lig_path: {"LIG": "path/to/ligand.mol2", ...} or None.
        radius: Optional extra filter. Keep only residues with a heavy atom
            within this distance (A) of any ligand heavy atom.

    Return:
        List of (chain, resno, name3), sorted by chain and residue number.
    """
    pl_inp = build_placer_input(pdb_path, lig_path)

    pdb_parser = PLACER.pdbparser.PDBParser(skip_res=["HOH"] + pl_inp.skip_ligands(), mols=placer.mols())
    cif_parser = PLACER.cifutils.CIFParser(skip_res=["HOH"] + pl_inp.skip_ligands(), mols=placer.mols())

    ligand_reference = None
    if pl_inp.ligand_reference() is not None:
        ligand_reference = PLACER.protocol.parse_ligand_reference(pl_inp.ligand_reference(), placer.mols())

    chains, obmol = PLACER.protocol.parse_input_structure(pl_inp, ligand_reference=ligand_reference,
                                                          pdbparser=pdb_parser, cifparser=cif_parser)
    _, fixed_ligands = PLACER.protocol.parse_fixed_ligand_input(pl_inp, chains)

    # the crop itself is deterministic (nearest maxatoms=600 heavy atoms around the
    # ligand); only the corruption center that build_crop also returns is random
    cropped_atoms, _ = PLACER.protocol.build_crop(placer.dataloader(), pl_inp, chains, obmol, fixed_ligands)

    ligand_xyz = np.array([a.xyz for k, v in chains.items() if v.type == "nonpoly"
                           for a in v.atoms.values() if a.occ > 0 and a.element > 1])

    residues, skipped = {}, []
    for a in cropped_atoms:
        chain, resno, name3 = a.name[0], int(a.name[1]), a.name[2]
        if "polypept" not in chains[chain].type or name3 not in AA3:
            continue
        if a.occ == 0 or a.element <= 1:
            continue
        if resno > 999:
            # PLACER matches mutated positions on the PDB "<chain><resSeq>" field,
            # which is not separable by whitespace for 4-digit residue numbers
            if (chain, resno) not in skipped:
                skipped.append((chain, resno))
            continue
        residues.setdefault((chain, resno, name3), []).append(a.xyz)

    if len(skipped) > 0:
        print(f"# WARNING: skipping {len(skipped)} crop residue(s) with a 4-digit residue "
              f"number, which PLACER cannot mutate: {skipped}")

    if radius is not None:
        assert len(ligand_xyz) > 0, "Cannot apply --radius without a ligand in the structure"
        residues = {res: xyz for res, xyz in residues.items()
                    if np.linalg.norm(np.array(xyz)[:, None, :] - ligand_xyz[None, :, :], axis=-1).min() <= radius}

    return sorted(residues.keys())


def parse_residue_id(res_id: str):
    """Parse a residue identifier such as "270A", "A270" or "270".

    Return:
        (chain, resno), where chain is None if it was not given.
    """
    match = re.fullmatch(r"([A-Za-z]?)(\d+)([A-Za-z]?)", res_id.strip())
    if match is None:
        raise ValueError(f'Cannot parse residue "{res_id}". Use e.g. "270A", "A270" or "270".')
    chain = match.group(1) or match.group(3) or None
    return chain, int(match.group(2))


def get_mutation_targets(pdb_path: str, lig_path: dict, fix_residues: list = None, radius: float = None):
    """Pick the crop residues that saturation mutagenesis is applied to.

    Args:
        pdb_path: Path to the input PDB/CIF.
        lig_path: {"LIG": "path/to/ligand.mol2", ...} or None.
        fix_residues: Residue identifiers ("270A", ...) kept as wildtype,
            e.g. catalytic residues.
        radius: Optional distance filter passed on to get_crop_residues().

    Return:
        List of (chain, resno, name3) to be mutated.
    """
    fixed = [parse_residue_id(r) for r in fix_residues] if fix_residues else []

    targets = []
    for chain, resno, name3 in get_crop_residues(pdb_path, lig_path, radius=radius):
        if any(resno == f_resno and f_chain in (None, chain) for f_chain, f_resno in fixed):
            continue
        targets.append((chain, resno, name3))

    return targets


def mutation_name(pdb_path: str, chain: str, resno: int, wt3: str, mut3: str):
    """Output name prefix of a single point mutant, e.g. "<structure>_A_S270W"."""
    return f"{structure_name(pdb_path)}_{chain}_{THREE_TO_ONE[wt3]}{resno}{THREE_TO_ONE[mut3]}"


def main():
    parser = argparse.ArgumentParser(
        description="Run PLACER on a directory of structures, or run single-residue "
                    "saturation mutagenesis on the active site of one structure.")
    parser.add_argument("-i", "--input", "--input_dir", dest="input", type=str, required=True,
                        help="Directory containing input PDB files (one PLACER run per file), "
                             "OR a single PDB file (saturation mutagenesis over its crop residues)")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save PLACER outputs")
    parser.add_argument("-n", "--iterations", type=int, default=50,
                        help="Number of iterations for PLACER (default: 50)")
    parser.add_argument("-l", "--lig_path", type=str, default=None, nargs="+",
                        help="Ligand path(s) in format 'LIG:path/to/ligand.mol2' (optional, multiple allowed)")
    parser.add_argument("-f", "--fix_residue", type=str, default=None, nargs="+",
                        help="Saturation mutagenesis only: residues kept as wildtype, "
                             "e.g. '270A 274A' (catalytic residues)")
    parser.add_argument("-r", "--radius", type=float, default=None,
                        help="Saturation mutagenesis only: further restrict the targets to crop "
                             "residues with a heavy atom within this distance (A) of any ligand atom")
    parser.add_argument("-wt", "--include_wt", action="store_true",
                        help="Saturation mutagenesis only: also run the unmutated structure as a reference")
    parser.add_argument("--overwrite", action="store_true",
                        help="Saturation mutagenesis only: rerun mutants whose outputs already exist")
    parser.add_argument("--list_only", action="store_true",
                        help="Saturation mutagenesis only: print the target residues and exit")
    args = parser.parse_args()

    # parse lig_path: ["LIG:path/to/lig1.mol2", "AMP:path/to/lig2.mol2"] → {"LIG": "...", "AMP": "..."}
    if args.lig_path:
        lig_path = {k.strip(): v.strip() for item in args.lig_path for k, v in [item.split(":")]}
    else:
        lig_path = None

    if os.path.isdir(args.input):
        run_batch(args.input, lig_path, args)
    else:
        run_saturation_mutagenesis(args.input, lig_path, args)


def run_batch(input_dir: str, lig_path: dict, args):
    """Run PLACER once for every PDB file in a directory."""
    pdb_names = [f for f in os.listdir(input_dir) if f.endswith(".pdb")]
    total = len(pdb_names)
    start_time = time.time()

    for idx, name in enumerate(pdb_names, start=1):
        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] START: {name}")
        print(f"{'#' * 70}")

        cycle_start = time.time()
        pdb_path = os.path.join(input_dir, name)
        run_placer(pdb_path, lig_path, args.output_dir, iterations=args.iterations)

        elapsed = time.time() - start_time
        cycle_time = time.time() - cycle_start
        avg = elapsed / idx
        eta = avg * (total - idx)
        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] DONE: {name}")
        print(f"# This file: {cycle_time / 60:.1f} min  "
              f"| Elapsed: {elapsed / 60:.1f} min  | Remaining: {eta / 60:.1f} min")
        print(f"{'#' * 70}")


def run_saturation_mutagenesis(pdb_path: str, lig_path: dict, args):
    """Run PLACER for every single point mutant of the active site of one structure.

    Every crop residue that is not fixed by the user is mutated to the other 19
    canonical amino acids, one PLACER run each. The mutation is applied inside
    PLACER, so all mutants share the wildtype backbone.
    """
    targets = get_mutation_targets(pdb_path, lig_path,
                                   fix_residues=args.fix_residue, radius=args.radius)

    print(f"\n{'#' * 70}")
    print(f"# Saturation mutagenesis: {structure_name(pdb_path)}")
    print(f"# Target residues ({len(targets)}): "
          f"{', '.join(f'{THREE_TO_ONE[n]}{r}{c}' for c, r, n in targets)}")
    print(f"# PLACER runs: {len(targets) * 19}{' (+1 wildtype)' if args.include_wt else ''}")
    print(f"{'#' * 70}")

    if args.list_only:
        return

    jobs = []
    if args.include_wt:
        jobs.append((None, structure_name(pdb_path)))
    for chain, resno, wt3 in targets:
        for mut3 in AA3:
            if mut3 == wt3:
                continue
            jobs.append(((chain, resno, mut3), mutation_name(pdb_path, chain, resno, wt3, mut3)))

    total = len(jobs)
    start_time = time.time()

    for idx, (mutation, name) in enumerate(jobs, start=1):
        if not args.overwrite and os.path.exists(f"{args.output_dir}/{name}_model.pdb"):
            print(f"# [{idx}/{total}] SKIP (exists): {name}")
            continue

        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] START: {name}")
        print(f"{'#' * 70}")

        cycle_start = time.time()
        run_placer(pdb_path, lig_path, args.output_dir, iterations=args.iterations,
                   mutation=mutation, name=name)

        elapsed = time.time() - start_time
        cycle_time = time.time() - cycle_start
        avg = elapsed / idx
        eta = avg * (total - idx)
        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] DONE: {name}")
        print(f"# This mutant: {cycle_time / 60:.1f} min  "
              f"| Elapsed: {elapsed / 60:.1f} min  | Remaining: {eta / 60:.1f} min")
        print(f"{'#' * 70}")


if __name__ == '__main__':
    main()
