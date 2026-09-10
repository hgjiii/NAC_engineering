import sys, os
import json
import time
import warnings
warnings.filterwarnings("ignore")
import argparse
import traceback

# DIR = os.getcwd()
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f"{DIR}/PLACER")
sys.path.insert(0, DIR)
import PLACER

from process.placer_scoring import parse_exclusions, install as install_score_mask

## Initializing PLACER model with default checkpoint
placer = PLACER.PLACER()


def load_residue_templates(paths):
    """Merge residue-library entries written by make_residue_template.py.

    Args:
        paths: List of JSON file paths, each holding {name3: entry}.

    Return:
        One combined dict, empty when paths is None.

    Raise:
        SystemExit on a missing file, unreadable JSON, or two files defining
        the same residue name differently.
    """
    templates = {}
    for path in paths or []:
        if not os.path.isfile(path):
            raise SystemExit(f"Residue template not found: {path}")
        try:
            entries = json.load(open(path))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path} is not valid JSON: {exc}")

        for name, entry in entries.items():
            if templates.get(name, entry) != entry:
                raise SystemExit(f"Residue '{name}' is defined differently in "
                                 f"more than one template.")
            templates[name] = entry

    return templates


def parse_modified(tokens):
    """Parse -mr tokens into residue selectors.

    A token names the residue, its position, or both: "XHB", "A296" or
    "XHB:A296".

    Args:
        tokens: Raw CLI tokens.

    Return:
        List of (name3, chain, resnum) with None wherever unspecified.

    Raise:
        SystemExit on a token that parses as neither.
    """
    selectors = []
    for token in tokens or []:
        name3, _, position = token.strip().upper().rpartition(":")
        if not position:
            raise SystemExit(f"Empty --modified_residue token in '{token}'.")

        if position.isalpha():
            # A bare residue name, e.g. "XHB".
            name3, chain, resnum = (name3 or position), None, None
        elif position[:1].isalpha() and position[1:].isdigit():
            chain, resnum = position[0], int(position[1:])
        elif position.isdigit():
            chain, resnum = None, int(position)
        else:
            raise SystemExit(
                f"Cannot read --modified_residue '{token}'. Use a residue "
                "name (XHB), a position (A296 or 296), or both (XHB:A296).")

        selectors.append((name3 or None, chain, resnum))
    return selectors


def promote_to_atom(pdbstr, selectors):
    """Rewrite a modified residue's HETATM records as ATOM records.

    PLACER splits protein from ligand on the record type alone: parseProtein
    reads ATOM, parse_ligand_from_pdb_to_obmol reads HETATM. Rosetta writes a
    patched residue as HETATM, so without this the modified residue is pulled
    out of the chain and handed to the ligand parser instead.

    Args:
        pdbstr: Contents of the PDB file.
        selectors: Output of parse_modified.

    Return:
        Tuple of (rewritten text, list of (name3, chain, resnum) matched).
    """
    matched, lines = [], pdbstr.split("\n")

    for i, line in enumerate(lines):
        if not line.startswith("HETATM"):
            continue
        name3, chain, resnum = line[17:20].strip(), line[21], line[22:26].strip()
        if not resnum.lstrip("-").isdigit():
            continue
        resnum = int(resnum)

        for want_name, want_chain, want_num in selectors:
            if want_name is not None and want_name != name3:
                continue
            if want_chain is not None and want_chain != chain:
                continue
            if want_num is not None and want_num != resnum:
                continue
            lines[i] = "ATOM  " + line[6:]
            if (name3, chain, resnum) not in matched:
                matched.append((name3, chain, resnum))
            break

    return "\n".join(lines), matched


def parse_center(token):
    """Split a -c token into the residue part and the atom names.

    The residue is named the way the rest of the pipeline names one: either
    by residue name (UQ3, XHB) or by position (X1, A296). Which of the two it
    is gets settled against the structure in resolve_center, since a name and
    a position can look alike.

    Args:
        token: "UQ3", "UQ3@C01,C02,C03", "X1@S05" or a point "12.3,4.5,-6.7".

    Return:
        Tuple of (residue spec, atom names), or ("coord", (x, y, z)).

    Raise:
        SystemExit on an empty residue part.
    """
    token = token.strip()

    parts = token.split(",")
    if len(parts) == 3 and "@" not in token:
        try:
            return "coord", tuple(float(p) for p in parts)
        except ValueError:
            pass

    position, _, atom_names = token.upper().partition("@")
    if not position:
        raise SystemExit(
            f"Cannot read --center '{token}'. Use a residue (UQ3 or X1), a "
            "residue with atoms (UQ3@C01,C02,C03) or a point (12.3,4.5,-6.7).")

    return position, [a.strip() for a in atom_names.split(",") if a.strip()]


def find_residue(pdbstr, spec):
    """Find the residue a -c token names, by residue name or by position.

    Args:
        pdbstr: Contents of the PDB file, after any promotion.
        spec: "UQ3" or "X1", upper-cased.

    Return:
        Tuple of (its atom lines, chain, resnum, name3).

    Raise:
        SystemExit if nothing matches, or if a residue name occurs more than
        once and the position is needed to tell the copies apart.
    """
    lines = [l for l in pdbstr.split("\n") if l.startswith(("ATOM", "HETATM"))]

    # A position, e.g. X1 or A296.
    if spec[:1].isalpha() and spec[1:].isdigit():
        chain, resnum = spec[0], int(spec[1:])
        picked = [l for l in lines
                  if l[21] == chain and l[22:26].strip() == str(resnum)]
        if picked:
            return picked, chain, resnum, picked[0][17:20].strip()

    # A residue name, e.g. UQ3 or XHB.
    picked = [l for l in lines if l[17:20].strip() == spec]
    copies = sorted({(l[21], int(l[22:26])) for l in picked})
    if len(copies) > 1:
        listed = ", ".join(f"{c}{n}" for c, n in copies)
        raise SystemExit(f"--center: '{spec}' occurs {len(copies)} times "
                         f"({listed}). Name the position instead.")
    if copies:
        chain, resnum = copies[0]
        return picked, chain, resnum, spec

    raise SystemExit(f"--center: no residue '{spec}' in the structure.")


def resolve_center(pdbstr, selector):
    """Turn a -c selector into the entry PLACER's crop_centers expects.

    A single atom is handed over as an atom key so PLACER resolves it itself;
    several atoms, or a whole residue, collapse to their centroid. Either way
    it is one point, which is what keeps every model on the same crop --
    PLACER draws at random from whatever list it is given.

    Args:
        pdbstr: Contents of the PDB file, after any promotion.
        selector: Output of parse_center.

    Return:
        Either a (chain, resno, name3, atom_name) tuple or an (x, y, z) point.

    Raise:
        SystemExit if the residue or any named atom is not present.
    """
    spec, value = selector
    if spec == "coord":
        return value

    residue, chain, resnum, name3 = find_residue(pdbstr, spec)
    available = {l[12:16].strip(): l for l in residue}

    if value:
        missing = [a for a in value if a not in available]
        if missing:
            raise SystemExit(f"--center: {name3} {chain}{resnum} has no "
                             f"atom(s) {missing}. Available: "
                             f"{sorted(available)}")
        if len(value) == 1:
            return (chain, resnum, name3, value[0])
        picked = [available[a] for a in value]
    else:
        picked = [l for l in residue if l[76:78].strip() != "H"]

    return tuple(sum(float(l[30 + 8 * i:38 + 8 * i]) for l in picked) / len(picked)
                 for i in range(3))


def run_placer(pdb_path: str, lig_path: dict, output_dir: str, iterations: int = 50,
               templates: dict = None, modified: list = None, center=None):

    assert isinstance(lig_path, dict) or lig_path == None, 'The argument "lig_path" must be a dictionary type and have the format of {"LIG": "path/to/ligand", ...} (or be a None type)'

    try:
        pl_inp = PLACER.PLACERinput()

        # A modified residue is promoted in memory, so the file on disk keeps
        # the HETATM records the rest of the pipeline expects.
        pdbstr = open(pdb_path).read()
        matched = []
        if modified:
            pdbstr, matched = promote_to_atom(pdbstr, modified)
            if not matched:
                print("Warning: --modified_residue matched nothing in "
                      f"{os.path.basename(pdb_path)}; nothing was promoted.")
            else:
                listed = ", ".join(f"{n} {c}{r}" for n, c, r in matched)
                print(f"# treating as part of the chain: {listed}")

        pl_inp.pdb(pdbstr)
        pl_inp.name(os.path.basename(pdb_path).replace(".pdb" if ".pdb" in pdb_path else ".cif", ""))

        # PLACER merges custom residues into one library shared by the whole
        # process and refuses a name that is already there, so past the first
        # structure only the ones it has not seen are offered.
        unseen = {name: entry for name, entry in (templates or {}).items()
                  if name not in placer.mols()}
        if unseen:
            pl_inp.add_custom_residues(unseen)

        if center is not None:
            # One centre only: PLACER draws at random from the list it is
            # given, so a single entry is what keeps every model on the same
            # crop.
            resolved = resolve_center(pdbstr, center)
            pl_inp.crop_centers([resolved])
            print(f"# crop centre: {resolved}")

        # pl_inp.ligand_reference({"AMP": amp_name, "SIN": sucfile})
        if lig_path is None:
            pl_inp.exclude_sm(True)
        else:
            pl_inp.ignore_ligand_hydrogens(True)
            pl_inp.ligand_reference(lig_path)

        outputs = placer.run(pl_inp, iterations)

        os.makedirs(output_dir, exist_ok=True)
        PLACER.protocol.dump_output(outputs, f"{output_dir}/{pl_inp.name()}", rerank="prmsd")

    except Exception as e:
        # One bad structure must not take the batch down with it: a relaxed
        # model whose ligand geometry drifted far enough to perceive an extra
        # bond, for instance, fails deep inside PLACER's parser. Report it,
        # hand the traceback back for the error log, and carry on with the
        # next file. KeyboardInterrupt is not an Exception, so Ctrl-C still
        # stops the run.
        print(f"# FAILED: {os.path.basename(pdb_path)}: "
              f"{type(e).__name__}: {e or '(no message)'}")
        return traceback.format_exc()

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir",  type=str, required=True,
                        help="Directory containing input PDB files")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save PLACER outputs")
    parser.add_argument("-n", "--iterations", type=int, default=50,
                        help="Number of iterations for PLACER (default: 50)")
    parser.add_argument("-l", "--lig_path", type=str, default=None, nargs="+",
                        help="Ligand path(s) in format 'LIG:path/to/ligand.mol2' (optional, multiple allowed)")
    parser.add_argument("-rt", "--residue_template", type=str, default=None, nargs="+",
                        help="Residue library entry JSON(s) from "
                             "make_residue_template.py, for residues PLACER "
                             "does not know (optional, multiple allowed).")
    parser.add_argument("-mr", "--modified_residue", type=str, default=None, nargs="+",
                        help="Modified residue(s) written as HETATM that "
                             "belong to the protein chain, given as a residue "
                             "name (XHB), a position (A296 or 296) or both "
                             "(XHB:A296). Their records are promoted to ATOM "
                             "in memory so PLACER reads them as chain, not "
                             "ligand; the file on disk is untouched.")
    parser.add_argument("-xs", "--exclude_score", type=str, default=None, nargs="+",
                        help="Atom(s) to leave out of the score terms, named "
                             "like --center: a residue (UQ3) or some of its "
                             "atoms (UQ3@N36,C37,N38). They are still "
                             "predicted; only prmsd, plddt, plddt_pde and the "
                             "loss terms ignore them, so the CSV keeps its "
                             "usual columns.")
    parser.add_argument("-c", "--center", type=str, default=None,
                        help="Pin the crop to one place: a residue by name "
                             "(UQ3) or position (X1), some of its atoms "
                             "(UQ3@C01,C02,C03), or a point (12.3,4.5,-6.7). "
                             "Several atoms collapse to their centroid, so "
                             "the crop is the same for every model. Left "
                             "unset, PLACER crops around the ligand as it "
                             "does by default.")
    args = parser.parse_args()

    templates = load_residue_templates(args.residue_template)
    modified = parse_modified(args.modified_residue)
    center = parse_center(args.center) if args.center else None
    install_score_mask(PLACER, placer, parse_exclusions(args.exclude_score))


    # parse lig_path: ["LIG:path/to/lig1.mol2", "AMP:path/to/lig2.mol2"] → {"LIG": "...", "AMP": "..."}
    if args.lig_path:
        lig_path = {k.strip(): v.strip() for item in args.lig_path for k, v in [item.split(":")]}
    else:
        lig_path = None

    pdb_names = [f for f in os.listdir(args.input_dir) if f.endswith(".pdb")]
    total = len(pdb_names)
    start_time = time.time()

    failures = []

    for idx, name in enumerate(pdb_names, start=1):
        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] START: {name}")
        print(f"{'#' * 70}")

        cycle_start = time.time()
        pdb_path = os.path.join(args.input_dir, name)
        failure = run_placer(pdb_path, lig_path, args.output_dir,
                             iterations=args.iterations, templates=templates,
                             modified=modified, center=center)
        if failure is not None:
            failures.append((name, failure))

        elapsed = time.time() - start_time
        cycle_time = time.time() - cycle_start
        avg = elapsed / idx
        eta = avg * (total - idx)
        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] {'FAILED' if failure else 'DONE'}: {name}")
        print(f"# This file: {cycle_time / 60:.1f} min  "
              f"| Elapsed: {elapsed / 60:.1f} min  | Remaining: {eta / 60:.1f} min")
        print(f"{'#' * 70}")

    if failures:
        # Only written when something went wrong, so its presence is itself
        # the signal that the batch was not clean.
        log_path = os.path.join(args.output_dir, "error.log")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"{len(failures)} of {total} structures failed\n")
            for name, trace in failures:
                f.write(f"\n{'=' * 70}\n{name}\n{'=' * 70}\n{trace}")

        print(f"\n{len(failures)} of {total} structures failed:")
        for name, _trace in failures:
            print(f"  {name}")
        print(f"Wrote {log_path}")

if __name__ == '__main__':
    main()