"""Generate a Rosetta patch for a chemically modified residue from a structure.

A modified residue -- an acyl-enzyme intermediate, a phosphorylated serine,
an alkylated lysine -- is a canonical amino acid plus a few extra atoms. It
must stay a POLYMER residue type: turned into a ligand params file it loses
its LOWER/UPPER connections, the chain breaks at that position and Rosetta
caps the neighbours with charged termini.

A patch keeps it a polymer. It says "take this canonical residue, add these
atoms", so the backbone, the polymer connections, the termini variants and
the parent's backbone-dependent Dunbrack rotamer library all carry over.
This script writes that patch, deducing everything from the structure:

    parent residue   guessed from the atom names (or forced with -p)
    added atoms      whatever is not in the parent's own params
    bonds            perceived with Open Babel
    atom types       molfile_to_params' Rosetta typing rules
    ICOOR            measured off the model, one entry per added atom
    CHI              one per rotatable bond in the added group

Nothing outside the structure file and the Rosetta database is needed; no
per-ligand template.

Load the result with -extra_patch_fa. Note that a patch cannot go through
the params route that run_fastrelax.py/run_mutation.py use for ligands, so
those scripts need to pass it at pyrosetta.init() time.

Usage:
    python pipeline/make_residue_patch.py -i model.pdb -r XHB -o xhb.txt
    python pipeline/make_residue_patch.py -i model.pdb -r SEP -o sep.txt -p SER
"""

import os
import sys
import glob
import tempfile
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process.parsing import (read_residue, build_obmol, harvest_obmol,
                             render_mol2, find_carbonyl_carbons,
                             mol2_bond_type, distance, angle, dihedral)

# VariantTypes Rosetta already knows, tried in this order. The name only has
# to be unique per parent residue; it carries no chemistry of its own.
VARIANT_CANDIDATES = ("ACETYLATION", "METHYLATION", "DIMETHYLATION",
                      "TRIMETHYLATION", "PHOSPHORYLATION", "SULFATION",
                      "SIDECHAIN_CONJUGATION", "SPECIAL_ROT")


def find_rosetta_database(hint):
    """Locate the fa_standard residue type set.

    Args:
        hint: Explicit path to the fa_standard directory, or None to search
            the installed pyrosetta packages.

    Return:
        Path to the fa_standard directory.

    Raise:
        SystemExit if it cannot be found.
    """
    if hint:
        if os.path.isdir(os.path.join(hint, "residue_types")):
            return hint
        raise SystemExit(f"{hint} does not look like a fa_standard directory.")

    try:
        import pyrosetta
        base = os.path.join(os.path.dirname(pyrosetta.__file__), "database")
        found = os.path.join(base, "chemical/residue_type_sets/fa_standard")
        if os.path.isdir(found):
            return found
    except ImportError:
        pass

    pattern = os.path.join(os.path.expanduser("~"), "*", "envs", "*", "lib",
                           "python*", "site-packages", "pyrosetta", "database",
                           "chemical", "residue_type_sets", "fa_standard")
    for found in sorted(glob.glob(pattern)):
        if os.path.isdir(os.path.join(found, "residue_types")):
            return found

    raise SystemExit("Could not find the Rosetta fa_standard database. "
                     "Pass it with --rosetta-db.")


def parse_params(path):
    """Read the parts of a Rosetta residue params file this script needs.

    Args:
        path: Path to a .params file.

    Return:
        Dict with name, name1, atoms (name -> (rosetta type, mm type)),
        order (atom names in file order), bonds (list of name pairs) and
        chis (index -> four atom names).
    """
    out = {"name": os.path.splitext(os.path.basename(path))[0], "name1": "X",
           "atoms": {}, "order": [], "bonds": [], "chis": {}}

    for line in open(path):
        field = line.split()
        if not field:
            continue
        if field[0] == "IO_STRING" and len(field) >= 3:
            out["name1"] = field[2]
        elif field[0] == "ATOM" and len(field) >= 3:
            out["atoms"][field[1]] = (field[2], field[3] if len(field) > 3 else "X")
            out["order"].append(field[1])
        elif field[0] in ("BOND", "BOND_TYPE") and len(field) >= 3:
            out["bonds"].append((field[1], field[2]))
        elif field[0] == "CHI" and len(field) >= 6:
            out["chis"][int(field[1])] = tuple(field[2:6])

    return out


def load_parents(db):
    """Read every canonical amino acid params file in the database.

    Args:
        db: fa_standard directory.

    Return:
        Dict mapping residue name to the parse_params result.
    """
    library = {}
    for path in glob.glob(os.path.join(db, "residue_types", "l-caa", "*.params")):
        parsed = parse_params(path)
        library[parsed["name"]] = parsed
    return library


def pick_parent(observed, library, hint):
    """Decide which canonical residue the modified one is built on.

    Args:
        observed: Heavy atom names seen in the structure.
        library: Output of load_parents.
        hint: Residue name forced on the command line, or None.

    Return:
        The chosen entry from library.

    Raise:
        SystemExit if no parent is contained in the observed atoms.
    """
    if hint:
        if hint not in library:
            raise SystemExit(f"No params for parent residue '{hint}' in the "
                             f"database. Known: {sorted(library)}")
        parent = library[hint]
        missing = sorted(heavy_atoms(parent) - observed)
        if missing:
            raise SystemExit(f"Residue is missing {parent['name']} atom(s) "
                             f"{missing}; it cannot be a modified {hint}.")
        return parent

    matches = [p for p in library.values() if heavy_atoms(p) <= observed]
    if not matches:
        raise SystemExit(
            "No canonical residue's heavy atoms are a subset of this one. "
            "Force the parent with -p, or check that the residue really is a "
            "modified amino acid rather than a free ligand.")
    return max(matches, key=lambda p: len(heavy_atoms(p)))


def heavy_atoms(parsed):
    """Heavy atom names of a parsed params file.

    Args:
        parsed: Output of parse_params.

    Return:
        Set of atom names whose Rosetta type is not a hydrogen or virtual.
    """
    return {name for name, (ros, _mm) in parsed["atoms"].items()
            if not ros.startswith("H") and ros != "VIRT"}


def perceive(atoms, resname):
    """Perceive bonds and add hydrogens with Open Babel.

    Args:
        atoms: List of (name, element, x, y, z) read from the structure.
        resname: Residue name, stamped on the scratch PDB.

    Return:
        Tuple of (atom list including hydrogens, bond row list).
    """
    ob, mol = build_obmol(atoms, resname)
    mol.AddHydrogens(False, False)
    return harvest_obmol(ob, mol, [a[0] for a in atoms])


def rosetta_typing(atoms, bonds, resname, net_charge):
    """Run molfile_to_params' typing rules over the residue.

    Args:
        atoms: List of (name, element, x, y, z), hydrogens included.
        bonds: List of (name1, name2, order, aromatic).
        resname: Residue name for the scratch mol2.
        net_charge: Net charge to normalise the partial charges to.

    Return:
        Dict mapping atom name to (rosetta type, partial charge).
    """
    import molfile_to_params as mtp
    from rosetta_py.io.mdl_molfile import read_tripos_mol2, find_rings

    index = {a[0]: n for n, a in enumerate(atoms, start=1)}
    aromatic = {n for b in bonds if b["pdbx_aromatic_flag"] == "Y"
                for n in (b["atom_id_1"], b["atom_id_2"])}
    records = [(n, e, n in aromatic, x, y, z) for n, e, x, y, z in atoms]
    carbonyl = find_carbonyl_carbons(bonds, {a[0]: a[1] for a in atoms})
    pairs = [(index[b["atom_id_1"]], index[b["atom_id_2"]],
              mol2_bond_type(b, {a[0]: a[1] for a in atoms}, carbonyl))
             for b in bonds]

    handle, tmp = tempfile.mkstemp(suffix=".mol2")
    os.close(handle)
    try:
        with open(tmp, "w") as f:
            f.write(render_mol2(resname, records, pairs))
        molfile = read_tripos_mol2(tmp)[0]
    finally:
        os.unlink(tmp)

    find_rings(molfile.bonds)
    mtp.add_fields_to_atoms(molfile.atoms)
    mtp.add_fields_to_bonds(molfile.bonds)
    mtp.find_virtual_atoms(molfile.atoms)
    mtp.assign_rosetta_types(molfile.atoms)
    mtp.assign_partial_charges(molfile.atoms, net_charge, True)

    return {a.name.strip(): (a.ros_type.strip(), a.partial_charge)
            for a in molfile.atoms}


def variants_in_use(db, parent):
    """Collect the VariantTypes the database already patches onto a residue.

    Two patches declaring the same VariantType for the same parent leave the
    packer with residue types it cannot reconcile, and building a PackerTask
    dies on an assertion rather than a readable error. Reusing Rosetta's own
    ACETYLATION on CYS, for instance, collides with cys_acetylated.txt.

    Args:
        db: fa_standard directory.
        parent: Parent residue name, e.g. "CYS".

    Return:
        Set of VariantType names already taken for that parent.
    """
    taken = set()
    for path in glob.iglob(os.path.join(db, "patches", "**", "*.txt"),
                           recursive=True):
        types, names = None, set()
        for line in open(path):
            field = line.split()
            if not field:
                continue
            if field[0] == "TYPES" and types is None:
                types = field[1:]
            elif field[0] == "NAME3":
                names.update(field[1:])
        if types and parent in names:
            taken.update(types)
    return taken


def choose_variant(db, parent, requested):
    """Pick a VariantType that is free for this parent residue.

    Args:
        db: fa_standard directory.
        parent: Parent residue name.
        requested: Name forced on the command line, or None.

    Return:
        Tuple of (variant name, set of names already taken).

    Raise:
        SystemExit if every candidate is taken.
    """
    taken = variants_in_use(db, parent)
    if requested:
        return requested, taken

    for candidate in VARIANT_CANDIDATES:
        if candidate not in taken:
            return candidate, taken

    raise SystemExit(
        f"Every candidate VariantType is already patched onto {parent} "
        f"({sorted(taken)}). Pass an unused one with --variant.")


def mm_types_from_database(db):
    """Collect the Rosetta-type to MM-type mapping the database itself uses.

    Args:
        db: fa_standard directory.

    Return:
        Dict mapping Rosetta atom type to the commonest MM type.
    """
    tally = {}
    for path in glob.glob(os.path.join(db, "residue_types", "*", "*.params")):
        for line in open(path):
            field = line.split()
            if field[:1] == ["ATOM"] and len(field) >= 4:
                tally.setdefault(field[2], {})
                tally[field[2]][field[3]] = tally[field[2]].get(field[3], 0) + 1
    return {ros: max(counts, key=counts.get) for ros, counts in tally.items()}





def build_tree(added, attach_parent, attach_atom, neighbours):
    """Breadth-first order the added atoms away from the attachment.

    Args:
        added: Set of added atom names.
        attach_parent: Parent-residue atom the group hangs off.
        attach_atom: First added atom.
        neighbours: Dict mapping atom name to its bonded atom names.

    Return:
        Tuple of (ordered atom names, dict of atom -> its predecessor).
    """
    order, previous = [attach_atom], {attach_atom: attach_parent}
    queue = [attach_atom]
    while queue:
        current = queue.pop(0)
        heavy = [n for n in neighbours[current]
                 if n in added and n not in previous and not n.startswith("H")]
        light = [n for n in neighbours[current]
                 if n in added and n not in previous and n.startswith("H")]
        for nxt in heavy + light:
            previous[nxt] = current
            order.append(nxt)
            queue.append(nxt)
    return order, previous


def reference_atoms(atom, previous, parent, attach_parent, added):
    """Find the three reference atoms an ICOOR entry is measured against.

    Args:
        atom: Atom to place.
        previous: Predecessor map from build_tree.
        parent: Parsed parent params.
        attach_parent: Parent-residue atom the group hangs off.
        added: Set of added atom names.

    Return:
        Tuple of (parent, grandparent, great-grandparent) atom names.
    """
    par = previous[atom]
    if par in added:
        gpa = previous[par]
        gga = previous[gpa] if gpa in added else _parent_step(parent, gpa, par)
    else:
        gpa = _parent_step(parent, par, atom)
        gga = _parent_step(parent, gpa, par)
    return par, gpa, gga


def _parent_step(parent, atom, exclude):
    """Walk one bond further into the parent residue, towards the backbone.

    Args:
        parent: Parsed parent params.
        atom: Atom to step away from.
        exclude: Atom not to step back onto.

    Return:
        A neighbouring heavy atom name.

    Raise:
        SystemExit if the parent graph runs out of heavy atoms.
    """
    heavy = heavy_atoms(parent)
    rank = {name: n for n, name in enumerate(parent["order"])}
    options = [b for a, b in parent["bonds"] if a == atom] + \
              [a for a, b in parent["bonds"] if b == atom]
    options = [o for o in options if o in heavy and o != exclude]
    if not options:
        raise SystemExit(f"Cannot find a reference atom past {atom} in "
                         f"{parent['name']}; the modification site is too "
                         "close to the chain end for automatic ICOOR.")
    return min(options, key=lambda o: rank.get(o, 99))


def main():
    parser = argparse.ArgumentParser(
        description="Write a Rosetta patch for a modified residue, deduced "
                    "from a structure file.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Structure file holding the residue (.pdb or .cif).")
    parser.add_argument("-r", "--residue", type=str, required=True,
                        help="Residue name of the modified residue, e.g. XHB.")
    parser.add_argument("-o", "-O", "--output", type=str, default=None,
                        help="Output patch path (default: <residue>.txt beside "
                             "the structure).")
    parser.add_argument("-p", "--parent", type=str, default=None,
                        help="Canonical residue it is a modification of "
                             "(default: guessed from the atom names).")
    parser.add_argument("-c", "--chain", type=str, default=None,
                        help="Chain id, when the residue occurs more than once.")
    parser.add_argument("-n", "--resnum", type=str, default=None,
                        help="Residue number, when it occurs more than once.")
    parser.add_argument("--variant", type=str, default=None,
                        help="Rosetta VariantType to tag the patch with "
                             "(default: the first candidate not already "
                             "patched onto the parent). It must be a name "
                             "Rosetta knows -- an unregistered one crashes it.")
    parser.add_argument("--patch-name", type=str, default=None,
                        help="NAME field of the patch (default: derived from "
                             "the residue name).")
    parser.add_argument("--charge", type=float, default=0.0,
                        help="Net charge of the added group (default 0).")
    parser.add_argument("--rosetta-db", type=str, default=None,
                        help="fa_standard directory (default: found via the "
                             "installed pyrosetta).")
    args = parser.parse_args()

    code = args.residue.strip().upper()
    db = find_rosetta_database(args.rosetta_db)
    library = load_parents(db)

    structure_atoms = read_residue(args.input, code, args.chain, args.resnum)
    observed = {a[0] for a in structure_atoms}
    parent = pick_parent(observed, library, args.parent)

    variant, taken = choose_variant(db, parent["name"], args.variant)

    atoms, bonds = perceive(structure_atoms, code)
    coords = {a[0]: (a[2], a[3], a[4]) for a in atoms}
    element = {a[0]: a[1] for a in atoms}

    neighbours = {a[0]: [] for a in atoms}
    for bond in bonds:
        neighbours[bond["atom_id_1"]].append(bond["atom_id_2"])
        neighbours[bond["atom_id_2"]].append(bond["atom_id_1"])

    # The added group is the heavy atoms the parent does not have, plus the
    # hydrogens hanging off them. Open Babel also caps the parent's severed
    # backbone with hydrogens; those belong to the parent and are dropped.
    parent_names = set(parent["atoms"])
    added = {a[0] for a in atoms
             if a[0] not in parent_names and element[a[0]] != "H"}
    if not added:
        raise SystemExit(f"{code} has no atoms beyond {parent['name']}; "
                         "there is nothing to patch.")
    added |= {a[0] for a in atoms
              if element[a[0]] == "H" and a[0] not in parent_names
              and any(n in added for n in neighbours[a[0]])}

    links = [(bond["atom_id_1"], bond["atom_id_2"]) for bond in bonds
             if (bond["atom_id_1"] in added) != (bond["atom_id_2"] in added)]
    links = [(q, p) if p in added else (p, q) for p, q in links]
    anchors = sorted({p for p, _q in links})
    if len(anchors) != 1 or len(links) != 1:
        raise SystemExit(
            f"Expected exactly one bond joining the added group to "
            f"{parent['name']}, found {len(links)} at {anchors}. A ring "
            "closing back onto the parent is not supported.")
    attach_parent, attach_atom = links[0]

    types = rosetta_typing(atoms, bonds, code, args.charge)
    mm_map = mm_types_from_database(db)
    added_charges = {n: types[n][1] for n in added}
    shift = (args.charge - sum(added_charges.values())) / len(added)

    order, previous = build_tree(added, attach_parent, attach_atom, neighbours)
    if set(order) != added:
        raise SystemExit("The added atoms are not all reachable from the "
                         "attachment point; the residue may be disconnected.")

    virtual_h = [n for n in neighbours.get(attach_parent, [])
                 if n in parent_names
                 and parent["atoms"][n][0].startswith("H")]
    if not virtual_h:
        virtual_h = [b for a, b in parent["bonds"] if a == attach_parent
                     and parent["atoms"].get(b, ("", ""))[0].startswith("H")] + \
                    [a for a, b in parent["bonds"] if b == attach_parent
                     and parent["atoms"].get(a, ("", ""))[0].startswith("H")]

    redefine = None
    for number, chi in parent["chis"].items():
        if virtual_h and chi[3] == virtual_h[0]:
            redefine = (number, chi[:3] + (attach_atom,))

    lines = [f"## {code}: {parent['name']} carrying a {len(added)}-atom "
             "modification.",
             f"## Generated by make_residue_patch.py from "
             f"{os.path.basename(args.input)}.",
             "",
             f"NAME {args.patch_name or code.lower() + 'mod'}",
             f"TYPES {variant}",
             "",
             "BEGIN_SELECTOR",
             "PROPERTY PROTEIN",
             f"NAME3 {parent['name']}",
             f"NOT VARIANT_TYPE {variant}",
             "NOT VARIANT_TYPE DISULFIDE",
             "NOT VARIANT_TYPE SIDECHAIN_CONJUGATION",
             "END_SELECTOR",
             "",
             "BEGIN_CASE",
             "",
             f"SET_IO_STRING {code} {parent['name1']}",
             f"SET_INTERCHANGEABILITY_GROUP {code}"]

    if virtual_h:
        lines += ["", f"SET_ATOM_TYPE {virtual_h[0]:<4} VIRT"]

    lines.append("")
    for name in order:
        ros, charge = types[name][0], types[name][1] + shift
        lines.append("ADD_ATOM %-4s %-4s %-4s %7.3f"
                     % (name, ros, mm_map.get(ros, "X"), charge))

    lines.append("")
    lines.append("ADD_BOND %-4s %-4s" % (attach_parent, attach_atom))
    for name in order:
        if previous[name] != attach_parent:
            lines.append("ADD_BOND %-4s %-4s" % (previous[name], name))

    lines.append("")
    for name in order:
        par, gpa, gga = reference_atoms(name, previous, parent,
                                        attach_parent, added)
        lines.append("SET_ICOOR %-4s %11.2f %10.2f %9.3f   %-4s %-4s %-4s"
                     % (name,
                        dihedral(coords[gga], coords[gpa], coords[par],
                                 coords[name]),
                        180.0 - angle(coords[gpa], coords[par], coords[name]),
                        distance(coords[par], coords[name]),
                        par, gpa, gga))

    lines.append("")
    if redefine:
        lines.append("REDEFINE_CHI %d  %-4s %-4s %-4s %-4s"
                     % (redefine[0], *redefine[1]))

    # One chi per rotatable bond in the added group. The bond the redefined
    # parent chi already turns is skipped; rotations that only move a methyl
    # or methylene are left out, the way the database's own patches do.
    chi_number = max(parent["chis"], default=0)
    covered = {(redefine[1][1], redefine[1][2])} if redefine else set()
    proton_chis = []
    for name in order:
        if element[name] == "H":
            continue
        par = previous[name]
        if (par, name) in covered:
            continue
        children = [n for n in neighbours[name] if n in added and n != par]
        heavy_child = [c for c in children if element[c] != "H"]
        polar_child = [c for c in children if types[c][0] == "Hpol"]
        if heavy_child:
            child, is_proton = heavy_child[0], False
        elif polar_child:
            child, is_proton = polar_child[0], True
        else:
            continue

        _p, gpa, _g = reference_atoms(name, previous, parent, attach_parent, added)
        chi_number += 1
        lines.append("ADD_CHI %d  %-4s %-4s %-4s %-4s"
                     % (chi_number, gpa, par, name, child))
        if is_proton:
            proton_chis.append(chi_number)

    for number in proton_chis:
        lines.append("ADD_PROTON_CHI %d SAMPLES 3 60 -60 180 EXTRA 0"
                     % number)

    lines += ["",
              "DELETE_PROPERTY CANONICAL_AA",
              "DELETE_PROPERTY CANONICAL_NUCLEIC",
              "",
              "END_CASE",
              ""]

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                f"{code.lower()}.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    n_heavy = sum(1 for n in order if element[n] != "H")
    print(f"Wrote {out_path}")
    print(f"  residue   : {code} from {os.path.basename(args.input)}")
    print(f"  parent    : {parent['name']} (attached at {attach_parent})")
    print(f"  added     : {len(order)} atoms ({n_heavy} heavy, "
          f"{len(order) - n_heavy} H)")
    print(f"  variant   : {variant}"
          + (f" (already taken for {parent['name']}: {sorted(taken)})"
             if taken else ""))
    if virtual_h:
        print(f"  virtual   : {virtual_h[0]} set to VIRT")
    if redefine:
        print(f"  chi       : redefined {redefine[0]}, "
              f"added {max(parent['chis'], default=0) + 1}-{chi_number}")
    else:
        print(f"  chi       : added {max(parent['chis'], default=0) + 1}-"
              f"{chi_number}")


if __name__ == "__main__":
    main()
