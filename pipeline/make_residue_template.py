"""Build a PLACER residue-library entry for a modified residue from a structure.

PLACER keys its chemistry off a residue library: the entry supplies the
topology and the structure supplies coordinates for whatever atom names
match. A residue the library does not know cannot be parsed as part of the
protein at all -- pdbparser looks it up unconditionally and dies on None.

This script writes that entry straight from a structure file. Nothing but
the structure is needed: bonds are perceived with Open Babel, and the only
thing read off the residue itself is whether it carries a backbone, which
decides how it is capped.

A library entry describes the *free* residue, so a mid-chain residue is
capped back up before it is written -- OXT on the carboxyl carbon, a second
amine hydrogen on N -- and those capping atoms are flagged as leaving.
PLACER drops them again when it forms the peptide bonds, and works out each
atom's leaving group from the flags and the bond graph.

The result is loaded with PLACERinput.add_custom_residues(), which merges it
into the library for that run only; the shipped ligands.json.gz is untouched.

Usage:
    python pipeline/make_residue_template.py -i model.pdb -r XHB -o xhb.json
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process.parsing import (read_residue, build_obmol, harvest_obmol,
                             require_openbabel, distance)

BACKBONE = ("N", "CA", "C", "O")

# Hydrogen names the PDB fixes by convention; everything else is derived from
# the heavy atom it hangs off.
CONVENTIONAL_H = {"N": ["H", "H2", "H3"],
                  "CA": ["HA", "HA2", "HA3"],
                  "OXT": ["HXT"]}


def is_amino_acid(names):
    """Decide whether a residue carries a protein backbone.

    Args:
        names: Atom names present in the residue.

    Return:
        True if all of N, CA, C and O are present.
    """
    return all(atom in names for atom in BACKBONE)


def build_oxt(atoms):
    """Place the carboxyl OXT a mid-chain residue is missing.

    The carboxyl carbon is trigonal, so with CA and O known the third
    substituent sits opposite their bisector.

    Args:
        atoms: List of (name, element, x, y, z).

    Return:
        The OXT atom tuple.

    Raise:
        SystemExit if the backbone atoms needed to place it are absent.
    """
    coords = {a[0]: (a[2], a[3], a[4]) for a in atoms}
    for atom in ("C", "CA", "O"):
        if atom not in coords:
            raise SystemExit(f"Cannot cap the residue: backbone atom {atom} "
                             "is missing.")

    c, ca, o = coords["C"], coords["CA"], coords["O"]
    direction = [0.0, 0.0, 0.0]
    for other in (ca, o):
        vector = [other[i] - c[i] for i in range(3)]
        length = distance(other, c)
        for i in range(3):
            direction[i] -= vector[i] / length

    length = sum(v * v for v in direction) ** 0.5
    return ("OXT", "O") + tuple(c[i] + 1.25 * direction[i] / length
                                for i in range(3))


def hydrogen_parents(bonds, heavy_names):
    """Map each added hydrogen to the heavy atom it hangs off.

    Args:
        bonds: Bond row list, as harvested.
        heavy_names: Names that came from the structure, plus any cap.

    Return:
        Dict mapping heavy atom name to its list of hydrogen names.
    """
    attached = {}
    for bond in bonds:
        a, b = bond["atom_id_1"], bond["atom_id_2"]
        for light, heavy in ((a, b), (b, a)):
            if light not in heavy_names and heavy in heavy_names:
                attached.setdefault(heavy, []).append(light)
    return {heavy: sorted(hydrogens) for heavy, hydrogens in attached.items()}


def name_hydrogens(attached, heavy_names):
    """Give the hydrogens Open Babel added names of their own.

    Backbone hydrogens take their conventional PDB names so the entry reads
    like the library's own amino acids. The rest get the usual "H" plus the
    heavy atom's suffix, falling back to the full heavy atom name wherever
    two atoms would otherwise claim the same suffix -- O3 and C3 both want
    H3, so they become HO3 and HC3.

    Args:
        attached: Output of hydrogen_parents.
        heavy_names: Names that came from the structure, plus any cap.

    Return:
        Dict mapping the harvested hydrogen name to its final name.
    """
    short = {}
    for heavy in attached:
        if heavy in CONVENTIONAL_H:
            short[heavy] = CONVENTIONAL_H[heavy][0]
        else:
            short[heavy] = "H" + (heavy[1:] if len(heavy) > 1
                                  and heavy[:1].isalpha() else heavy)

    claimed = {}
    for heavy, stem in short.items():
        claimed.setdefault(stem, []).append(heavy)

    renamed = {}
    used = set(heavy_names)
    for heavy, hydrogens in attached.items():
        if heavy in CONVENTIONAL_H and len(hydrogens) <= len(CONVENTIONAL_H[heavy]):
            names = CONVENTIONAL_H[heavy][:len(hydrogens)]
        else:
            stem = short[heavy] if len(claimed[short[heavy]]) == 1 else "H" + heavy
            names = [stem] if len(hydrogens) == 1 else \
                [f"{stem}{n}" for n in range(2, len(hydrogens) + 2)]

        for light, name in zip(hydrogens, names):
            while name in used:
                name += "X"
            used.add(name)
            renamed[light] = name

    return renamed


def fix_carboxyl(ob, mol, names):
    """Pin the carboxyl bond orders instead of letting geometry decide.

    The OXT built above sits at a distance Open Babel cannot tell a carbonyl
    from a hydroxyl by, so it is free to protonate the backbone O and leave
    OXT bare -- the opposite of the PDB convention, and it loses HXT. Setting
    the two orders explicitly before hydrogens are added settles it.

    Args:
        ob: The openbabel module.
        mol: OBMol whose atom order matches names.
        names: Atom names in OBMol index order.

    Return:
        None.
    """
    index = {name: i for i, name in enumerate(names, start=1)}
    if not {"C", "O", "OXT"} <= set(index):
        return

    for partner, order, hydrogens in (("O", 2, 0), ("OXT", 1, 1)):
        bond = mol.GetBond(index["C"], index[partner])
        if bond is not None:
            bond.SetBondOrder(order)
            bond.SetAromatic(False)
        # AddHydrogens goes by the implicit count, which was fixed when the
        # scratch PDB was read and does not follow a bond order changed after.
        mol.GetAtom(index[partner]).SetImplicitHCount(hydrogens)


def write_sdf(ob, mol, code):
    """Serialise the perceived molecule as an SDF string.

    Args:
        ob: The openbabel module.
        mol: OBMol to write.
        code: Residue name, used as the molecule title.

    Return:
        SDF text.
    """
    mol.SetTitle(code)
    conv = ob.OBConversion()
    conv.SetOutFormat("sdf")
    return conv.WriteString(mol)


def verify_entry(entry, code, structure_names):
    """Run the entry through PLACER's own parser and report what it made.

    Args:
        entry: The library entry dict for this residue.
        code: Residue name.
        structure_names: Atom names present in the structure, which must all
            appear in the entry or their coordinates would be dropped.

    Return:
        True if PLACER parsed it and every structure atom is covered.
    """
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PLACER"))
        from modules import cifutils
    except ImportError as exc:
        print(f"  [verify] skipped: PLACER modules not importable ({exc})")
        return True

    parser = cifutils.CIFParser(mols={code: entry})
    residue = parser.getRes(code)
    if residue is None:
        print("  [verify] FAILED: PLACER did not accept the entry")
        return False

    parsed = residue["res"]
    missing = sorted(set(structure_names) - set(parsed.atoms))
    leaving = sorted(a for a, atom in parsed.atoms.items() if atom.leaving)
    print(f"  [verify] parsed {len(parsed.atoms)} atoms, {len(parsed.bonds)} bonds")
    print(f"  [verify] leaving: {leaving or 'none'}")
    print(f"  [verify] structure atoms not covered: {missing or 'NONE'}")
    for atom in ("C", "N"):
        if atom in parsed.atoms:
            print(f"  [verify] {atom}.leaving_group: "
                  f"{sorted(parsed.atoms[atom].leaving_group) or 'none'}")
    return not missing


def main():
    parser = argparse.ArgumentParser(
        description="Write a PLACER residue-library entry for a modified "
                    "residue, deduced from a structure file.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Structure file holding the residue (.pdb or .cif).")
    parser.add_argument("-r", "--residue", type=str, required=True,
                        help="Residue name of the modified residue, e.g. XHB.")
    parser.add_argument("-o", "-O", "--output", type=str, default=None,
                        help="Output JSON path (default: <residue>.json beside "
                             "the structure).")
    parser.add_argument("-c", "--chain", type=str, default=None,
                        help="Chain id, when the residue occurs more than once.")
    parser.add_argument("-n", "--resnum", type=str, default=None,
                        help="Residue number, when it occurs more than once.")
    parser.add_argument("--no-cap", action="store_true", default=False,
                        help="Skip backbone capping and mark nothing as "
                             "leaving. Use for a residue that is genuinely a "
                             "free molecule rather than a chain member.")
    args = parser.parse_args()

    code = args.residue.strip().upper()
    structure_atoms = read_residue(args.input, code, args.chain, args.resnum)
    structure_names = [a[0] for a in structure_atoms]

    polymer = is_amino_acid(structure_names) and not args.no_cap
    atoms_in = list(structure_atoms)
    if polymer and "OXT" not in structure_names:
        atoms_in.append(build_oxt(structure_atoms))

    heavy_names = [a[0] for a in atoms_in]
    ob, mol = build_obmol(atoms_in, code)
    if polymer:
        fix_carboxyl(ob, mol, heavy_names)
    mol.AddHydrogens(False, False)
    atoms, bonds = harvest_obmol(ob, mol, heavy_names)

    attached = hydrogen_parents(bonds, set(heavy_names))
    renamed = name_hydrogens(attached, set(heavy_names))
    atom_id = [renamed.get(a[0], a[0]) for a in atoms]

    # The capping atoms leave again when PLACER forms the peptide bonds: OXT
    # with its own hydrogen, and every amine hydrogen past the first. They are
    # picked out through the bond graph rather than by name, so that a
    # modification's own hydrogens cannot be caught by the same label.
    leaving = set()
    if polymer:
        leaving.add("OXT")
        leaving.update(renamed[h] for h in attached.get("OXT", []))
        leaving.update(renamed[h] for h in attached.get("N", [])[1:])

    entry = {"sdf": write_sdf(ob, mol, code),
             "atom_id": atom_id,
             "leaving": [name in leaving for name in atom_id],
             "pdbx_align": [1] * len(atom_id)}

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                f"{code.lower()}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({code: entry}, f, indent=2)

    n_heavy = sum(1 for a in atoms if a[1] != "H")
    print(f"Wrote {out_path}")
    print(f"  residue   : {code} from {os.path.basename(args.input)}")
    print(f"  form      : {'polymer residue (capped)' if polymer else 'free molecule'}")
    print(f"  atoms     : {len(atom_id)} ({n_heavy} heavy, "
          f"{len(atom_id) - n_heavy} H)")
    print(f"  bonds     : {len(bonds)}")
    print(f"  leaving   : {sorted(leaving) or 'none'}")

    if not verify_entry(entry, code, structure_names):
        raise SystemExit("The entry does not cover every atom in the "
                         "structure; PLACER would drop the rest.")


if __name__ == "__main__":
    main()
