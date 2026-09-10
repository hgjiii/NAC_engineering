"""Extract one ligand from a PDB/mmCIF structure and write it as a mol2.

Built for ligands that are *not* in the PDB Chemical Component Dictionary:
give it a structure file and a residue name and it produces a params-ready
mol2, working the bonds out itself. Atom names and coordinates always come
from the model, so the result lines up with the structure Rosetta reads.

Bond orders are perceived with Open Babel, which is both fast and accurate
at this job -- on lactyl-CoA it recovers the CCD's own bond table exactly
(55/55 bonds, same connectivity and orders, bar the arbitrary choice of
which phosphate oxygen carries the P=O).

Exact chemistry can be supplied instead when it happens to be available:

    -t/--template   a CCD component cif; bonds matched by atom name
    -s/--smiles     a SMILES of the same heavy-atom skeleton (needs RDKit)

and the structure's own _chem_comp_bond loop is picked up automatically if
it has one.

Open Babel is needed for perception and for --add-h. Passing --template
without --add-h needs nothing but the standard library.

Usage:
    python pipeline/extract_ligand_mol2.py -i model.pdb -l LIG -o lig.mol2 --add-h
    python pipeline/extract_ligand_mol2.py -i model.pdb -l UQ3 -o uq3.mol2 \\
           -t UQ3_ccd.cif --add-h
    python pipeline/extract_ligand_mol2.py -i model.cif -l XHB -o xhb.mol2 \\
           -s "CC(O)CC(=O)SCC(N)C=O" --add-h
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process.parsing import (read_cif_category, read_residue, build_obmol,
                             apply_bond_rows, harvest_obmol, render_mol2,
                             find_carbonyl_carbons, mol2_bond_type)

RDKIT_ORDER = {"SINGLE": "SING", "DOUBLE": "DOUB", "TRIPLE": "TRIP",
               "QUADRUPLE": "QUAD", "AROMATIC": "SING"}












def bonds_from_template(path, code, names):
    """Read a CCD component's bond table, restricted to the atoms present.

    Args:
        path: CCD component cif path.
        code: Expected component id, for a sanity check.
        names: Atom names present in the extracted ligand.

    Return:
        List of bond row dicts.

    Raise:
        ValueError if the template does not cover every extracted atom.
    """
    lines = open(path).read().splitlines()
    comp = read_cif_category(lines, "_chem_comp")
    comp_id = comp[0].get("id", "").upper() if comp else ""
    if comp_id and comp_id != code:
        print(f"  [warn] template is component {comp_id}, not {code}")

    template_names = {a["atom_id"] for a in read_cif_category(lines, "_chem_comp_atom")}
    missing = sorted(set(names) - template_names)
    if missing:
        raise ValueError(
            f"Template {os.path.basename(path)} has no atom(s) {missing}. "
            "The structure and the template use different atom naming.")

    present = set(names)
    return [b for b in read_cif_category(lines, "_chem_comp_bond")
            if b["atom_id_1"] in present and b["atom_id_2"] in present]


def bonds_from_structure_cif(path, code, names):
    """Read a _chem_comp_bond loop out of the structure file itself.

    Args:
        path: Structure mmCIF path.
        code: Component id to filter on.
        names: Atom names present in the extracted ligand.

    Return:
        List of bond row dicts, empty if the file has no such loop.
    """
    rows = read_cif_category(open(path).read().splitlines(), "_chem_comp_bond")
    present = set(names)
    return [b for b in rows
            if b.get("comp_id", code).upper() == code
            and b["atom_id_1"] in present and b["atom_id_2"] in present]


def smiles_from_structure_cif(path, code):
    """Look up a component's SMILES in the structure's _chem_comp table.

    Args:
        path: Structure mmCIF path.
        code: Component id.

    Return:
        SMILES string, or None if absent or undefined.
    """
    for row in read_cif_category(open(path).read().splitlines(), "_chem_comp"):
        if row.get("id", "").strip().upper() != code:
            continue
        smiles = row.get("pdbx_smiles", "?").strip()
        return None if smiles in ("?", ".", "") else smiles
    return None


def bonds_from_smiles(atoms, smiles):
    """Assign bond orders by matching the ligand against a SMILES template.

    Args:
        atoms: List of (name, element, x, y, z), heavy atoms only.
        smiles: SMILES of the same heavy-atom skeleton.

    Return:
        List of bond row dicts.

    Raise:
        SystemExit if RDKit is unavailable.
        ValueError if the template does not match the extracted atoms.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDetermineBonds
        from rdkit.Geometry import Point3D
    except ImportError:
        raise SystemExit(
            "RDKit is required for --smiles, but it is not importable here. "
            "Drop --smiles to let Open Babel perceive the bonds.")

    mol = Chem.RWMol()
    for _name, elem, _x, _y, _z in atoms:
        atom = Chem.Atom(elem)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (_n, _e, x, y, z) in enumerate(atoms):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conf)
    rdDetermineBonds.DetermineConnectivity(mol)

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    heavy = sum(1 for a in atoms if a[1] != "H")
    if template.GetNumAtoms() != heavy:
        raise ValueError(
            f"SMILES has {template.GetNumAtoms()} heavy atoms but the ligand "
            f"has {heavy}; they must describe the same molecule.")

    try:
        assigned = AllChem.AssignBondOrdersFromTemplate(template, mol)
    except Exception as exc:
        raise ValueError(f"SMILES template did not match the ligand: {exc}")

    names = [a[0] for a in atoms]
    rows = []
    for bond in assigned.GetBonds():
        rows.append({"atom_id_1": names[bond.GetBeginAtomIdx()],
                     "atom_id_2": names[bond.GetEndAtomIdx()],
                     "value_order": RDKIT_ORDER.get(str(bond.GetBondType()), "SING"),
                     "pdbx_aromatic_flag": "Y" if bond.GetIsAromatic() else "N"})
    return rows


def resolve_bonds(atoms, args, code):
    """Pick the best bond source available.

    Args:
        atoms: Heavy atoms read from the structure.
        args: Parsed command line.
        code: Ligand residue name.

    Return:
        Tuple of (bond row list or None, description). None means "let Open
        Babel perceive them".
    """
    names = [a[0] for a in atoms]

    if args.template:
        return (bonds_from_template(args.template, code, names),
                f"CCD template {os.path.basename(args.template)} (exact)")

    is_cif = args.input.lower().endswith((".cif", ".mmcif"))
    if is_cif:
        rows = bonds_from_structure_cif(args.input, code, names)
        if rows:
            return rows, "structure's own _chem_comp_bond (exact)"

    smiles = args.smiles
    if smiles is None and is_cif:
        smiles = smiles_from_structure_cif(args.input, code)
        if smiles:
            print("  [info] using _chem_comp.pdbx_smiles from the structure")
    if smiles:
        return bonds_from_smiles(atoms, smiles), "SMILES template (exact orders)"

    return None, "Open Babel perception"


def main():
    parser = argparse.ArgumentParser(
        description="Extract one ligand from a PDB/mmCIF structure as a mol2, "
                    "keeping the model's atom names and coordinates and "
                    "perceiving the bonds. Works on ligands that are not in "
                    "the PDB Chemical Component Dictionary.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Structure file holding the ligand (.pdb or .cif).")
    parser.add_argument("-l", "--ligand", type=str, required=True,
                        help="Ligand residue name in that file, e.g. UQ3.")
    parser.add_argument("-o", "-O", "--output", type=str, default=None,
                        help="Output mol2 path (default: <ligand>.mol2 beside "
                             "the structure).")
    parser.add_argument("-t", "--template", type=str, default=None,
                        help="CCD component cif to take exact bonds from. "
                             "Optional; only useful when the ligand is in the CCD.")
    parser.add_argument("-s", "--smiles", type=str, default=None,
                        help="SMILES of the ligand, to pin the bond orders "
                             "instead of perceiving them. Requires RDKit.")
    parser.add_argument("-c", "--chain", type=str, default=None,
                        help="Chain id, when the ligand occurs more than once.")
    parser.add_argument("-r", "--resnum", type=str, default=None,
                        help="Residue number, when the ligand occurs more "
                             "than once.")
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="Residue name to stamp on the mol2 "
                             "(default: the -l value).")
    parser.add_argument("--add-h", action="store_true", default=False,
                        help="Add missing hydrogens with coordinates. Rosetta "
                             "types atoms by attached-H count, so params need "
                             "this.")
    args = parser.parse_args()

    code = args.ligand.strip().upper()
    resname = (args.name or code).strip().upper()

    atoms = read_residue(args.input, code, args.chain, args.resnum)
    names = [a[0] for a in atoms]
    bond_rows, source = resolve_bonds(atoms, args, code)

    if bond_rows is not None and not args.add_h:
        # Exact bonds and no hydrogens to place: no chemistry toolkit needed.
        final_atoms, final_bonds = atoms, bond_rows
    else:
        ob, mol = build_obmol(atoms, resname)
        if bond_rows is not None:
            apply_bond_rows(mol, names, bond_rows)
        if args.add_h:
            mol.AddHydrogens(False, False)
        final_atoms, final_bonds = harvest_obmol(ob, mol, names)

    element = {a[0]: a[1] for a in final_atoms}
    carbonyl = find_carbonyl_carbons(final_bonds, element)
    aromatic = set()
    for bond in final_bonds:
        if bond.get("pdbx_aromatic_flag") == "Y":
            aromatic.update((bond["atom_id_1"], bond["atom_id_2"]))

    index = {a[0]: n for n, a in enumerate(final_atoms, start=1)}
    records = [(n, e, n in aromatic, x, y, z) for n, e, x, y, z in final_atoms]
    pairs = [(index[b["atom_id_1"]], index[b["atom_id_2"]],
              mol2_bond_type(b, element, carbonyl)) for b in final_bonds]

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                f"{code.lower()}.mol2")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(render_mol2(resname, records, pairs))

    counts = {}
    for _i, _j, kind in pairs:
        counts[kind] = counts.get(kind, 0) + 1
    n_heavy = sum(1 for a in final_atoms if a[1] != "H")

    print(f"Wrote {out_path}")
    print(f"  ligand    : {code} from {os.path.basename(args.input)}"
          f" -> residue name {resname}")
    print(f"  atoms     : {len(final_atoms)} ({n_heavy} heavy, "
          f"{len(final_atoms) - n_heavy} H)")
    print(f"  bonds     : {len(pairs)} {counts}")
    print(f"  bond src  : {source}")

    if len(final_atoms) == n_heavy:
        print("\n  [warn] No hydrogens. Rosetta types atoms by attached-H "
              "count, so params built\n         from this will read every -OH "
              "as a charged -O(-). Rerun with --add-h.")


if __name__ == "__main__":
    main()
