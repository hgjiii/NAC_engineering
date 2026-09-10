"""
Description:
    Shared parsing and chemistry helpers for the pipeline scripts that turn a
    residue or ligand out of a structure file into a topology another program
    can read (Rosetta params, Rosetta patches, PLACER library entries).

    The common problem those scripts face is that PDB and model mmCIF files
    carry atom names, elements and coordinates but no bonds. Names have to be
    preserved exactly, because every downstream tool matches its topology to
    the structure by atom name, and bonds have to come from somewhere -- here,
    from Open Babel's perception, which is fast and accurate on this job.
"""

import os
import math
import shlex
import tempfile

OB_ORDER = {1: "SING", 2: "DOUB", 3: "TRIP"}
BOND_ORDER = {"SING": "1", "DOUB": "2", "TRIP": "3", "QUAD": "4"}


def require_openbabel(reason):
    """
    Description:
        Import Open Babel, or fail with a message naming what needed it.

    Args:
        reason: What the caller needs Open Babel for.

    Returns:
        The openbabel.openbabel module.

    Raises:
        SystemExit if Open Babel is not importable.
    """
    try:
        from openbabel import openbabel
        return openbabel
    except ImportError:
        raise SystemExit(
            f"Open Babel is required to {reason}, but it is not importable "
            "in this Python environment.")


def normalise_element(symbol):
    """
    Description:
        Normalise an element symbol to conventional capitalisation, so that
        two-letter elements read out of a PDB in upper case ("CL") match what
        chemistry toolkits expect ("Cl").

    Args:
        symbol: Raw symbol from a PDB or cif file.

    Returns:
        Capitalised symbol.
    """
    symbol = symbol.strip()
    return symbol.capitalize() if len(symbol) == 2 else symbol.upper()


def read_cif_category(lines, tag):
    """
    Description:
        Read one mmCIF category into a list of row dicts. Handles both the
        loop_ form and the key-value form mmCIF uses for a single row.

    Args:
        lines: The cif file as a list of lines.
        tag: Category name including the leading underscore, e.g.
            "_chem_comp_atom".

    Returns:
        List of dicts mapping the item name, without the category prefix, to
        its raw string value. Empty if the category is absent.
    """
    prefix = tag + "."
    keys, rows, single = [], [], {}
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line.startswith(prefix):
            i += 1
            continue

        fields = shlex.split(line)
        if len(fields) >= 2:
            single[fields[0][len(prefix):]] = fields[1]
            i += 1
            continue

        # Bare item name: this is a loop_ header block.
        while i < len(lines) and lines[i].startswith(prefix):
            keys.append(lines[i].strip()[len(prefix):])
            i += 1
        while i < len(lines) and not lines[i].startswith(("#", "loop_", "_")):
            if lines[i].strip():
                values = shlex.split(lines[i])
                if len(values) == len(keys):
                    rows.append(dict(zip(keys, values)))
            i += 1

    if rows:
        return rows
    return [single] if single else []


def read_residue_from_pdb(path, code, chain, resnum):
    """
    Description:
        Collect one residue's atoms from a PDB file, matching on the residue
        name in columns 18-20 and reading both ATOM and HETATM records.

    Args:
        path: PDB file path.
        code: Residue name, e.g. "UQ3".
        chain: Chain id to restrict to, or None.
        resnum: Residue number to restrict to, or None.

    Returns:
        Tuple of (atom list, copies), where each atom is
        (name, element, x, y, z) and copies lists the (chain, resnum) the
        code was seen in.
    """
    atoms, copies = [], []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[17:20].strip().upper() != code:
            continue

        this_chain = line[21].strip()
        this_num = line[22:26].strip()
        if (this_chain, this_num) not in copies:
            copies.append((this_chain, this_num))
        if chain is not None and this_chain != chain:
            continue
        if resnum is not None and this_num != str(resnum):
            continue

        elem = line[76:78].strip() or line[12:16].strip()[0]
        atoms.append((line[12:16].strip(), normalise_element(elem),
                      float(line[30:38]), float(line[38:46]), float(line[46:54])))

    return atoms, copies


def read_residue_from_cif(path, code, chain, resnum):
    """
    Description:
        Collect one residue's atoms from an mmCIF file's _atom_site loop.

    Args:
        path: mmCIF file path.
        code: Component id, matched against auth/label_comp_id.
        chain: Chain id (auth_asym_id) to restrict to, or None.
        resnum: Residue number (auth_seq_id) to restrict to, or None.

    Returns:
        Same shape as read_residue_from_pdb.
    """
    rows = read_cif_category(open(path).read().splitlines(), "_atom_site")
    atoms, copies = [], []

    for row in rows:
        comp = row.get("auth_comp_id") or row.get("label_comp_id", "")
        if comp.strip().upper() != code:
            continue

        this_chain = (row.get("auth_asym_id") or row.get("label_asym_id", "")).strip()
        this_num = (row.get("auth_seq_id") or row.get("label_seq_id", "")).strip()
        if (this_chain, this_num) not in copies:
            copies.append((this_chain, this_num))
        if chain is not None and this_chain != chain:
            continue
        if resnum is not None and this_num != str(resnum):
            continue

        name = (row.get("auth_atom_id") or row.get("label_atom_id", "")).strip()
        atoms.append((name, normalise_element(row.get("type_symbol", "")),
                      float(row["Cartn_x"]), float(row["Cartn_y"]),
                      float(row["Cartn_z"])))

    return atoms, copies


def read_residue(path, code, chain=None, resnum=None):
    """
    Description:
        Read one residue out of a PDB or mmCIF file, chosen by file extension.
        Refuses ambiguous input rather than silently picking a copy.

    Args:
        path: Structure file path.
        code: Residue name to extract.
        chain: Chain id to restrict to, or None.
        resnum: Residue number to restrict to, or None.

    Returns:
        List of (name, element, x, y, z).

    Raises:
        ValueError if the code is absent, occurs more than once without a
        chain/resnum selector, or has duplicate atom names.
    """
    reader = read_residue_from_cif if path.lower().endswith((".cif", ".mmcif")) \
        else read_residue_from_pdb
    atoms, copies = reader(path, code, chain, resnum)

    if not copies:
        raise ValueError(f"No residue named '{code}' in {path}.")
    if len(copies) > 1 and chain is None and resnum is None:
        listed = ", ".join(f"chain {c or '?'} residue {n}" for c, n in copies)
        raise ValueError(f"'{code}' occurs {len(copies)} times ({listed}). "
                         "Pick one by chain and/or residue number.")
    if not atoms:
        raise ValueError(f"No atoms of '{code}' matched the given "
                         "chain/residue-number selection.")

    names = [a[0] for a in atoms]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(f"Duplicate atom names in '{code}': {duplicated}. "
                         "Alternate locations are not supported.")

    return atoms


def pdb_atom_field(name, element):
    """
    Description:
        Lay an atom name out in PDB columns 13-16, following the convention
        that one-letter elements start in column 14.

    Args:
        name: Atom name, e.g. "C01", "HO2'".
        element: Element symbol.

    Returns:
        A four-character string.
    """
    if len(name) >= 4 or len(element) == 2:
        return "%-4s" % name[:4]
    return " %-3s" % name


def write_fragment_pdb(atoms, resname, path):
    """
    Description:
        Write a list of atoms as a one-residue PDB. Open Babel reads this
        back with the atom names intact and perceives the bonds from the
        geometry, which is how the pipeline scripts get connectivity for a
        residue that no dictionary defines.

    Args:
        atoms: List of (name, element, x, y, z).
        resname: Residue name for columns 18-20.
        path: Where to write.

    Returns:
        None.
    """
    with open(path, "w") as f:
        for n, (name, elem, x, y, z) in enumerate(atoms, start=1):
            f.write("HETATM%5d %s %-3s X   1    %8.3f%8.3f%8.3f  1.00  0.00"
                    "          %2s\n"
                    % (n, pdb_atom_field(name, elem), resname[:3], x, y, z,
                       elem.upper()))
        f.write("END\n")


def build_obmol(atoms, resname):
    """
    Description:
        Read a residue into Open Babel, perceiving its bonds from geometry.
        The atoms go out through a scratch PDB and come back in the same
        order, so names can be mapped by index afterwards.

    Args:
        atoms: List of (name, element, x, y, z).
        resname: Residue name stamped on the scratch PDB.

    Returns:
        Tuple of (openbabel module, OBMol) whose atom order matches atoms.

    Raises:
        SystemExit if Open Babel cannot read the fragment back, or returns a
        different number of atoms than were written.
    """
    ob = require_openbabel("perceive bonds")

    handle, tmp = tempfile.mkstemp(suffix=".pdb")
    os.close(handle)
    try:
        write_fragment_pdb(atoms, resname, tmp)
        conv = ob.OBConversion()
        conv.SetInFormat("pdb")
        mol = ob.OBMol()
        if not conv.ReadFile(mol, tmp):
            raise SystemExit("Open Babel could not read the extracted residue.")
    finally:
        os.unlink(tmp)

    if mol.NumAtoms() != len(atoms):
        raise SystemExit(f"Open Babel returned {mol.NumAtoms()} atoms for "
                         f"{len(atoms)} written; cannot map names back.")
    return ob, mol


def apply_bond_rows(mol, names, bond_rows):
    """
    Description:
        Force a molecule's bonds to match an exact bond table, used when a
        dictionary entry or a SMILES is available and should override what
        was perceived from geometry.

    Args:
        mol: OBMol whose first len(names) atoms correspond to names.
        names: Atom names in OBMol index order.
        bond_rows: Bond row dicts with atom_id_1, atom_id_2, value_order and
            pdbx_aromatic_flag.

    Returns:
        None.
    """
    order = {"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}
    index = {name: i for i, name in enumerate(names, start=1)}
    wanted = {}
    for row in bond_rows:
        pair = (index[row["atom_id_1"]], index[row["atom_id_2"]])
        wanted[frozenset(pair)] = (pair, order.get(row["value_order"], 1),
                                   row.get("pdbx_aromatic_flag") == "Y")

    for n in range(mol.NumBonds() - 1, -1, -1):
        bond = mol.GetBond(n)
        if frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) not in wanted:
            mol.DeleteBond(bond)

    for pair, value, aromatic in wanted.values():
        bond = mol.GetBond(*pair)
        if bond is None:
            mol.AddBond(pair[0], pair[1], value)
            bond = mol.GetBond(*pair)
        bond.SetBondOrder(value)
        bond.SetAromatic(aromatic)


def name_new_hydrogens(names, total):
    """
    Description:
        Name hydrogens Open Babel added, which come back unnamed. Numbering
        starts at H1 and skips anything the structure already uses.

    Args:
        names: Names of the original atoms, in order.
        total: Atom count after hydrogens were added.

    Returns:
        The full name list, original names first.
    """
    out = list(names)
    used = set(names)
    counter = 0
    for _ in range(len(names), total):
        counter += 1
        while f"H{counter}" in used:
            counter += 1
        used.add(f"H{counter}")
        out.append(f"H{counter}")
    return out


def harvest_obmol(ob, mol, names):
    """
    Description:
        Read atoms and bonds back out of an OBMol into the plain tuples and
        row dicts the rest of the pipeline uses.

    Args:
        ob: The openbabel module.
        mol: OBMol to read.
        names: Names of the original atoms, in OBMol index order.

    Returns:
        Tuple of (atom list, bond row list), where atoms are
        (name, element, x, y, z) and bonds carry atom_id_1, atom_id_2,
        value_order and pdbx_aromatic_flag.
    """
    out_names = name_new_hydrogens(names, mol.NumAtoms())

    atoms = []
    for i in range(1, mol.NumAtoms() + 1):
        atom = mol.GetAtom(i)
        atoms.append((out_names[i - 1],
                      normalise_element(ob.GetSymbol(atom.GetAtomicNum())),
                      atom.GetX(), atom.GetY(), atom.GetZ()))

    bond_rows = []
    for n in range(mol.NumBonds()):
        bond = mol.GetBond(n)
        bond_rows.append({
            "atom_id_1": out_names[bond.GetBeginAtomIdx() - 1],
            "atom_id_2": out_names[bond.GetEndAtomIdx() - 1],
            "value_order": OB_ORDER.get(bond.GetBondOrder(), "SING"),
            "pdbx_aromatic_flag": "Y" if bond.IsAromatic() else "N"})

    return atoms, bond_rows


def find_carbonyl_carbons(bonds, element):
    """
    Description:
        Collect carbons carrying a C=O double bond, which is what tells an
        amide apart from an ester or a carboxylate downstream.

    Args:
        bonds: List of bond row dicts.
        element: Dict mapping atom name to element symbol.

    Returns:
        Set of atom names.
    """
    carbonyl = set()
    for bond in bonds:
        a, b = bond["atom_id_1"], bond["atom_id_2"]
        if bond["value_order"] == "DOUB" and {element[a], element[b]} == {"C", "O"}:
            carbonyl.add(a if element[a] == "C" else b)
    return carbonyl


def mol2_bond_type(bond, element, carbonyl):
    """
    Description:
        Map a bond onto a Tripos bond type. Aromatic bonds must be flagged
        explicitly, since molfile_to_params.py reads 'ar' and 'am' as
        Bond.AROMATIC and warns that a Kekule structure "won't cut it".
        Amides get 'am' so their carbon stays typed CNH2 rather than COO,
        and so the packer keeps them planar.

    Args:
        bond: A bond row dict.
        element: Dict mapping atom name to element symbol.
        carbonyl: Set of carbons bearing a C=O, from find_carbonyl_carbons.

    Returns:
        One of "ar", "am", "1", "2", "3", "4".
    """
    a, b = bond["atom_id_1"], bond["atom_id_2"]
    order = bond["value_order"]

    if bond.get("pdbx_aromatic_flag") == "Y":
        return "ar"
    if order == "SING" and {element[a], element[b]} == {"C", "N"}:
        if (a if element[a] == "C" else b) in carbonyl:
            return "am"
    return BOND_ORDER.get(order, "1")


def render_mol2(resname, atoms, bonds):
    """
    Description:
        Render atom and bond records as Tripos mol2 text. Partial charges are
        written as zero: molfile_to_params.py assigns Rosetta's own typed
        charges when --recharge is given, and those are what the score
        function was parameterised with.

    Args:
        resname: Substructure name stamped on every atom.
        atoms: List of (name, element, is_aromatic, x, y, z) in output order.
        bonds: List of (index1, index2, bond_type) with 1-based indices into
            atoms and a Tripos bond type string.

    Returns:
        The mol2 file contents as one string.
    """
    out = ["@<TRIPOS>MOLECULE", resname,
           " %d %d 1 0 0" % (len(atoms), len(bonds)),
           "SMALL", "NO_CHARGES", "", "", "@<TRIPOS>ATOM"]

    for n, (name, elem, aromatic, x, y, z) in enumerate(atoms, start=1):
        if elem == "H":
            sybyl = "H"
        elif aromatic:
            sybyl = elem + ".ar"
        else:
            sybyl = elem + ".3"
        out.append("%7d %-8s %9.4f %9.4f %9.4f %-6s %3d %-8s %9.4f"
                   % (n, name, x, y, z, sybyl, 1, resname, 0.0))

    out.append("@<TRIPOS>BOND")
    for n, (i, j, kind) in enumerate(bonds, start=1):
        out.append("%6d %5d %5d %-4s" % (n, i, j, kind))

    return "\n".join(out) + "\n"


def distance(a, b):
    """
    Description:
        Distance between two xyz tuples.

    Args:
        a: First point.
        b: Second point.

    Returns:
        Distance in the input units.
    """
    return math.dist(a, b)


def angle(a, b, c):
    """
    Description:
        Angle a-b-c.

    Args:
        a: First point.
        b: Vertex.
        c: Third point.

    Returns:
        Angle in degrees.
    """
    u = [a[i] - b[i] for i in range(3)]
    v = [c[i] - b[i] for i in range(3)]
    nu = math.sqrt(sum(t * t for t in u))
    nv = math.sqrt(sum(t * t for t in v))
    cos = sum(u[i] * v[i] for i in range(3)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def dihedral(a, b, c, d):
    """
    Description:
        Dihedral a-b-c-d.

    Args:
        a: First point.
        b: Second point.
        c: Third point.
        d: Fourth point.

    Returns:
        Dihedral in degrees, in (-180, 180].
    """
    def sub(p, q):
        return [p[i] - q[i] for i in range(3)]

    def cross(p, q):
        return [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2],
                p[0] * q[1] - p[1] * q[0]]

    b1, b2, b3 = sub(b, a), sub(c, b), sub(d, c)
    n1, n2 = cross(b1, b2), cross(b2, b3)
    norm = math.sqrt(sum(t * t for t in b2))
    m = cross(n1, [t / norm for t in b2])
    return math.degrees(math.atan2(sum(m[i] * n2[i] for i in range(3)),
                                   sum(n1[i] * n2[i] for i in range(3))))
