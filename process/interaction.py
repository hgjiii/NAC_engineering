import warnings
import numpy as np

from Bio import BiopythonWarning
warnings.simplefilter("ignore", BiopythonWarning)

from process.structure import get_ligand_heavy_coords, get_coord

def build_catalytic_key_res(model, cat_entry, cat_keys, chain_id="A",
                            interaction_pairs=None):
    """
    Description:
        Build a key-residue list for the catalytic residues, looking up
        resnames on the model. Resids come from the per-entry catalytic
        mapping; when no mapping is given (cat_entry is None), they are read
        from the resids written explicitly in the interaction-pair template.

    Args:
        model: Bio.PDB Model used to read resnames.
        cat_entry: Per-entry catalytic dict, e.g. {"cat_His": {"resid_1based": 294}},
            or None to take resids from interaction_pairs instead.
        cat_keys: Catalytic keys to include (e.g. ["cat_His", "cat_Thr"]).
        chain_id: Chain identifier where the residues live.
        interaction_pairs: Interaction pair template, used only when cat_entry
            is None.

    Returns:
        List of dicts with chain/resid/resname, deduplicated by resid.
        Residues missing from the model are skipped.
    """
    if cat_entry is not None:
        resids = [cat_entry[k]["resid_1based"] for k in cat_keys if k in cat_entry]
    else:
        resids = [spec["resid"]
                  for pair in (interaction_pairs or []) for spec in pair.values()
                  if "ligand" not in spec and spec.get("resid") is not None
                  and spec.get("chain") == chain_id]

    out, seen = [], set()
    for resid in resids:
        if resid in seen:
            continue
        seen.add(resid)
        try:
            res = model[chain_id][(" ", resid, " ")]
        except KeyError:
            continue
        out.append({"chain": chain_id, "resid": resid,
                    "resname": res.get_resname().strip()})
    return out

def compute_pair_distances(models, pairs, sym_pairs=None):
    """
    Description:
        Compute atom-atom distances for a list of pair specs across all models.
        Each pair is a dict {label: spec} with exactly two entries. When either
        or both atoms are symmetric, the minimum distance over all equivalent
        atom combinations is returned, independent of spec order.

    Args:
        models: List of Bio.PDB Models.
        pairs: List of dicts, each with two {label: spec} entries.
        sym_pairs: Optional symmetry swap pairs (flat list or per-ligand dict).

    Returns:
        Dict mapping "labelA-labelB" to ndarray of length n_models in Angstrom.
    """
    out = {}
    for pair in pairs:
        (label_a, spec_a), (label_b, spec_b) = list(pair.items())
        joint_label = f"{label_a}-{label_b}"
        dists = []
        for m in models:
            xas = _get_atom_xyz(m, spec_a, sym_pairs=sym_pairs)
            xbs = _get_atom_xyz(m, spec_b, sym_pairs=sym_pairs)
            d = min(np.linalg.norm(xa - xb) for xa in xas for xb in xbs)
            dists.append(float(d))
        out[joint_label] = np.array(dists)
    return out

def _get_atom_xyz(model, spec, sym_pairs=None):
    """
    Description:
        Return all symmetry-equivalent candidate coordinates for one atom spec.
        Residue specs and non-symmetric ligand atoms yield a single coord.

    Args:
        model: Bio.PDB Model.
        spec: Atom spec dict with "chain", "atom", and either "ligand" or "resid".
        sym_pairs: Flat list of (a, b) pairs or per-ligand dict; None disables.

    Returns:
        List of numpy 3-vectors (length >= 1).
    """
    if "ligand" not in spec:
        try:
            res = model[spec["chain"]][(" ", spec["resid"], " ")]
        except KeyError:
            raise KeyError(f"Residue {spec['chain']}{spec['resid']} not found in model.")
        resname = res.get_resname().strip()
        if spec.get("residue_type") not in (None, resname):
            raise ValueError(f"Residue {spec['chain']}{spec['resid']} is {resname}, "
                             f"config expects {spec['residue_type']}.")
        atom_map = {a.get_name(): a for a in res.get_atoms()}
        if spec["atom"] not in atom_map:
            raise KeyError(f"Atom {spec['atom']} not found in residue "
                           f"{spec['chain']}{spec['resid']}.")
        return [get_coord(atom_map[spec["atom"]])]

    ligs = get_ligand_heavy_coords(model)
    lig = next((l for l in ligs
                if l["resname"] == spec["ligand"] and l["chain"] == spec["chain"]),
               None)
    if lig is None:
        raise KeyError(f"Ligand {spec['ligand']} not found in chain {spec['chain']}.")

    name = spec["atom"]
    candidates = {name}
    if sym_pairs is not None:
        pairs = sym_pairs.get(spec["ligand"]) if isinstance(sym_pairs, dict) else sym_pairs
        if pairs:
            candidates |= {b for a, b in pairs if a == name}
    candidates &= set(lig["atoms"])
    if not candidates:
        raise KeyError(f"Atom {name} not found in ligand {spec['ligand']}.")
    return [lig["coords"][lig["atoms"].index(n)] for n in candidates]

# from process.symmetry import resolve_symmetric_atom

# def compute_pair_distances(models, pairs, sym_pairs=None):
#     """
#     Description:
#         Compute per-model atom-atom distances for each two-atom pair spec,
#         resolving the second atom's symmetry against the first.

#     Args:
#         models: List of Bio.PDB Models.
#         pairs: List of dicts, each with two {label: spec} entries.
#         sym_pairs: Optional symmetry swap pairs (flat list or per-ligand dict).

#     Returns:
#         Dict mapping "labelA-labelB" to ndarray of length n_models in Angstrom.
#     """
#     out = {}
#     for pair in pairs:
#         (label_a, spec_a), (label_b, spec_b) = list(pair.items())
#         joint_label = f"{label_a}-{label_b}"
#         dists = []
#         for m in models:
#             xa = _get_atom_xyz(m, spec_a, partner_xyz=None, sym_pairs=sym_pairs)
#             xb = _get_atom_xyz(m, spec_b, partner_xyz=xa, sym_pairs=sym_pairs)
#             dists.append(float(np.linalg.norm(xa - xb)))
#         out[joint_label] = np.array(dists)
#     return out


# def _get_atom_xyz(model, spec, partner_xyz=None, sym_pairs=None):
#     """
#     Description:
#         Resolve one atom spec to a single xyz. Ligand atoms may be swapped to a
#         symmetry-equivalent atom closer to partner_xyz; residue atoms are fixed.

#     Args:
#         model: Bio.PDB Model.
#         spec: Atom spec dict with "chain", "atom", and either "ligand" or "resid".
#         partner_xyz: Reference point for symmetry resolution (ligand only).
#         sym_pairs: Flat list of (a, b) pairs or per-ligand dict; None disables.

#     Returns:
#         Numpy 3-vector.
#     """
#     if "ligand" not in spec:
#         res = model[spec["chain"]][(" ", spec["resid"], " ")]
#         atom_map = {a.get_name(): a for a in res.get_atoms()}
#         if spec["atom"] not in atom_map:
#             raise KeyError(f"Atom {spec['atom']} not found in residue "
#                            f"{spec['chain']}{spec['resid']}.")
#         return get_coord(atom_map[spec["atom"]])

#     ligs = get_ligand_heavy_coords(model)
#     lig = next((l for l in ligs
#                 if l["resname"] == spec["ligand"] and l["chain"] == spec["chain"]),
#                None)
#     if lig is None:
#         raise KeyError(f"Ligand {spec['ligand']} not found in chain {spec['chain']}.")

#     name = spec["atom"]
#     if partner_xyz is not None and sym_pairs is not None:
#         pairs = sym_pairs.get(spec["ligand"]) if isinstance(sym_pairs, dict) else sym_pairs
#         if pairs:
#             name = resolve_symmetric_atom(lig, name, partner_xyz, sym_pairs=pairs)
#     if name not in lig["atoms"]:
#         raise KeyError(f"Atom {name} not found in ligand {spec['ligand']}.")
#     return lig["coords"][lig["atoms"].index(name)]