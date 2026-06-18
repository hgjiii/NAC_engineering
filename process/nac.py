import os

import numpy as np


def extract_entry_id(filename):
    """
    Description:
        Derive an entry id from a PLACER PDB filename by stripping the
        extension, the dotted PLACER suffix, and a trailing "_model".

    Args:
        filename: PLACER PDB filename (e.g. "carA_A0A0F5NA60.relax_model.pdb").

    Returns:
        Entry id string (e.g. "carA_A0A0F5NA60").
    """
    stem = os.path.splitext(filename)[0]   # drop ".pdb"
    stem = stem.split(".")[0]              # drop ".relax_model" if present
    if stem.endswith("_model"):
        stem = stem[:-len("_model")]       # drop trailing "_model"
    return stem


def fill_interaction_pairs(template, resname_to_resid):
    """
    Description:
        Fill null resids in an interaction-pair template using catalytic
        residue ids, matched by each spec's "residue_type".

    Args:
        template: List of interaction pair dicts; residue specs have
            "resid": null and a "residue_type" field.
        resname_to_resid: Dict mapping residue type (e.g. "HIS") to 1-based resid.

    Returns:
        New list of pair dicts with resids filled. Raises KeyError if a needed
        residue_type is missing from resname_to_resid.
    """
    pairs = []
    for pair_tmpl in template:
        new_pair = {}
        for label, spec in pair_tmpl.items():
            s = dict(spec)
            if "resid" in s and s["resid"] is None:
                s["resid"] = resname_to_resid[s["residue_type"]]
            s.pop("residue_type", None)
            new_pair[label] = s
        pairs.append(new_pair)
    return pairs


def compute_nac(pair_distances, contact_cutoffs, conf_mask):
    """
    Description:
        Compute holo NAC statistics from precomputed per-model effective
        cutoffs: per-contact satisfaction fractions, the joint NAC fraction,
        and NAC model indices over confident models.

    Args:
        pair_distances: Dict label -> ndarray of per-model distances.
        contact_cutoffs: Dict label -> ndarray of per-model contact cutoffs.
        conf_mask: Boolean ndarray marking confident models.

    Returns:
        Dict with fracs (per-contact), frac_nac (scalar), nac_holo_idxs (list),
        n_confident (int).
    """
    n_conf = int(conf_mask.sum())

    if n_conf == 0:
        return {
            "fracs": {label: 0.0 for label in contact_cutoffs},
            "frac_nac": 0.0,
            "nac_holo_idxs": [],
            "n_confident_holo": 0,
        }

    fracs = {label: ((pair_distances[label] <= contact_cutoffs[label]) & conf_mask).sum() / n_conf
             for label in contact_cutoffs}
    nac_satisfied = np.logical_and.reduce(
        [pair_distances[label] <= contact_cutoffs[label] for label in contact_cutoffs]
    ) & conf_mask
    frac_nac = nac_satisfied.sum() / n_conf
    nac_holo_idxs = (np.where(nac_satisfied)[0]).tolist()

    return {
        "fracs": fracs,
        "frac_nac": float(frac_nac),
        "nac_holo_idxs": nac_holo_idxs,
        "n_confident_holo": n_conf,
    }