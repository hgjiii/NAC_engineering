import json


def load_config(path):
    """
    Description:
        Load and validate an enzyme definition config from JSON.

    Args:
        path: Path to the enzyme config JSON file.

    Returns:
        Dict with parsed config. symmetry_pairs is expanded from per-ligand
        equivalence edges into a {ligand: [(a, b), ...]} swap-pair dict over
        each connected component, or None if absent.
    """
    with open(path) as f:
        cfg = json.load(f)

    _validate(cfg)
    cfg["symmetry_pairs"] = _expand_symmetry(cfg.get("symmetry_pairs"))
    return cfg


def get_pair_label(pair):
    """
    Description:
        Build the "labelA-labelB" joint label from an interaction pair dict,
        matching the key format produced by compute_pair_distances.

    Args:
        pair: Interaction pair dict with exactly two {label: spec} entries.

    Returns:
        Joint label string "labelA-labelB".
    """
    labels = list(pair.keys())
    return f"{labels[0]}-{labels[1]}"


def _expand_symmetry(symmetry):
    """
    Description:
        Expand per-ligand symmetry edges into full transitive closure. Edges
        are undirected; all atoms in one connected component become mutually
        interchangeable.

    Args:
        symmetry: Dict {ligand: [[atom, atom], ...]} of equivalence edges,
            or None.

    Returns:
        Dict {ligand: [(a, b), ...]} of ordered pairs over each component,
        or None if input is None.
    """
    if not symmetry:
        return None
    out = {}
    for ligand, edges in symmetry.items():
        pairs = []
        for comp in _connected_components(edges):
            for a in comp:
                for b in comp:
                    if a != b:
                        pairs.append((a, b))
        out[ligand] = pairs
    return out


def _connected_components(edges):
    """
    Description:
        Group atoms into connected components from undirected equivalence edges.

    Args:
        edges: List of [atom, atom] pairs.

    Returns:
        List of sets, each a connected component of mutually equivalent atoms.
    """
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    seen = set()
    components = []
    for node in adj:
        if node in seen:
            continue
        stack = [node]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            seen.add(cur)
            stack.extend(adj[cur] - comp)
        components.append(comp)
    return components


def _validate(cfg):
    """Raise ValueError if the config is structurally inconsistent."""
    required = ["enzyme_name", "catalytic", "interaction_pairs", "contact_cutoffs"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required key: '{key}'")

    for i, pair in enumerate(cfg["interaction_pairs"]):
        if len(pair) != 2:
            raise ValueError(f"interaction_pairs[{i}] must have exactly 2 entries, "
                             f"got {len(pair)}")

    pair_labels = {get_pair_label(p) for p in cfg["interaction_pairs"]}
    for label in cfg["contact_cutoffs"]:
        if label not in pair_labels:
            raise ValueError(f"contact_cutoffs key '{label}' does not match any "
                             f"interaction pair label. Available: {sorted(pair_labels)}")

    cat = cfg["catalytic"]
    for key in ["cat_keys", "chain_id"]:
        if key not in cat:
            raise ValueError(f"catalytic block missing '{key}'")