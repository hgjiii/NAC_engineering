"""
Description:
    Restrict PLACER's score terms to part of the structure.

    PLACER reports one number per model for prmsd, plddt, plddt_pde, fape,
    lddt, rmsd and kabsch, each averaged over a fixed set of atoms. When a
    ligand has a large, floppy region that is beside the point -- the CoA arm
    of an acyl-CoA, say, when the question is about the acyl end -- that
    region dominates the average and buries the signal.

    This module drops chosen atoms from the averages without touching
    PLACER's own source. Two selections feed every score term:

        utils.get_prmsd / get_plddt / get_plddt_pde   take an index list
        Losses.get_losses                             reads topology['observed']

    and both are reachable by wrapping three functions. The atom index to
    atom name mapping comes from get_atom_graph, whose nodes are keyed by
    (chain, resnum, name3, atom_name) and carry their own index.

    The excluded atoms still go into the prediction unchanged; only the
    scoring ignores them, so the CSV keeps its usual columns and meaning.
"""

import torch


def parse_exclusions(tokens):
    """
    Description:
        Parse selector tokens naming atoms to leave out of the scores. The
        syntax matches run_placer.py's --center: a residue by name or by
        position, optionally narrowed to some of its atoms.

    Args:
        tokens: Raw CLI tokens, e.g. ["UQ3@N36,C37", "A296", "LEU"].

    Returns:
        List of (spec, atom names) pairs, an empty atom list meaning the
        whole residue.

    Raises:
        SystemExit on an empty residue part.
    """
    selectors = []
    for token in tokens or []:
        spec, _, atom_names = token.strip().upper().partition("@")
        if not spec:
            raise SystemExit(
                f"Cannot read --exclude_score '{token}'. Use a residue (UQ3 "
                "or X1), or a residue with atoms (UQ3@N36,C37).")
        selectors.append((spec, [a.strip() for a in atom_names.split(",")
                                 if a.strip()]))
    return selectors


def matches(node, selectors):
    """
    Description:
        Test one atom graph node against the exclusion selectors.

    Args:
        node: Atom key, (chain, resnum, name3, atom_name).
        selectors: Output of parse_exclusions.

    Returns:
        True if the atom should be left out of the scores.
    """
    chain, resnum, name3, atom_name = node
    position = f"{chain}{resnum}"

    for spec, atom_names in selectors:
        if spec != name3 and spec != position:
            continue
        if atom_names and atom_name not in atom_names:
            continue
        return True
    return False


class ScoreMask:
    """
    Description:
        Holds the excluded atom indices for the structure being scored, and
        the patches that make PLACER honour them. One instance is installed
        for a whole run; it refreshes itself each time PLACER builds a new
        atom graph, which happens once per model.
    """

    def __init__(self, selectors):
        """
        Args:
            selectors: Output of parse_exclusions.
        """
        self.selectors = selectors
        self.excluded = torch.zeros(0, dtype=torch.long)
        self.last_report = None

    def observe_graph(self, graph):
        """
        Description:
            Record which atom indices of a freshly built crop are excluded.

        Args:
            graph: The nx.Graph returned by get_atom_graph.

        Returns:
            None.
        """
        dropped = [data["index"] for node, data in graph.nodes(data=True)
                   if matches(node, self.selectors)]
        self.excluded = torch.tensor(sorted(dropped), dtype=torch.long)

        report = (len(dropped), len(graph))
        if report != self.last_report:
            print(f"# scoring on {report[1] - report[0]} of {report[1]} "
                  f"cropped atoms ({report[0]} excluded)")
            self.last_report = report

    def keep(self, sel):
        """
        Description:
            Drop the excluded atoms from a score-term selection.

        Args:
            sel: Index tensor PLACER was going to score over.

        Returns:
            The same tensor without the excluded indices.

        Raises:
            SystemExit if nothing is left to score.
        """
        if self.excluded.numel() == 0:
            return sel

        excluded = self.excluded.to(sel.device)
        kept = sel[~torch.isin(sel, excluded)]
        if kept.numel() == 0:
            raise SystemExit("--exclude_score left no atoms to score on.")
        return kept

    def keep_mask(self, mask):
        """
        Description:
            Clear the excluded atoms from a boolean per-atom mask, which is
            how the loss terms select what they average over.

        Args:
            mask: Boolean tensor of length equal to the crop.

        Returns:
            A copy with the excluded positions set False.
        """
        if self.excluded.numel() == 0:
            return mask

        out = mask.clone()
        out[self.excluded.to(mask.device)] = False
        return out


def install(placer_module, placer, selectors):
    """
    Description:
        Patch PLACER in place so its score terms skip the excluded atoms.
        Wrapping rather than editing keeps the vendored PLACER source clean
        and leaves the prediction itself untouched.

    Args:
        placer_module: The imported PLACER module.
        placer: An already built PLACER instance, used to reach the dataset
            class the atom graph is built on.
        selectors: Output of parse_exclusions, or an empty list to do nothing.

    Returns:
        The installed ScoreMask, or None when there is nothing to exclude.
    """
    if not selectors:
        return None

    mask = ScoreMask(selectors)
    utils = placer_module.utils
    dataset = type(placer.dataloader().dataset.dataset)

    original_graph = dataset.get_atom_graph

    def get_atom_graph(self, *args, **kwargs):
        graph = original_graph(self, *args, **kwargs)
        mask.observe_graph(graph)
        return graph

    dataset.get_atom_graph = get_atom_graph

    for name in ("get_prmsd", "get_plddt"):
        original = getattr(utils, name)

        def wrapped(values, sel, _original=original):
            return _original(values, mask.keep(sel))

        setattr(utils, name, wrapped)

    original_pde = utils.get_plddt_pde

    def get_plddt_pde(X, D, sel, *args, **kwargs):
        return original_pde(X, D, mask.keep(sel), *args, **kwargs)

    utils.get_plddt_pde = get_plddt_pde

    for losses_class in (placer_module.losses.StructureLossesCSD,
                         placer_module.losses.StructureLossesPDB):
        original_losses = losses_class.get_losses

        def get_losses(self, *args, _original=original_losses, **kwargs):
            topology = kwargs.get("topology")
            if topology is None and len(args) >= 4:
                topology = args[3]
            if topology is not None and "observed" in topology:
                topology = dict(topology)
                topology["observed"] = mask.keep_mask(topology["observed"])
                if "topology" in kwargs:
                    kwargs["topology"] = topology
                else:
                    args = args[:3] + (topology,) + args[4:]
            return _original(self, *args, **kwargs)

        losses_class.get_losses = get_losses

    return mask
