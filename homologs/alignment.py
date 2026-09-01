
from collections import Counter
from pathlib import Path

from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment

def parse_msa(msa_path):
    """
    Description:
        Read a MAFFT-aligned fasta as a Biopython MultipleSeqAlignment,
        cleaning each record id to its first token.

    Args:
        msa_path: Path to the aligned fasta file.

    Returns:
        Bio.Align.MultipleSeqAlignment with cleaned record ids.
    """
    alignment = AlignIO.read(str(msa_path), "fasta")

    for rec in alignment:
        first_token = rec.description.split("|", 1)[0].strip()
        rec.id = first_token
        rec.name = first_token
    return alignment




def alignment_as_dict(alignment):
    """
    Description:
        Build a {seq_id: uppercase aligned sequence} dict for fast id-based
        access.

    Args:
        alignment: Bio.Align.MultipleSeqAlignment.

    Returns:
        Dict {seq_id: uppercase_aligned_str}.
    """
    return {rec.id: str(rec.seq).upper() for rec in alignment}


def column_conservation(alignment, ignore_gap=True):
    """
    Description:
        Compute the dominant residue and its frequency for every alignment
        column.

    Args:
        alignment: Bio.Align.MultipleSeqAlignment.
        ignore_gap: If True, gaps are excluded from the denominator.

    Returns:
        List of (dominant_aa, frequency, n_non_gap) tuples, one per column.
        Empty (all-gap) columns yield ('-', 0.0, 0).
    """
    n_col = alignment.get_alignment_length()
    out = []
    for c in range(n_col):
        col = str(alignment[:, c]).upper()
        if ignore_gap:
            col = col.replace("-", "")
        if not col:
            out.append(("-", 0.0, 0))
            continue
        aa, n = Counter(col).most_common(1)[0]
        out.append((aa, n / len(col), len(col)))
    return out


def find_conserved_columns(alignment, target_aa, min_freq=0.9, min_coverage=0.5):
    """
    Description:
        Return MSA columns where target_aa is the dominant non-gap residue
        and passes the coverage and frequency thresholds.

    Args:
        alignment: Bio.Align.MultipleSeqAlignment.
        target_aa: Single-letter amino acid code (e.g. 'K', 'S', 'D', 'H').
        min_freq: Minimum fraction of non-gap residues equal to target_aa.
        min_coverage: Minimum fraction of all sequences with a non-gap residue.

    Returns:
        List of dicts shaped as [{'col', 'aa', 'freq', 'coverage'}, ...].
    """
    n_seq = len(alignment)
    n_col = alignment.get_alignment_length()
    hits = []
    for c in range(n_col):
        col = str(alignment[:, c]).upper()
        non_gap = col.replace("-", "")
        cov = len(non_gap) / n_seq
        if cov < min_coverage or not non_gap:
            continue
        aa, n = Counter(non_gap).most_common(1)[0]
        f = n / len(non_gap)
        if aa == target_aa and f >= min_freq:
            hits.append({"col": c, "aa": aa, "freq": f, "coverage": cov})
    return hits

def resid_to_col(aligned_seq, resid_1based):
    """
    Description:
        Convert a 1-based ungapped residue index to its 0-based MSA column
        index.

    Args:
        aligned_seq: One aligned sequence string (with '-' as gap).
        resid_1based: 1-based residue index in the ungapped sequence.

    Returns:
        0-based MSA column index where this residue sits.
    """
    assert resid_1based >= 1, "resid_1based must be >= 1"
    seen = 0
    for c, ch in enumerate(aligned_seq):
        if ch == "-":
            continue
        seen += 1
        if seen == resid_1based:
            return c
    raise ValueError(
        f"resid_1based={resid_1based} exceeds ungapped length "
        f"{seen} of this sequence"
    )


def locate_catalytic_from_reference(alignment, ref_id, catalytic_map):
    """
    Description:
        Map known catalytic residues (1-based) on a reference sequence to
        MSA columns, with cross-checks. The reference is matched by
        substring containment in record ids.

    Args:
        alignment: Bio.Align.MultipleSeqAlignment.
        ref_id: Substring to match against record ids; the first record
            whose id contains ref_id is used.
        catalytic_map: Dict {label: (resid_1based, expected_aa)};
            expected_aa can be None to skip identity checking.

    Returns:
        Dict {label: {'col', 'ref_resid_1based', 'ref_aa', 'expected_aa',
                      'match', 'msa_dominant_aa', 'msa_freq',
                      'matched_ref_id'}}.
    """
    # Substring match against record ids; first hit wins.
    ref_seq = None
    matched_id = None
    for rec in alignment:
        if ref_id in rec.id:
            ref_seq = str(rec.seq).upper()
            matched_id = rec.id
            break
    if ref_seq is None:
        raise KeyError(f"No record id contains '{ref_id}'")

    cons = column_conservation(alignment)

    out = {}
    for label, (resid, expected_aa) in catalytic_map.items():
        col = resid_to_col(ref_seq, resid)
        ref_aa = ref_seq[col]
        dom_aa, dom_f, _ = cons[col]
        match = (expected_aa is None) or (ref_aa == expected_aa)
        out[label] = {
            "col": col,
            "ref_resid_1based": resid,
            "ref_aa": ref_aa,
            "expected_aa": expected_aa,
            "match": match,
            "msa_dominant_aa": dom_aa,
            "msa_freq": dom_f,
            "matched_ref_id": matched_id,
        }
    return out


def col_to_resid(aligned_seq, col):
    """
    Description:
        Convert a 0-based MSA column to its 0-based ungapped residue index,
        or None if the column is a gap in this sequence.

    Args:
        aligned_seq: One aligned sequence string (with '-' as gap).
        col: 0-based MSA column index.

    Returns:
        0-based residue index in the ungapped sequence, or None when the
        column is a gap in this sequence.
    """
    if aligned_seq[col] == "-":
        return None
    return sum(1 for ch in aligned_seq[: col + 1] if ch != "-") - 1


def locate_residues(alignment, msa_cols, labels=None):
    """
    Description:
        For every sequence, report the residue identity and 1-based index at
        each requested MSA column.

    Args:
        alignment: Bio.Align.MultipleSeqAlignment.
        msa_cols: Iterable of 0-based MSA column indices to query.
        labels: Optional iterable of human-readable names, same length as
            msa_cols. Defaults to 'col_<index>'.

    Returns:
        Dict shaped as
        {seq_id: {label: {'col': c, 'aa': aa, 'resid_1based': i_or_None}}}.
    """
    cols = list(msa_cols)
    if labels is None:
        labels = [f"col_{c}" for c in cols]
    assert len(labels) == len(cols), "labels must align with msa_cols"

    msa = alignment_as_dict(alignment)
    out = {}
    for sid, aln in msa.items():
        per_seq = {}
        for col, lab in zip(cols, labels):
            aa = aln[col]
            r0 = col_to_resid(aln, col)
            per_seq[lab] = {
                "col": col,
                "aa": aa,
                "resid_1based": None if r0 is None else r0 + 1,
            }
        out[sid] = per_seq
    return out

def map_catalytic_across_msa(alignment, ref_id, catalytic_map):
    """
    Description:
        End-to-end shortcut: resolve catalytic columns from a reference
        sequence and report each column's residue and 1-based index for
        every sequence in the MSA.

    Args:
        alignment: Bio.Align.MultipleSeqAlignment.
        ref_id: Substring to match against record ids.
        catalytic_map: Dict {label: (resid_1based, expected_aa)};
            expected_aa can be None to skip identity checking.

    Returns:
        Dict with two keys:
            'reference': output of locate_catalytic_from_reference (column
                resolution + per-column verification).
            'per_sequence': output of locate_residues, i.e.
                {seq_id: {label: {'col', 'aa', 'resid_1based'}}}.
    """
    ref_info = locate_catalytic_from_reference(alignment, ref_id, catalytic_map)

    cols = [v["col"] for v in ref_info.values()]
    labels = list(ref_info.keys())
    per_seq = locate_residues(alignment, cols, labels)

    return {"reference": ref_info, "per_sequence": per_seq}


def find_pcp_serine(alignment, min_ser_freq=0.85, motif="GxxS",
                    min_motif_freq=0.5, position_range=None):
    """
    Description:
        Locate the conserved PCP active-site Ser column: the strongest
        highly-conserved Ser whose preceding context matches the motif
        (default 'GxxS').

    Args:
        alignment: Bio.Align.MultipleSeqAlignment.
        min_ser_freq: Minimum non-gap frequency of Ser at the candidate col.
        motif: Motif ending in 'S'; 'x' is a wildcard. Default 'GxxS'.
        min_motif_freq: Minimum non-gap frequency required at each non-x
            motif position other than the Ser anchor.
        position_range: Optional (start_col, end_col) restricting search.

    Returns:
        Dict shaped as {'col', 'ser_freq', 'ser_coverage', 'motif'} for the
        best hit, or None if nothing passes the filters.
    """
    assert motif.endswith("S"), "motif must end with 'S'"
    cons = column_conservation(alignment)
    n_col = len(cons)
    motif_len = len(motif)

    lo, hi = (0, n_col) if position_range is None else position_range
    hi = min(hi, n_col)
    n_seq = len(alignment)

    best = None
    for c in range(max(lo, motif_len - 1), hi):
        aa, f, cov_n = cons[c]
        if aa != "S" or f < min_ser_freq:
            continue
        start = c - (motif_len - 1)
        ok = True
        for i, m in enumerate(motif[:-1]):
            if m == "x":
                continue
            cc, cf, _ = cons[start + i]
            if cc != m or cf < min_motif_freq:
                ok = False
                break
        if not ok:
            continue
        cand = {
            "col": c,
            "ser_freq": f,
            "ser_coverage": cov_n / n_seq,
            "motif": motif,
        }
        if best is None or f > best["ser_freq"]:
            best = cand
    return best


def truncate_at_column(alignment, col, include_col=False):
    """
    Description:
        Truncate each sequence to its N-terminal portion up to the given MSA
        column, returning gap-stripped unaligned sequences.

    Args:
        alignment: Bio.Align.MultipleSeqAlignment.
        col: 0-based MSA column index used as the truncation anchor.
        include_col: If True, residues at column `col` are kept; otherwise
            the cut stops just before the anchor.

    Returns:
        Dict {seq_id: truncated_unaligned_sequence}.
    """
    upper = col + 1 if include_col else col
    sub_aln = alignment[:, :upper]
    return {rec.id: str(rec.seq).replace("-", "") for rec in sub_aln}


def write_fasta(seqs, out_path, line_width=60):
    """
    Description:
        Write a {id: sequence} dict to a fasta file, wrapping each sequence
        at line_width characters per line (set <= 0 for single-line output).

    Args:
        seqs: Dict of {seq_id: sequence_string}; aligned or unaligned both ok.
        out_path: Destination fasta path.
        line_width: Max characters per sequence line; <= 0 means single line.

    Returns:
        Path object pointing to the written file.
    """
    out = Path(out_path)
    with out.open("w") as fh:
        for sid, s in seqs.items():
            fh.write(f">{sid}\n")
            if line_width and line_width > 0:
                for i in range(0, len(s), line_width):
                    fh.write(s[i:i + line_width] + "\n")
            else:
                fh.write(s + "\n")
    return out