import os
import re
import csv
import argparse

from Bio import SeqIO
from modeller import Alignment, Environ, Model
from tqdm import tqdm


# All template structures are read from chain A

TEMPLATE_CHAIN = "A"

# Internal SALIGN parameters for structure-guided target alignment

MAX_GAP_LENGTH = 20

GAP_PENALTIES_1D = (-450, 0)

GAP_PENALTIES_2D = (
    0.35, 1.2, 0.9,
    1.2, 0.6, 8.6,
    1.2, 0.0, 0.0,
)

VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYX")
VALID_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def get_template_files(template_input):
    """
    Description:
        Read one template PDB or all PDB files directly contained in a
        template directory.

    Arg:
        template_input: Path to one PDB file or a directory of PDB files.

    Return:
        List of template dicts containing code, path, and filename.
    """
    template_input = os.path.abspath(template_input)

    if os.path.isfile(template_input):
        if not template_input.lower().endswith(".pdb"):
            raise ValueError(
                f"Template file must have a .pdb extension: {template_input}"
            )

        template_paths = [template_input]

    elif os.path.isdir(template_input):
        template_paths = sorted(
            os.path.join(template_input, filename)
            for filename in os.listdir(template_input)
            if filename.lower().endswith(".pdb")
            and os.path.isfile(os.path.join(template_input, filename))
        )

        if not template_paths:
            raise ValueError(
                f"No PDB files were found in: {template_input}"
            )

    else:
        raise FileNotFoundError(
            f"Template file or directory not found: {template_input}"
        )

    templates = []

    for template_path in template_paths:
        template_name = os.path.basename(template_path)
        template_code = os.path.splitext(template_name)[0]

        if not VALID_CODE_PATTERN.fullmatch(template_code):
            raise ValueError(
                f"Invalid template filename '{template_name}'. "
                "The basename may contain only letters, numbers, '.', '_', "
                "and '-'."
            )

        templates.append({
            "code": template_code,
            "path": template_path,
            "name": template_name,
        })

    template_codes = [template["code"] for template in templates]

    if len(template_codes) != len(set(template_codes)):
        raise ValueError("Duplicate template codes were found.")

    return templates


def create_template_alignment(templates, output_dir, env):
    """
    Description:
        Load all template structures and generate a multiple structural
        alignment when more than one template is provided.

    Args:
        templates: List of template specification dicts.
        output_dir: Root directory for alignment results.
        env: MODELLER environment.

    Return:
        Path to the generated template PIR alignment.
    """
    alignment = Alignment(env)

    for template in templates:
        model = Model(
            env,
            file=template["name"],
            model_segment=(
                f"FIRST:{TEMPLATE_CHAIN}",
                f"LAST:{TEMPLATE_CHAIN}",
            ),
        )

        alignment.append_model(
            model,
            atom_files=template["name"],
            align_codes=template["code"],
        )

    # Multiple templates must first be aligned using their 3D structures
    if len(templates) > 1:
        feature_weight_sets = [
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (1.0, 0.5, 1.0, 1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        ]

        tree_path = os.path.join(
            output_dir,
            "template_alignment.tree",
        )

        for feature_weights in feature_weight_sets:
            alignment.salign(
                rms_cutoff=3.5,
                normalize_pp_scores=False,
                rr_file="$(LIB)/as1.sim.mat",
                overhang=30,
                gap_penalties_1d=(-450, -50),
                gap_penalties_3d=(0, 3),
                gap_gap_score=0,
                gap_residue_score=0,
                dendrogram_file=tree_path,
                alignment_type="tree",
                feature_weights=feature_weights,
                improve_alignment=True,
                fit=True,
                write_fit=False,
                write_whole_pdb=False,
                output="ALIGNMENT QUALITY",
            )

    pir_path = os.path.join(
        output_dir,
        "template_alignment.ali",
    )
    pap_path = os.path.join(
        output_dir,
        "template_alignment.pap",
    )

    alignment.write(
        file=pir_path,
        alignment_format="PIR",
    )
    alignment.write(
        file=pap_path,
        alignment_format="PAP",
    )

    return pir_path


def create_target_alignment(record, target_id, template_alignment,
                            output_dir, env):
    """
    Description:
        Align one target sequence to the pre-aligned template structure block
        and write MODELLER PIR and human-readable PAP files.

    Args:
        record: BioPython SeqRecord containing the target protein sequence.
        target_id: Target identifier used in filenames and PIR entries.
        template_alignment: PIR alignment containing all template structures.
        output_dir: Root directory for target-specific alignment results.
        env: MODELLER environment.

    Return:
        Summary dict containing target ID, sequence length, and status.
    """
    sequence = str(record.seq).replace("*", "").upper()
    invalid_residues = sorted(set(sequence) - VALID_RESIDUES)

    if not sequence:
        raise ValueError(f"Target '{target_id}' has an empty sequence.")

    if invalid_residues:
        raise ValueError(
            f"Target '{target_id}' contains invalid residues: "
            f"{invalid_residues}"
        )

    target_dir = os.path.join(output_dir, target_id)
    os.makedirs(target_dir, exist_ok=True)

    alignment = Alignment(env)

    # Read the single-template or multiple-template alignment
    alignment.append(
        file=template_alignment,
        alignment_format="PIR",
        align_codes="all",
    )

    template_count = len(alignment)

    # Add the target sequence after all template entries
    alignment.append_sequence(sequence)
    alignment[-1].code = target_id

    # Preserve the template block and align only the target sequence
    alignment.salign(
        rr_file="$(LIB)/as1.sim.mat",
        alignment_type="PAIRWISE",
        align_what="BLOCK",
        align_block=template_count,
        output="",
        feature_weights=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gap_function=True,
        max_gap_length=MAX_GAP_LENGTH,
        overhang=0,
        gap_penalties_1d=GAP_PENALTIES_1D,
        gap_penalties_2d=GAP_PENALTIES_2D,
        similarity_flag=True,
    )

    alignment.check()

    alignment.write(
        file=os.path.join(target_dir, "alignment.ali"),
        alignment_format="PIR",
    )
    alignment.write(
        file=os.path.join(target_dir, "alignment.pap"),
        alignment_format="PAP",
    )

    with open(os.path.join(target_dir, "target.fasta"), "w") as handle:
        handle.write(f">{target_id}\n{sequence}\n")

    return {
        "source_id": record.id,
        "target_id": target_id,
        "sequence_length": len(sequence),
        "status": "completed",
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Align multiple target sequences to one template PDB or all PDB "
            "files contained in a template directory."
        )
    )
    parser.add_argument("-i", "--input_fasta", type=str, required=True,
                        help="Input multi-FASTA containing target sequences.")
    parser.add_argument("-t", "--template", type=str, required=True,
                        help="Path to one template PDB or a directory of PDBs.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to write target-specific alignments.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Regenerate alignments that already exist.")
    args = parser.parse_args()

    input_fasta = os.path.abspath(args.input_fasta)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(input_fasta):
        parser.error(f"Input FASTA not found: {input_fasta}")

    try:
        templates = get_template_files(args.template)
    except Exception as exc:
        parser.error(str(exc))

    target_records = list(SeqIO.parse(input_fasta, "fasta"))

    if not target_records:
        parser.error("No sequences were found in the input FASTA.")

    target_ids = [record.id for record in target_records]

    if len(target_ids) != len(set(target_ids)):
        parser.error("Duplicate target IDs were found in the input FASTA.")

    invalid_ids = [
        target_id for target_id in target_ids
        if not VALID_CODE_PATTERN.fullmatch(target_id)
    ]

    if invalid_ids:
        parser.error(
            "Target IDs may contain only letters, numbers, '.', '_', and '-': "
            f"{invalid_ids[:10]}"
        )

    template_codes = {
        template["code"]
        for template in templates
    }

    overlapping_ids = sorted(
        set(target_ids) & template_codes
    )

    if overlapping_ids:
        parser.error(
            "Target IDs must not match template codes: "
            f"{overlapping_ids[:10]}"
        )

    os.makedirs(output_dir, exist_ok=True)

    template_dir = os.path.dirname(templates[0]["path"])

    env = Environ()
    env.io.atom_files_directory = [template_dir]

    template_alignment = create_template_alignment(
        templates,
        output_dir,
        env,
    )

    template_summary_path = os.path.join(
        output_dir,
        "template_summary.csv",
    )

    with open(template_summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["template_code", "template_pdb", "chain"],
        )
        writer.writeheader()

        for template in templates:
            writer.writerow({
                "template_code": template["code"],
                "template_pdb": template["path"],
                "chain": TEMPLATE_CHAIN,
            })

    results = []

    for record in tqdm(target_records, desc="Aligning targets"):
        target_id = record.id
        alignment_path = os.path.join(
            output_dir,
            target_id,
            "alignment.ali",
        )

        if os.path.exists(alignment_path) and not args.overwrite:
            results.append({
                "source_id": record.id,
                "target_id": target_id,
                "sequence_length": len(record.seq),
                "status": "skipped",
                "error": "",
            })
            continue

        try:
            result = create_target_alignment(
                record,
                target_id,
                template_alignment,
                output_dir,
                env,
            )

        except Exception as exc:
            tqdm.write(
                f"[skip] {target_id}: {type(exc).__name__}: {exc}"
            )
            result = {
                "source_id": record.id,
                "target_id": target_id,
                "sequence_length": len(record.seq),
                "status": "failed",
                "error": str(exc),
            }

        results.append(result)

    summary_path = os.path.join(
        output_dir,
        "alignment_summary.csv",
    )

    columns = [
        "source_id", "target_id", "sequence_length", "status", "error"
    ]

    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(results)

    n_completed = sum(
        result["status"] == "completed"
        for result in results
    )
    n_skipped = sum(
        result["status"] == "skipped"
        for result in results
    )
    n_failed = sum(
        result["status"] == "failed"
        for result in results
    )

    mode = (
        "single-template"
        if len(templates) == 1
        else "multiple-template"
    )

    print(f"Template mode: {mode}")
    print(f"Templates: {len(templates)}")
    print(f"Completed: {n_completed}")
    print(f"Skipped: {n_skipped}")
    print(f"Failed: {n_failed}")
    print(f"Template alignment: {template_alignment}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()