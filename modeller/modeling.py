import os
import re
import csv
import json
import shutil
import argparse
import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr

from modeller import Environ
from modeller.automodel import AutoModel, assess, refine
from tqdm import tqdm


REFINEMENT_LEVELS = {
    "very_fast": refine.very_fast,
    "fast": refine.fast,
    "slow": refine.slow,
    "very_slow": refine.very_slow,
}

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


def read_alignment_codes(alignment_path):
    """
    Description:
        Read template and target identifiers from a MODELLER PIR alignment.
        All entries except the last are interpreted as templates.

    Arg:
        alignment_path: Path to the MODELLER PIR alignment file.

    Return:
        Tuple containing all template codes and the target code.
    """
    codes = []

    with open(alignment_path) as handle:
        for line in handle:
            if line.startswith(">P1;"):
                codes.append(line.split(";", 1)[1].strip())

    if len(codes) < 2:
        raise ValueError(
            "Alignment must contain at least one template and one target."
        )

    template_codes = tuple(codes[:-1])
    target_code = codes[-1]

    return template_codes, target_code


def remove_intermediate_files(output_dir):
    """
    Description:
        Remove MODELLER optimization and diagnostic intermediate files.

    Arg:
        output_dir: Target-specific modeling output directory.

    Return:
        None.
    """
    for filename in os.listdir(output_dir):
        path = os.path.join(output_dir, filename)

        if filename.endswith((".ini", ".rsr", ".sch")):
            os.remove(path)

        elif re.search(r"\.[DV]\d+$", filename):
            os.remove(path)


def write_scores(models, output_path):
    """
    Description:
        Write ranked MODELLER scores for one target to a CSV file.

    Args:
        models: List of successful MODELLER output dictionaries.
        output_path: Output CSV file path.

    Return:
        None.
    """
    columns = [
        "rank", "model_name", "molpdf", "dope_score",
        "normalized_dope", "ga341_score",
    ]

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()

        for rank, model in enumerate(models, start=1):
            writer.writerow({
                "rank": rank,
                "model_name": model.get("name", ""),
                "molpdf": model.get("molpdf", ""),
                "dope_score": model.get("DOPE score", ""),
                "normalized_dope": model.get(
                    "Normalized DOPE score",
                    "",
                ),
                "ga341_score": model.get("GA341 score", ""),
            })


def model_target(target_dir_name, alignment_dir, template_dir,
                 expected_template_codes, output_dir, n_models,
                 refinement, keep_intermediates, best_only, overwrite):
    """
    Description:
        Build comparative models for one target using all templates listed in
        its PIR alignment and save the best-scoring structure.

    Args:
        target_dir_name: Name of the target-specific alignment directory.
        alignment_dir: Root directory containing target alignment folders.
        template_dir: Directory containing the template PDB files.
        expected_template_codes: Template codes expected from the input path.
        output_dir: Root directory for target-specific model results.
        n_models: Number of independent models generated for the target.
        refinement: MODELLER refinement level name.
        keep_intermediates: Whether to retain MODELLER diagnostic files.
        best_only: Whether to retain only best_model.pdb.
        overwrite: Whether to replace an existing completed result.

    Return:
        Summary dict containing status and best-model scores.
    """
    input_dir = os.path.join(
        alignment_dir,
        target_dir_name,
    )
    alignment_path = os.path.join(
        input_dir,
        "alignment.ali",
    )

    template_codes, target_code = read_alignment_codes(
        alignment_path
    )

    if template_codes != tuple(expected_template_codes):
        raise ValueError(
            "Template codes in the alignment do not match the supplied "
            f"template input. Alignment: {template_codes}; "
            f"input: {tuple(expected_template_codes)}"
        )

    target_output_dir = os.path.join(
        output_dir,
        target_code,
    )
    best_model_path = os.path.join(
        target_output_dir,
        "best_model.pdb",
    )

    if os.path.exists(best_model_path) and not overwrite:
        return {
            "target_id": target_code,
            "status": "skipped",
            "best_model": best_model_path,
            "dope_score": "",
            "normalized_dope": "",
            "ga341_score": "",
            "error": "",
        }

    if os.path.isdir(target_output_dir):
        shutil.rmtree(target_output_dir)

    os.makedirs(target_output_dir, exist_ok=True)

    for filename in [
        "alignment.ali",
        "alignment.pap",
        "target.fasta",
    ]:
        source_path = os.path.join(
            input_dir,
            filename,
        )

        if os.path.exists(source_path):
            shutil.copy2(
                source_path,
                os.path.join(target_output_dir, filename),
            )

    local_alignment = os.path.join(
        target_output_dir,
        "alignment.ali",
    )
    log_path = os.path.join(
        target_output_dir,
        "modeller.log",
    )

    original_dir = os.getcwd()

    try:
        os.chdir(target_output_dir)

        with open(log_path, "w") as log_handle:
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                env = Environ()

                env.io.atom_files_directory = [
                    template_dir,
                    target_output_dir,
                ]

                knowns = (
                    template_codes[0]
                    if len(template_codes) == 1
                    else template_codes
                )

                model = AutoModel(
                    env,
                    alnfile=local_alignment,
                    knowns=knowns,
                    sequence=target_code,
                    assess_methods=(
                        assess.DOPE,
                        assess.normalized_dope,
                        assess.GA341,
                    ),
                )

                model.starting_model = 1
                model.ending_model = n_models
                model.md_level = REFINEMENT_LEVELS[refinement]

                model.make()

        successful_models = [
            result for result in model.outputs
            if result.get("failure") is None
        ]

        if not successful_models:
            raise RuntimeError(
                "No MODELLER model was generated successfully."
            )

        successful_models.sort(
            key=lambda result: result["DOPE score"]
        )

        write_scores(
            successful_models,
            os.path.join(target_output_dir, "scores.csv"),
        )

        best_model = successful_models[0]
        best_model_name = os.path.basename(
            best_model["name"]
        )

        shutil.copy2(
            os.path.join(target_output_dir, best_model_name),
            best_model_path,
        )

        if best_only:
            for filename in os.listdir(target_output_dir):
                if filename.endswith(".pdb") and ".B9999" in filename:
                    os.remove(
                        os.path.join(target_output_dir, filename)
                    )

        if not keep_intermediates:
            remove_intermediate_files(target_output_dir)

        summary = {
            "target_id": target_code,
            "status": "completed",
            "best_model": best_model_name,
            "dope_score": best_model.get("DOPE score", ""),
            "normalized_dope": best_model.get(
                "Normalized DOPE score",
                "",
            ),
            "ga341_score": str(
                best_model.get("GA341 score", "")
            ),
            "error": "",
        }

        with open(
            os.path.join(target_output_dir, "status.json"),
            "w",
        ) as handle:
            json.dump(summary, handle, indent=2)

        return summary

    except Exception as exc:
        summary = {
            "target_id": target_code,
            "status": "failed",
            "best_model": "",
            "dope_score": "",
            "normalized_dope": "",
            "ga341_score": "",
            "error": str(exc),
        }

        with open(
            os.path.join(target_output_dir, "status.json"),
            "w",
        ) as handle:
            json.dump(summary, handle, indent=2)

        with open(log_path, "a") as log_handle:
            log_handle.write("\n\nPython traceback\n")
            log_handle.write(traceback.format_exc())

        return summary

    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build MODELLER structures using one template PDB or all PDB "
            "files contained in a template directory."
        )
    )
    parser.add_argument("-a", "--alignment_dir", type=str, required=True,
                        help="Directory produced by alignments.py.")
    parser.add_argument("-t", "--template", type=str, required=True,
                        help="Path to one template PDB or a directory of PDBs.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to write target-specific models.")
    parser.add_argument("-n", "--n_models", type=int, default=5,
                        help="Number of models generated per target. Default: 5.")
    parser.add_argument("-r", "--refinement", type=str, default="fast",
                        choices=list(REFINEMENT_LEVELS),
                        help="MODELLER refinement level. Default: fast.")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="Number of targets modeled in parallel. Default: 1.")
    parser.add_argument("-b", "--best_only", action="store_true",
                        help="Keep only best_model.pdb for each target.")
    parser.add_argument("-k", "--keep_intermediates", action="store_true",
                        help="Keep .ini, .rsr, .sch, .D*, and .V* files.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Regenerate targets that already have best_model.pdb.")
    args = parser.parse_args()

    alignment_dir = os.path.abspath(
        args.alignment_dir
    )
    output_dir = os.path.abspath(
        args.output_dir
    )

    if not os.path.isdir(alignment_dir):
        parser.error(
            f"Alignment directory not found: {alignment_dir}"
        )

    try:
        templates = get_template_files(args.template)
    except Exception as exc:
        parser.error(str(exc))

    if args.n_models < 1:
        parser.error("--n_models must be at least 1.")

    if args.workers < 1:
        parser.error("--workers must be at least 1.")

    os.makedirs(output_dir, exist_ok=True)

    template_dir = os.path.dirname(
        templates[0]["path"]
    )
    template_codes = [
        template["code"]
        for template in templates
    ]

    target_dirs = sorted(
        name for name in os.listdir(alignment_dir)
        if os.path.isfile(
            os.path.join(
                alignment_dir,
                name,
                "alignment.ali",
            )
        )
    )

    if not target_dirs:
        parser.error(
            "No target directories containing alignment.ali were found."
        )

    params = [
        (
            target_dir,
            alignment_dir,
            template_dir,
            template_codes,
            output_dir,
            args.n_models,
            args.refinement,
            args.keep_intermediates,
            args.best_only,
            args.overwrite,
        )
        for target_dir in target_dirs
    ]

    results = []

    if args.workers == 1:
        for values in tqdm(
            params,
            desc="Modeling targets",
        ):
            try:
                result = model_target(*values)

            except Exception as exc:
                result = {
                    "target_id": values[0],
                    "status": "failed",
                    "best_model": "",
                    "dope_score": "",
                    "normalized_dope": "",
                    "ga341_score": "",
                    "error": str(exc),
                }

            results.append(result)

            if result["status"] == "failed":
                tqdm.write(
                    f"[skip] {result['target_id']}: "
                    f"{result['error']}"
                )

    else:
        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    model_target,
                    *values,
                ): values[0]
                for values in params
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Modeling targets",
            ):
                target_dir = futures[future]

                try:
                    result = future.result()

                except Exception as exc:
                    result = {
                        "target_id": target_dir,
                        "status": "failed",
                        "best_model": "",
                        "dope_score": "",
                        "normalized_dope": "",
                        "ga341_score": "",
                        "error": str(exc),
                    }

                results.append(result)

                if result["status"] == "failed":
                    tqdm.write(
                        f"[skip] {result['target_id']}: "
                        f"{result['error']}"
                    )

    summary_path = os.path.join(
        output_dir,
        "modeling_summary.csv",
    )

    columns = [
        "target_id", "status", "best_model", "dope_score",
        "normalized_dope", "ga341_score", "error",
    ]

    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                results,
                key=lambda result: result["target_id"],
            )
        )

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
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()