import os
import time
import argparse

from pymol import cmd

import pyrosetta
from pyrosetta import Pose, get_fa_scorefxn, Vector1
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols import docking, rigid

def load_ligand_params_to_pose(params, pose):
    """Attach ligand params to a pose's residue type set.

    Args:
        params: List of params file paths.
        pose: Pose to attach the residue types to.
    """
    if len(params) != 0 and params[0] != "":
        params = pyrosetta.Vector1(params)
        res_set = pose.conformation().modifiable_residue_type_set_for_conf()
        res_set.read_files_for_base_residue_types(params)
        pose.conformation().reset_residue_type_set_for_conf(res_set)


def strip_hydrogens_pymol(in_pdb, out_pdb):
    """Load a PDB in PyMOL, remove hydrogens, and save it back.

    Args:
        in_pdb: Path to the input PDB (with hydrogens).
        out_pdb: Path to write the hydrogen-free PDB.
    """
    cmd.load(in_pdb, "obj")
    cmd.remove("hydrogens")
    cmd.save(out_pdb, "obj")
    cmd.delete("obj")


def get_ligand_coords(pose):
    """Collect heavy-atom xyz of every ligand residue in the pose.

    Arg:
        pose: Input pose.

    Return:
        List of xyzVector for all ligand heavy atoms.
    """
    ligand_atoms = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_ligand():
            for a in range(1, res.natoms() + 1):
                if res.atom_type(a).is_heavyatom():
                    ligand_atoms.append(res.xyz(a))
    return ligand_atoms


def min_distance(atom_xyz, ligand_coords):
    """Minimum distance from a single atom to any ligand atom.

    Args:
        atom_xyz: xyzVector of the query atom.
        ligand_coords: List of ligand heavy-atom xyzVectors.

    Return:
        Minimum distance, or None if atom_xyz is None.
    """
    if atom_xyz is None:
        return None
    return min((atom_xyz - l).norm() for l in ligand_coords)


def determine_repack_residues(pose, cut3=10.0, cut4=12.0):
    """Pick protein residues near the ligand for repack/relax.

    Mirrors the repack logic from run_mutation.py (design region dropped).

    Args:
        pose: Complex pose with ligand present.
        cut3: CA distance cutoff for inclusion.
        cut4: Extended CA cutoff if CB is closer than CA.

    Return:
        List of residue indices to repack.
    """
    ligand_atoms = get_ligand_coords(pose)
    repackable = []

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue

        ca_xyz = res.xyz("CA")
        d_ca = min_distance(ca_xyz, ligand_atoms)

        has_cb = res.has("CB")
        d_cb = None
        if has_cb:
            d_cb = min_distance(res.xyz("CB"), ligand_atoms)

        if d_ca <= cut3:
            repackable.append(i)
        elif d_ca <= cut4 and has_cb and d_cb < d_ca:
            repackable.append(i)

    return repackable


def run_fastrelax(pose, scorefxn, repack):
    """Apply FastRelax with bb+chi unlocked only on selected residues.

    Args:
        pose: Pose to relax in place.
        scorefxn: Score function (will be cloned and reweighted).
        repack: List of residue indices allowed to move.
    """
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    for r in repack:
        mm.set_bb(r, True)
        mm.set_chi(r, True)

    for i in range(1, pose.total_residue() + 1):
        if pose.residue(i).is_ligand():
            mm.set_chi(i, True)

    mm.set_jump(False)
    # ft = pose.fold_tree()
    # for j in range(1, pose.num_jump() + 1):
    #     downstream_res = ft.downstream_jump_residue(j)
    #     if pose.residue(downstream_res).is_ligand():
    #         mm.set_jump(j, True)

    scorefxn = scorefxn.clone()
    scorefxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0
    )

    if not os.getenv("DEBUG"):
        relax = FastRelax(scorefxn, 1)
        relax.constrain_relax_to_start_coords(True)
        relax.max_iter(100)
        relax.set_movemap(mm)
        relax.apply(pose)


def unbind(pose, partners):
    """Translate one partner far away to estimate the unbound state.

    Args:
        pose: Pose to modify in place.
        partners: Docking partner string, e.g. "A_X".
    """
    STEP_SIZE = 100
    JUMP = 1
    docking.setup_foldtree(pose, partners, Vector1([-1, -1, -1]))
    trans_mover = rigid.RigidBodyTransMover(pose, JUMP)
    trans_mover.step_size(STEP_SIZE)
    trans_mover.apply(pose)


def local_docking(pose, partners):
    """Rigid-body dock the ligand into the pocket via Monte Carlo.

    Args:
        pose: Pose to dock in place.
        partners: Docking partner string, e.g. "A_X".
    """
    dock_jump = 1
    docking.setup_foldtree(pose, partners, pyrosetta.Vector1([dock_jump]))
    dock_scorefxn = pyrosetta.create_score_function("ligand.wts")
    docking_protocol = docking.DockMCMProtocol()
    docking_protocol.set_scorefxn(dock_scorefxn)
    docking_protocol.apply(pose)


def compute_binding_metrics(pose, scorefxn, partners):
    """Compute bound, unbound, and binding scores via partner translation.

    Args:
        pose: Pose to score (cloned internally before unbinding).
        scorefxn: Score function.
        partners: Docking partner string (e.g. 'A_X'). Must not be None.

    Return:
        Dict with keys 'bound', 'unbound', 'binding'.
    """
    bound = scorefxn(pose)
    tmp = Pose()
    tmp.assign(pose)
    unbind(tmp, partners)
    unbound = scorefxn(tmp)
    return {"bound": bound, "unbound": unbound, "binding": unbound - bound}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Directory containing input PDB files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to write relaxed PDB files.")
    parser.add_argument("-pr", "--params", type=str, default=None, nargs="+",
                        help="Ligand path(s) in format 'LIG:path/to/ligand.mol2' (optional, multiple allowed)")
    parser.add_argument("-cm", "--compute_metrics", action="store_true", default=False,
                        help="If set, compute pre/post-relax energies and "
                             "docking energies (requires --partners), and write TSV.")
    parser.add_argument("-rd", "--run_docking", action="store_true", default=False,
                        help="If set, run local docking before FastRelax "
                             "(requires --partners).")
    parser.add_argument("-pt", "--partners", type=str, default=None,
                        help="Partner string for docking energy (e.g. 'A_X'). "
                             "Required when --compute_metrics is set.")
    args = parser.parse_args()

    if args.compute_metrics and args.partners is None:
        parser.error("--compute_metrics requires --partners (e.g. 'A_X').")
    if args.run_docking and args.partners is None:
        parser.error("--run_docking requires --partners (e.g. 'A_X').")

    pyrosetta.init()
    os.makedirs(args.output_dir, exist_ok=True)

    pdb_files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".pdb")
    ])

    if not pdb_files:
        print(f"No PDB files found in {args.input_dir}")
        return

    scorefxn = get_fa_scorefxn()
    params_list = args.params if args.params else [""]

    tsv_path = None
    if args.compute_metrics:
        tsv_path = os.path.join(args.output_dir, "relax_metrics.tsv")
        with open(tsv_path, "w") as f:
            f.write(
                "filename\t"
                "pre_bound\tpre_unbound\tpre_binding\t"
                "post_bound\tpost_unbound\tpost_binding\t"
                "delta_bound\tdelta_binding\n"
            )

    total = len(pdb_files)
    start_time = time.time()

    for idx, pdb in enumerate(pdb_files, start=1):
        name = os.path.splitext(os.path.basename(pdb))[0]

        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] START: {name}")
        print(f"{'#' * 70}")

        pose = Pose()
        load_ligand_params_to_pose(params_list, pose)
        pyrosetta.io.pose_from_file(pose, pdb)

        pre = None
        if args.compute_metrics:
            pre = compute_binding_metrics(pose, scorefxn, args.partners)

        repack = determine_repack_residues(pose)
        run_fastrelax(pose, scorefxn, repack)

        if args.run_docking:
            local_docking(pose, args.partners)

        out_path = os.path.join(args.output_dir, f"{name}.relax.pdb")
        
        pose.dump_pdb(out_path)

        # Strip hydrogens using PyMOL
        strip_hydrogens_pymol(out_path, out_path)

        if args.compute_metrics:
            post = compute_binding_metrics(pose, scorefxn, args.partners)
            row = (
                f"{name}\t"
                f"{pre['bound']}\t{pre['unbound']}\t{pre['binding']}\t"
                f"{post['bound']}\t{post['unbound']}\t{post['binding']}\t"
                f"{post['bound'] - pre['bound']}\t"
                f"{post['binding'] - pre['binding']}\n"
            )
            with open(tsv_path, "a") as f:
                f.write(row)

        elapsed = time.time() - start_time
        avg = elapsed / idx
        eta = avg * (total - idx)

        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] DONE: {name}  ->  {out_path}")
        print(f"# Progress: {idx}/{total} ({100 * idx / total:.1f}%)  "
              f"| Elapsed: {elapsed / 60:.1f} min  | ETA: {eta / 60:.1f} min")
        print(f"{'#' * 70}")

if __name__ == "__main__":
    main()