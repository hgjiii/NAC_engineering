import os
import argparse

from functools import partial

import pyrosettacolabsetup; pyrosettacolabsetup.install_pyrosetta()
import pyrosetta; pyrosetta.init()
from pyrosetta import * 
from pyrosetta.rosetta.core.pose import *
from pyrosetta.rosetta.core.pack.task import *
from pyrosetta.rosetta.protocols import *
from pyrosetta.rosetta.core.select import *

from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.kinematics import MoveMap

one_letter_aa = ['G','A','L','M','F','W','K','Q','E','S','P','V','I','C','Y','H','R','N','D','T']
one_to_three = {
    'G':'GLY','A':'ALA','L':'LEU','M':'MET','F':'PHE','W':'TRP',
    'K':'LYS','Q':'GLN','E':'GLU','S':'SER','P':'PRO','V':'VAL',
    'I':'ILE','C':'CYS','Y':'TYR','H':'HIS','R':'ARG','N':'ASN',
    'D':'ASP','T':'THR'
}

def load_ligand_params_to_pose(params, pose):
    if len(params) != 0 and params[0] != "":
        params = pyrosetta.Vector1(params)
        res_set = pose.conformation().modifiable_residue_type_set_for_conf()
        res_set.read_files_for_base_residue_types(params)
        pose.conformation().reset_residue_type_set_for_conf(res_set)

def run_pack(pose, posi, amino, scorefxn, repack):

    # mutation position
    mut_posi = residue_selector.ResidueIndexSelector()
    mut_posi.set_index(posi)

    # repacking position
    repack_pos = residue_selector.ResidueIndexSelector()
    repack_pos.set_index(','.join(str(pos) for pos in repack))

    # fix_postion
    fix_pos = residue_selector.AndResidueSelector(
        residue_selector.NotResidueSelector(mut_posi),
        residue_selector.NotResidueSelector(repack_pos),
    )
    
    tf = TaskFactory()

    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.IncludeCurrent())
    tf.push_back(operation.NoRepackDisulfides())
    
    tf.push_back(
        operation.OperateOnResidueSubset(
            operation.PreventRepackingRLT(),
            fix_pos
        )
    )
    
    tf.push_back(
        operation.OperateOnResidueSubset(
            operation.RestrictToRepackingRLT(), 
            residue_selector.NotResidueSelector(mut_posi)
        )
    )
    
    # residue mutation
    aa_to_design = operation.RestrictAbsentCanonicalAASRLT()
    aa_to_design.aas_to_keep(amino)

    tf.push_back(
        operation.OperateOnResidueSubset(
            aa_to_design, 
            mut_posi
        )
    )

    packer = minimization_packing.PackRotamersMover()
    packer.task_factory(tf)

    if not os.getenv("DEBUG"):
        packer.apply(pose)

def run_fastrelax(pose, posi, amino, scorefxn, repack):
    mutate = MutateResidue()
    mutate.set_target(posi)
    # mutate.set_res_name(pyrosetta.rosetta.core.chemical.aa_from_one_letter(amino))
    mutate.set_res_name(one_to_three[amino])
    mutate.apply(pose)

    repack = repack + [posi]
    
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
    scorefxn = scorefxn.clone()
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0)
    
    if not os.getenv("DEBUG"):
        relax = FastRelax(scorefxn, 1)
        relax.constrain_relax_to_start_coords(True)
        relax.max_iter(100)
        # relax.set_scorefxn(scorefxn)
        relax.set_movemap(mm)
        relax.apply(pose)
    
def local_docking(pose,
                  partners,
                  # ligand_params=[""],
                  # jobs=1,
                  # job_output="ligand_output"
                 ):
    
    working_dir = os.getcwd()
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # if len(ligand_params) != 0 and ligand_params[0] != "":
    #     ligand_params = pyrosetta.Vector1(ligand_params)
    #     res_set = pose.conformation().modifiable_residue_type_set_for_conf()
    #     res_set.read_files_for_base_residue_types(ligand_params)
    #     pose.conformation().reset_residue_type_set_for_conf(res_set)
    
    dock_jump = 1
    docking.setup_foldtree(
        pose,
        partners,
        pyrosetta.Vector1([dock_jump])
    )
    
    scorefxn = pyrosetta.create_score_function("ligand.wts")
    docking_protocol = docking.DockMCMProtocol()
    docking_protocol.set_scorefxn(scorefxn)

    # Setup the PyJobDistributor
    # jd = toolbox.py_jobdistributor.PyJobDistributor(job_output,
    #                                                 jobs,
    #                                                 scorefxn,
    #                                                 compress=False)

    # while not jd.job_complete:
        # test_pose = pose.clone()
        # docking_protocol.apply(test_pose)
        # jd.output_decoy(test_pose)
    
    docking_protocol.apply(pose)
    os.chdir(working_dir)
    
def unbind(pose, partners):
    STEP_SIZE = 100
    JUMP = 1
    docking.setup_foldtree(pose, partners, Vector1([-1,-1,-1]))
    trans_mover = rigid.RigidBodyTransMover(pose,JUMP)
    trans_mover.step_size(STEP_SIZE)
    trans_mover.apply(pose)

def wildtype(aatype = 'AA.aa_gly'):
    AA = ['G','A','L','M','F','W','K','Q','E','S','P'
            ,'V','I','C','Y','H','R','N','D','T']

    AA_3 = ['AA.aa_gly','AA.aa_ala','AA.aa_leu','AA.aa_met','AA.aa_phe','AA.aa_trp'
            ,'AA.aa_lys','AA.aa_gln','AA.aa_glu', 'AA.aa_ser','AA.aa_pro','AA.aa_val'
            ,'AA.aa_ile','AA.aa_cys','AA.aa_tyr','AA.aa_his','AA.aa_arg','AA.aa_asn'
            ,'AA.aa_asp','AA.aa_thr']

    for i in range(0, len(AA_3)):
        if(aatype == AA_3[i]):
            return AA[i]
    
# def mutate(pose, posi, amino, repack_pos, partners, name, output_path, use_fastrelax):
#     #main function for mutation
#     PDB_PREFIX = f'{name}_'
#     pdb_files = 'pdb_files'
#     #Initiate test pose
#     testPose = Pose()
#     testPose.assign(pose)

#     #Initiate energy function
#     scorefxn = get_fa_scorefxn()
#     # unbind(testPose, partners)
#     # wt_unbound = scorefxn(testPose)
#     # testPose.assign(pose)
    
#     #Variables initiation
#     wt = wildtype(str(pose.aa(posi)))
#     mutation_id = f'{wt}{posi}{amino}'
#     pdbname = PDB_PREFIX + mutation_id + '.pdb'
#     refine_fn = run_fastrelax if use_fastrelax else run_pack
#     refine_fn(testPose, posi, amino, scorefxn, repack_pos)
    
#     if partners is not None:
#         local_docking(testPose, partners) if not use_fastrelax else ...
    
#     testPose.dump_pdb(os.path.join(output_path, pdb_files, pdbname))
#     mt_bound = scorefxn(testPose)
    
#     if partners is not None:
#         unbind(testPose, partners)
        
#     mt_unbound = scorefxn(testPose)
#     mt_binding = mt_unbound - mt_bound
#     testPose.assign(pose)

#     if (wt == amino):
#         wt_unbound = mt_unbound
#         wt_bound = mt_bound
#         wt_binding = mt_binding
#     else:
#         refine_fn(testPose, posi, wt, scorefxn, repack_pos)
#         if partners is not None:
#             local_docking(testPose, partners) if not use_fastrelax else ...
                    
#         wt_bound = scorefxn(testPose)
#         if partners is not None:
#             unbind(testPose, partners)
#         wt_unbound = scorefxn(testPose)
#         wt_binding = wt_unbound - wt_bound
#         testPose.assign(pose)

    
#     content=(str(mutation_id) + '\t' + str(posi) + '\t' + wt + '\t' + amino + '\t'
#              + str(wt_unbound) + '\t' + str(wt_bound) + '\t' + str(wt_binding) + '\t' + str(mt_unbound) + '\t' + str(mt_bound) + '\t'
#              + str(mt_binding) + '\t'+ str(mt_unbound/wt_unbound) + '\t' + str(mt_binding/wt_binding) + '\n')

#     return content

def mutate(pose, posi, amino, repack_pos, partners, name, output_path, use_fastrelax,
           compute_metrics=True):
    """Apply a single point mutation, refine, and optionally score metrics.

    Args:
        pose: Input complex pose.
        posi: Residue index to mutate.
        amino: Target one-letter amino acid code.
        repack_pos: List of residue indices allowed to repack/relax.
        partners: Docking partner string for unbinding (or None).
        name: Base name used for output PDB filename.
        output_path: Directory to write the mutant PDB into.
        use_fastrelax: If True use FastRelax, else PackRotamers.
        compute_metrics: If True compute wt/mt bound/unbound/binding scores.

    Return:
        TSV-formatted line of metrics if compute_metrics else empty string.
    """
    def score_bound_unbound(p):
        """Return (bound, unbound, binding) scores. If no partners, all equal."""
        bound = scorefxn(p)
        if partners is not None:
            unbind(p, partners)
        unbound = scorefxn(p)
        return bound, unbound, unbound - bound

    testPose = Pose()
    testPose.assign(pose)
    scorefxn = get_fa_scorefxn()

    wt = wildtype(str(pose.aa(posi)))
    mutation_id = f'{wt}{posi}{amino}'
    pdbname = f'{name}_{mutation_id}.pdb'

    refine_fn = run_fastrelax if use_fastrelax else run_pack
    refine_fn(testPose, posi, amino, scorefxn, repack_pos)

    if partners is not None and not use_fastrelax:
        local_docking(testPose, partners)

    testPose.dump_pdb(os.path.join(output_path, 'pdb_files', pdbname))

    if not compute_metrics:
        return ""

    mt_bound, mt_unbound, mt_binding = score_bound_unbound(testPose)

    if wt == amino:
        wt_bound, wt_unbound, wt_binding = mt_bound, mt_unbound, mt_binding
    else:
        testPose.assign(pose)
        refine_fn(testPose, posi, wt, scorefxn, repack_pos)
        if partners is not None and not use_fastrelax:
            local_docking(testPose, partners)
        wt_bound, wt_unbound, wt_binding = score_bound_unbound(testPose)

    content = (f"{mutation_id}\t{posi}\t{wt}\t{amino}\t"
               f"{wt_unbound}\t{wt_bound}\t{wt_binding}\t"
               f"{mt_unbound}\t{mt_bound}\t{mt_binding}\t"
               f"{mt_unbound/wt_unbound}\t{mt_binding/wt_binding if wt_binding != 0 else 'N/A'}\n")
    return content


def get_ligand_coords(pose):
    ligand_atoms = list()

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_ligand():
            for a in range(1, res.natoms() + 1):
                if res.atom_type(a).is_heavyatom():
                    ligand_atoms.append(res.xyz(a))

    return ligand_atoms
    
def min_distance(atom_xyz, ligand_coords):
    if atom_xyz is None:
        return None
    else:
        return min((atom_xyz - l).norm() for l in ligand_coords)
    
def determine_design_repack_residues(
    pose,
    cut1 = 6.0,
    cut2 = 8.0,
    cut3 = 10.0,
    cut4 = 12.0
):
    ligand_atoms = get_ligand_coords(pose)
    designable = list()
    repackable = list()
    
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue

        ca_xyz = res.xyz("CA")
        d_ca = min_distance(ca_xyz, ligand_atoms)

        has_cb = res.has("CB")
        d_cb = None
        if has_cb:
            cb_xyz = res.xyz("CB")
            d_cb = min_distance(cb_xyz, ligand_atoms)

        # Design region
        if d_ca <= cut1:
            designable.append(i)
        elif d_ca <= cut2 and has_cb and d_cb < d_ca:
            designable.append(i)

        # Repack region
        if d_ca <= cut3:
            repackable.append(i)
        elif d_ca <= cut4 and has_cb and d_cb < d_ca:
            repackable.append(i)

    return designable, repackable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_pdb", type=str, required=True)
    parser.add_argument("-o", "--output_name", type=str, default='outputs/result')
    parser.add_argument("-m", "--mode", type=str, default='auto')
    parser.add_argument("-pt", "--partners", type=str, default=None)
    parser.add_argument("-pr", "--params", type=str, default=None)
    parser.add_argument("-mp", "--mutation_positions", type=str, default=None)
    parser.add_argument("-rp", "--repack_positions", type=str, default='local')
    parser.add_argument("-f", "--use_fastrelax", action='store_true', default=False)
    parser.add_argument("--compute_metrics", action='store_true', default=False,
                        help="If set, compute energies per mutation and write TSV.")

    args = parser.parse_args()
    os.makedirs(args.output_name, exist_ok=True)
    os.makedirs(os.path.join(args.output_name, 'pdb_files'), exist_ok=True)

    tsv_file = None
    if args.compute_metrics:
        tsv_file = os.path.join(args.output_name, 'mutation_output.tsv')
        with open(tsv_file, 'w') as f:
            f.write("mutation_id\tposition\twildtype\tmutation\twt_unbound\twt_bound\twt_binding\tmt_unbound\tmt_bound\tmt_binding\tunbound_ratio\tbinding_ratio\n")

    pose = Pose()
    if args.params:
        load_ligand_params_to_pose(args.params.split(','), pose)
    pyrosetta.io.pose_from_file(pose, args.input_pdb)

    if args.mode == 'auto':
        mutation_positions, repack_positions = determine_design_repack_residues(pose)
    elif args.mode == 'manual' and args.mutation_positions is not None:
        mutation_positions = [int(s) for s in args.mutation_positions.split(',')]
        repack_positions = [int(s) for s in args.repack_positions.split(',')]

    print(mutation_positions, repack_positions)
    if not os.getenv("DEBUG"):
        for posi in mutation_positions:
            wt = wildtype(str(pose.aa(int(posi))))
            print("\nMutating Position: ", str(posi), " (wildtype:", wt, ")\n")
            for amino in one_letter_aa:
                if amino == wt:
                    continue
                content = mutate(pose, int(posi), amino, repack_positions, args.partners,
                                 args.input_pdb.split('/')[-1].split('.')[0],
                                 args.output_name, args.use_fastrelax,
                                 args.compute_metrics)
                if args.compute_metrics:
                    with open(tsv_file, 'a') as f:
                        f.write(content)


if __name__ == '__main__':
    main()
