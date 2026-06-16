import os
import argparse

import pyrosettacolabsetup; pyrosettacolabsetup.install_pyrosetta()
import pyrosetta

from pyrosetta import * 

from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.toolbox import cleanATOM


def load_ligand_params(params):
    if params:
        flags = f"""
        -extra_res_fa {params}
        """
        # -ignore_unrecognized_res 1
        # -mute all
    else:
        flags = ""
    
    pyrosetta.init(flags)


def get_relax_pdb(input_pdb, output_dir):
    '''Step 0. load_pdb'''
    relax_pdb = f"{input_pdb.rstrip('.pdb')}.relax.pdb"
    
    # init()
    # pose = pose_from_pdb(clean_pdb)
    pose = pose_from_pdb(input_pdb)
    testPose = Pose()
    testPose.assign(pose)
    print(testPose)
    
    # relax = FastRelax()
    # # # scorefxn = get_score_function()
    # scorefxn = get_fa_scorefxn()
    # relax.set_scorefxn(scorefxn)
    
    # relax.constrain_relax_to_start_coords(True)
    
    # relax.max_iter(100)
    # print(relax)
    
    # if not os.getenv("DEBUG"):
    #     relax.apply(pose)
    # pose.dump_pdb(relax_pdb)

    # relax = FastRelax()
    # scorefxn = get_score_function()
    relax = rosetta.protocols.relax.FastRelax()
    scorefxn = get_fa_scorefxn()
    relax.set_scorefxn(scorefxn)
    relax.constrain_relax_to_start_coords(True)
    
    print(relax)
    relax.apply(testPose)        
    testPose.dump_pdb(relax_pdb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_pdb", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default='./outputs')
    parser.add_argument("-p", "--params", type=str, default=None)
    args = parser.parse_args()
    
    if args.params:
        params = args.params.split(',')
        for p in params:
            load_ligand_params(p)

    else:
        load_ligand_params(None)
    # load_ligand_params(args.params)
    get_relax_pdb(args.input_pdb, args.output_dir)

if __name__ == '__main__':
    main()