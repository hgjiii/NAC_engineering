import sys, os
import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
import matplotlib.pyplot as plt
DIR = os.getcwd()
sys.path.append(f"{DIR}/PLACER")
import PLACER

## Initializing PLACER model with default checkpoint
placer = PLACER.PLACER()

def run_placer(pdb_path: str, lig_path: dict, output_dir: str):
    
    assert isinstance(lig_path, dict), 'The argument "lig_path" must be a dictionary type and have the format of {"LIG": "path/to/ligand", ...}' 
    
    try:
        pl_inp = PLACER.PLACERinput()
        pl_inp.pdb(pdb_path)
        pl_inp.name(os.path.basename(pdb_path).replace(".pdb" if ".pdb" in pdb_path else ".cif", ""))
        pl_inp.ignore_ligand_hydrogens(True)
        pl_inp.ligand_reference(lig_path)
        outputs = placer.run(pl_inp, 50)
    
        os.makedirs(output_dir, exist_ok=True)
        PLACER.protocol.dump_output(outputs, f"{output_dir}/{pl_inp.name()}", rerank="prmsd")
        
    except AssertionError as e:
        print("AssertionError: ", e)
        pass

def main():
    folder_path = os.path.join(os.path.dirname(os.getcwd()), 'inputs', 'placer', 'macarA_holo_apa_amp')
    pdb_names = [f for f in os.listdir(folder_path) if f.endswith(".pdb")]
    
    subs_path = os.path.join(os.path.dirname(os.getcwd()), 'inputs', 'placer', 'subs')
    lig_path = {'LIG': os.path.join(subs_path, 'apa_amp.mol2')}
    
    output_dir = os.path.join(os.path.dirname(os.getcwd()), 'outputs', 'placer', 'macarA_holo_apa_amp')
    
    for name in pdb_names:
        pdb_path = os.path.join(folder_path, name)
        run_placer(pdb_path, lig_path, output_dir)

if __name__ == '__main__':
    main()