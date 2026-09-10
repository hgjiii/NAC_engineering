import sys, os
import time
import warnings
warnings.filterwarnings("ignore")
import argparse

# DIR = os.getcwd()
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f"{DIR}/PLACER")
import PLACER

## Initializing PLACER model with default checkpoint
placer = PLACER.PLACER()

def run_placer(pdb_path: str, lig_path: dict, output_dir: str, iterations: int = 50):
    
    assert isinstance(lig_path, dict) or lig_path == None, 'The argument "lig_path" must be a dictionary type and have the format of {"LIG": "path/to/ligand", ...} (or be a None type)' 
    
    try:
        pl_inp = PLACER.PLACERinput()
        pl_inp.pdb(pdb_path)
        pl_inp.name(os.path.basename(pdb_path).replace(".pdb" if ".pdb" in pdb_path else ".cif", ""))
        # pl_inp.ligand_reference({"AMP": amp_name, "SIN": sucfile})
        if lig_path is None:
            pl_inp.exclude_sm(True)
        else:
            pl_inp.ignore_ligand_hydrogens(True)
            pl_inp.ligand_reference(lig_path)
            
        outputs = placer.run(pl_inp, iterations)
    
        os.makedirs(output_dir, exist_ok=True)
        PLACER.protocol.dump_output(outputs, f"{output_dir}/{pl_inp.name()}", rerank="prmsd")
        
    except AssertionError as e:
        print("AssertionError: ", e)
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir",  type=str, required=True,
                        help="Directory containing input PDB files")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save PLACER outputs")
    parser.add_argument("-n", "--iterations", type=int, default=50,
                        help="Number of iterations for PLACER (default: 50)")
    parser.add_argument("-l", "--lig_path", type=str, default=None, nargs="+",
                        help="Ligand path(s) in format 'LIG:path/to/ligand.mol2' (optional, multiple allowed)")
    args = parser.parse_args()


    # parse lig_path: ["LIG:path/to/lig1.mol2", "AMP:path/to/lig2.mol2"] → {"LIG": "...", "AMP": "..."}
    if args.lig_path:
        lig_path = {k.strip(): v.strip() for item in args.lig_path for k, v in [item.split(":")]}
    else:
        lig_path = None

    pdb_names = [f for f in os.listdir(args.input_dir) if f.endswith(".pdb")]
    total = len(pdb_names)
    start_time = time.time()

    for idx, name in enumerate(pdb_names, start=1):
        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] START: {name}")
        print(f"{'#' * 70}")

        cycle_start = time.time()
        pdb_path = os.path.join(args.input_dir, name)
        run_placer(pdb_path, lig_path, args.output_dir, iterations=args.iterations)

        elapsed = time.time() - start_time
        cycle_time = time.time() - cycle_start
        avg = elapsed / idx
        eta = avg * (total - idx)
        print(f"\n{'#' * 70}")
        print(f"# [{idx}/{total}] DONE: {name}")
        print(f"# This file: {cycle_time / 60:.1f} min  "
              f"| Elapsed: {elapsed / 60:.1f} min  | Remaining: {eta / 60:.1f} min")
        print(f"{'#' * 70}")
        
if __name__ == '__main__':
    main()