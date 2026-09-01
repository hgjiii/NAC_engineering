import sys, os
import warnings
warnings.filterwarnings("ignore")
import time
import glob
import itertools
import argparse
import numpy as np
import pandas as pd
import itertools
import torch
import json
from openbabel import openbabel
openbabel.obErrorLog.SetOutputLevel(0)

DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)
import PLACER



def main(args):
    ########################################################
    # 0. load the network
    ########################################################
    DIR = os.path.dirname(__file__)
    weightsfile = args.weights

    placer = PLACER.PLACER(weightsfile)


    ########################################################
    # 1. setup
    ########################################################

    if args.idir is not None:
        fnames = glob.glob(args.idir+'/*.pdb')
        if len(fnames)<1:
            sys.exit("Error: no .pdb files found in '%s'"%(args.idir))

    elif args.ifile is not None:
        if any([args.ifile.endswith(ext) for ext in ['.pdb', '.ent', '.cif', '.cif.gz']]):
            fnames = [args.ifile]
        else:
            with open(args.ifile) as f:
                fnames = [line.strip() for line in f.readlines()]
            if len(fnames)<1:
                sys.exit("Error: no .pdb files found in '%s'"%(args.ifile))

    print("# number of PDB files to process: %d"%(len(fnames)))


    ######################################################################
    # 1. Parsing user input arguments and adding them to the input object
    ######################################################################
    placer_input = PLACER.PLACERinput()

    if args.exclude_common_ligands is True:
        placer_input.skip_ligands(PLACER.utils.get_common_ligands())

    if args.ligand_file is not None:
        ligand_ref = {}
        for lr in args.ligand_file:
            ligand_ref[lr.split(":")[0]] = lr.split(":")[1]
        placer_input.ligand_reference(ligand_ref)

    if args.ignore_ligand_hydrogens is True:
        placer_input.ignore_ligand_hydrogens(True)

    if args.use_sm is False:
        placer_input.exclude_sm(True)
    
    if args.fixed_ligand_noise is not None:
        placer_input.fixed_ligand_noise(args.fixed_ligand_noise)
    
    def evaluate_pred_fix_ligand_input(ligands):
        fixed_ligands = []
        for lig in ligands:
            if "-" in lig:
                if len(lig.split("-")) == 2:
                    fixed_ligands.append((lig.split("-")[0], int(lig.split("-")[1])))
                elif len(lig.split("-")) == 3:
                    fixed_ligands.append((lig.split("-")[0], lig.split("-")[1], int(lig.split("-")[2])))
                else:
                    sys.exit(f"Invalid fixed/predict ligand input: {lig}")
            else:
                fixed_ligands.append(lig)
        return fixed_ligands

    if args.fixed_ligand is not None:
        placer_input.fixed_ligand(evaluate_pred_fix_ligand_input(args.fixed_ligand))

    if args.predict_ligand is not None:
        placer_input.predict_ligand(evaluate_pred_fix_ligand_input(args.predict_ligand))

    if args.predict_multi is True:
        placer_input.predict_multi(True)

    if args.target_res is not None:
        target_res = args.target_res.split("-")
        if len(target_res) == 2:
            placer_input.target_res((target_res[0], int(target_res[1])))
        elif len(target_res) == 3:
            placer_input.target_res((target_res[0], int(target_res[1]), target_res[2]))

    if args.bonds is not None:
        # user input: "A-42-ALA-CB:B-173-JRP-CL:bondlen"
        # API input: [(ch, resno, name3, atomname), (ch, resno, name3, atomname), bondlen]
        bonds = []
        for bond in args.bonds:
            a,b,bondlen = bond.split(':')
            a,b = a.split('-'),b.split('-')
            aname = (a[0],int(a[1]),a[2],a[3])
            bname = (b[0],int(b[1]),b[2],b[3])
            bonds.append([aname, bname, float(bondlen)])
        placer_input.bonds(bonds)

    if args.mutate is not None:
        # user input: 5A:TRP,6A:GLY
        # API input: {("A", 5): "TRP", ("A", 6): "GLY"}
        mutate_dict = {}
        for mutres in args.mutate:
            pos,resn = mutres.split(':')
            resno = ""
            n = 0
            while pos[n].isnumeric():
                resno += pos[n]
                n += 1
            chain = pos[n:]
            mutate_dict[(chain, int(resno))] = resn
        placer_input.mutate(mutate_dict)

    if args.residue_json is not None:
        placer_input.add_custom_residues(json.load(open(args.residue_json)))

    if args.crop_centers is not None:
        # Only accepting atom names from commandline.
        # Use API to provide coordinates.
        _centers = []
        for cntr in args.crop_centers:
            _cntr = cntr.split("-")
            assert len(_cntr) == 4
            _centers.append((_cntr[0], int(_cntr[1]), _cntr[2], _cntr[3]))
        placer_input.crop_centers(_centers)

    if args.corruption_centers is not None:
        # Only accepting atom names from commandline.
        # Use API to provide coordinates.
        _centers = []
        for cntr in args.corruption_centers:
            _cntr = cntr.split("-")
            assert len(_cntr) == 4
            _centers.append((_cntr[0], int(_cntr[1]), _cntr[2], _cntr[3]))
        placer_input.corruption_centers(_centers)


    ########################################################
    # 2. generate models
    ########################################################
    tic = time.time()

    for counter,fname in enumerate(fnames):

        if ".cif.gz" in os.path.basename(fname):
            label = os.path.basename(fname).replace(".cif.gz", "")
        elif ".cif" in os.path.basename(fname):
            label = os.path.basename(fname).replace(".cif", "")
        elif ".pdb" in os.path.basename(fname):
            label = os.path.basename(fname).replace(".pdb", "")
        if args.suffix is not None:
            label += f"_{args.suffix}"

        outfile_prefix = args.odir + "/" + label

        # ocsv = args.ocsv
        # if ocsv is None:
        #     ocsv = args.odir+'/'+label+'.csv'

        # If output exists then skipping this task
        if args.cautious is True:
            if os.path.exists(outfile_prefix+".csv"):
                print(f"{outfile_prefix}.csv already exists, skipping this prediction.")
                continue

        placer_input_iter = placer_input.copy()

        placer_input_iter.name(label)

        if fname.endswith(".pdb"):
            placer_input_iter.pdb(fname)
        elif fname.endswith(".cif") or fname.endswith(".cif.gz"):
            placer_input_iter.cif(fname)


        # execute PLACER
        outputs = placer.run(placer_input_iter, args.nsamples)

        # Rank the outputs based on a user-defined metric
        # if args.rerank is not None:
        #     outputs = PLACER.utils.rank_outputs(outputs, args.rerank)


        ### Save outputs to disk ###
        os.makedirs(args.odir, exist_ok=True)

        # Dump outputs
        PLACER.protocol.dump_output(output_dict=outputs, filename=outfile_prefix, rerank=args.rerank)

        # save scores to CSV
        # ignore_csv_keys = ["item", "model", "center", "Xs", "Ds", "plDDTs", "pDEVs"]
        # df = pd.DataFrame.from_dict({k: [outputs[n][k] for n in outputs] for k in outputs[0].keys() if k not in ignore_csv_keys})
        # df.to_csv(ocsv,index=False, mode='a', header=not os.path.exists(ocsv))
        # print(f"Wrote scores to {ocsv}")

        # # save models as a multimodel PDB
        # opdb = args.odir+'/'+label+'_model.pdb'
        # f = open(opdb,'w')
        # for n in outputs:
        #     for l in outputs[n]["model"]:
        #         f.write(l)
        # f.close()
        # print(f"Wrote generated models to {opdb}")

    print(f"Finished predicting {len(fnames)} structures in {(time.time() - tic):.2f} seconds.")


if __name__ == "__main__":

    rank_options = [
        'prmsd',
        'plddt',
        'plddt_pde'
    ]

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-i','--idir', type=str, required=False, help='input folder with PDB/mmCIF files')
    argparser.add_argument('-f','--ifile', type=str, required=False, help='file with a list of input PDB/mmCIF files or a single PDB/mmCIF file. Only mmCIF files from RCSB are correctly parsed.')
    argparser.add_argument('-o','--odir', type=str, required=False, default="./", help='output folder to save models and CSV files. Default is current run directory.')
    argparser.add_argument('-n','--nsamples', type=int, default=10, help='number of samples to generate. 50-100 is a good number in most cases.')
    argparser.add_argument('--ocsv', type=str, required=False, help='output .csv file to save scores. By default the CSV name is inferred from the input file name, with --suffix input added.')
    argparser.add_argument('--suffix', type=str, required=False, help='suffix added to output PDB file')
    argparser.add_argument('--cautious', action="store_true", default=False, help='Cautious mode. If output CSV exists, then will not run that prediction again.')
    argparser.add_argument('--exclude_common_ligands', action="store_true", default=False, help='All common solvents and crystallography additivies will be excluded from the prediction. '
                                                                                                'List of residues was obtained from AlphaFold3 supplementary data (DOI: 10.1038/s41586-024-07487-w). '
                                                                                                'Useful when predicting directly any crystal structures.')
    argparser.add_argument('--predict_multi', action="store_true", default=False, help='All allowed ligands in input will be predicted and scored. fixed_ligand and predict_ligand inputs are respected.')
    argparser.add_argument('--fixed_ligand', type=str, nargs="+", required=False, help='Ligand <name3> or <name3-resno> or <chain-name3-resno> that will remain fixed.')
    argparser.add_argument('--predict_ligand', type=str, nargs="+", required=False, help='Ligand <name3> or <name3-resno> or <chain-name3-resno> that will be predicted. All other ligands will be fixed.')
    argparser.add_argument('--target_res', type=str, required=False, help='Protein residue <chain-resno> or <chain-name3-resno> that will be used as crop center. Required when input has no ligands.')
    argparser.add_argument('--fixed_ligand_noise', type=float, required=False, help='Noise added to fixed ligand coordinates. Default is the same as backbone atom `sigma_bb` in the model params.')
    argparser.add_argument('--weights', type=str, required=False, default=f"{DIR}/weights/PLACER_model_1.pt", help=f'Weights file (pytorch .pt file).')

    argparser.add_argument('--rerank', type=str, required=False, choices=rank_options, help='Output CSV and PDB models files are ranked from best to worst based on one of the input metrics: prmsd, plddt, plddt_pde. '
                           'Prmsd is sorted in ascending order; plddt and plddt_pde in descending order. The model numbers that are printed to screen while the script runs no longer apply.')

    argparser.add_argument('--bonds', type=str, required=False, nargs="+", help='put a bond between two atoms, e.g. "A-42-ALA-CB:B-173-JRP-CL:<bondlen>", as space-separated list')
    argparser.add_argument('--mutate', type=str, required=False, nargs="+", help='mutate certain positions, e.g. "5A:TRP" or "5A:TRP 6A:GLY"')
    argparser.add_argument('--crop_centers', type=str, required=False, nargs="+", help='Atom names that will be used as CROP centers. This centers the crop to a particular part of the pocket, '
                                                                                'but the ligands are still corrupted based on their input coordinates. Used for refining where the cropped sphere is. '
                                                                                'This DOES NOT affect which atoms/ligands are selected for prediction. Use --predict_ligand ... for that. '
                                                                                'One atom will be picked randomly from the provided set. '
                                                                                'XYZ coordinate input available in the API. Example: "B-200-HEM-FE B-200-HEM-O1"')
    argparser.add_argument('--corruption_centers', type=str, required=False, nargs="+", help='Atom names that will be used as corruption centers. Allows sampling the ligand around in the whole protein. '
                                                                                'One will be picked randomly from the provided set. Must provide at least as many centers as there are ligands in the input. '
                                                                                'XYZ coordinate input available in the API. Example: "B-200-HEM-FE B-200-HEM-O1"')
    argparser.add_argument('--residue_json', type=str, required=False, help='JSON file that specifies any custom residues used in the PDB, or used with --mutate. These are added to the internal CCD library.\n'
                           "JSON format:\n{name3: {'sdf': <contents of SDF file as string>,\n"
                           "'atom_id': [atom names],\n"
                           "'leaving': [True/False for whether this atom is deleted when part of polymer],\n"
                           "'pdbx_align': [empty list]}}")
    argparser.add_argument('--ligand_file', type=str, nargs="+", help='SDF or MOL2 file of the ligand(s). (Input format: XXX:ligand1.sdf YYY:ligand2.mol2) ZZZ:CCD\n '
                                                                  'Used for refining the atom typing and connectivity in the ligand structures. '
                                                                  'Coordinates are still parsed form the input PDB/mmCIF. If ligand exists in CCD then ZZZ:CCD is a special input that enables reading the ligand in from an internal CCD ligands database.')
    argparser.add_argument('--ignore_ligand_hydrogens', action='store_true', default=False, help='Affects --ligand_file. Ignores hydrogen atoms that are defined in the PDB and SDF/MOL2 files, and will not throw errors if the protonation states are different. Hydrogen atoms are not predicted with PLACER anyway.')
    argparser.add_argument('--use_sm', action='store_true',default=True, help='make predictions with the small molecule (holo - turned on by default)')
    argparser.add_argument('--no-use_sm', dest='use_sm', action='store_false', default=False,help='make predictions w/o the small molecule (apo)')
    argparser.set_defaults(use_sm=True)
    args = argparser.parse_args()
    if args.idir is None and args.ifile is None:
        sys.exit('Error: One of -i/--idir or -f/--ifile must be provided.')

    main(args)
