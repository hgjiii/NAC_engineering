#!/bin/bash

# predicting a P450 in complex with an inhibitor. No target ligand is specified, so a random ligand is picked at each iteration, making the predictions meaningless.
python ../run_PLACER.py --ifile inputs/4dtz.cif --odir example_out_CLI --rerank prmsd --suffix meaningless -n 10

# predicting the binding of an inhibitor in a P450 pocket, while keeping heme fixed (heme is fixed automatically if it's not in --predict_ligand input)
python ../run_PLACER.py --ifile inputs/4dtz.cif --odir example_out_CLI --rerank prmsd --suffix D-LDP-501 -n 10 --predict_ligand D-LDP-501

# predicting the binding of an inhibitor and heme in a P450 pocket, docking and scoring two ligands simultaneously
python ../run_PLACER.py --ifile inputs/4dtz.cif --odir example_out_CLI --rerank prmsd --suffix LDP-HEM -n 10 --predict_ligand D-LDP-501 C-HEM-500 --predict_multi

# predicting heme in denovo protein
python ../run_PLACER.py --ifile inputs/dnHEM1.pdb --odir example_out_CLI --rerank prmsd -n 10 --ligand_file HEM:ligands/HEM.mol2

# predicting sidechains with fixed heme, defining crop center to a residue
python ../run_PLACER.py --ifile inputs/dnHEM1.pdb --odir example_out_CLI --suffix A149_fixHEM -n 10 --ligand_file HEM:ligands/HEM.mol2 --fixed_ligand HEM --target_res A-149

# predicting sidechains in apo denovo protein, defining crop center to a residue
python ../run_PLACER.py --ifile inputs/dnHEM1_apo.pdb --odir example_out_CLI --suffix A149 -n 10 --target_res A-149

# Mutating a residue to a non-canonical, loading that non-canonical into residue database from a JSON file. Existing ligand is omitted from prediction with --no-use_sm
python ../run_PLACER.py --ifile inputs/denovo_SER_hydrolase.pdb --odir example_out_CLI --suffix 75I -n 10 --mutate 128A:75I --residue_json ligands/75I.json --no-use_sm

