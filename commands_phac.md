### This is for running the process of NAC-preorganization evaluation ###

## PhaC1437 variants sequence screening ##

# Generate params file for the input ligands
python pipeline/molfile_to_params.py ligand.mol2 -o /path/to/output/dir -n LIG -p LIG

python pipeline/molfile_to_params.py laccoa.mol2 -o inputs/rosetta/subs -n LCA -p LIG


# Run FastRelax for structural refinement
python pipeline/run_fastrelax.py -i /path/to/input/dir -o /path/to/output/dir -pr /path/to/params/file -cm -pt A_YZ

python pipeline/run_fastrelax.py -i inputs/rosetta/carA_holo_adi_amp_homologs_modeller -o inputs/placer/carA_holo_adi_amp_homologs_modeller -pr inputs/rosetta/subs/ADI.params -cm -pt A_YX

# Run PLACER for complex structures in specific directory
python pipeline/run_placer.py -i /path/to/input/dir -o /path/to/output/dir -n 100 -l LIG:/path/to/lig/file.mol2 LIG: CCD

** APO state **
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_apo_homologs_modeller_100_1 -n 100
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_apo_homologs_modeller_100_2 -n 100

** HOLO state **
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_holo_adi_amp_homologs_modeller_100_1 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_holo_adi_amp_homologs_modeller_100_2 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_holo_adi_amp_homologs_modeller_100_3 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_holo_adi_amp_homologs_modeller_100_4 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_holo_adi_amp_homologs_modeller_100_5 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_holo_adi_amp_homologs_modeller_100_6 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_holo_adi_amp_homologs_modeller_100_7 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD
python pipeline/run_placer.py -i inputs/placer/carA_holo_adi_amp_homologs_modeller -o outputs/placer/carA_holo_adi_amp_homologs_modeller_100_8 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD


# Run NAC_PO_analysis based on the PLACER outputs
python pipeline/run_nac_analysis.py --holo_dir /path/to/holo/output/dir --apo_dir /path/to/apo/output/dir --catres /path/to/res_mapping/file.pkl --config /path/to/config/file.json --output /path/to/output/file.csv


python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_holo_adi_amp_homologs_modeller_100_1 --apo_dir outputs/placer/carA_apo_homologs_modeller_100_1 --catres homologs/car_catres_mapping.pkl --config config/carA_holo_adi_amp.json --output results/carA_holo_adi_amp_homologs_nac_candidates_modeller_1.csv
python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_holo_adi_amp_homologs_modeller_100_2 --apo_dir outputs/placer/carA_apo_homologs_modeller_100_1 --catres homologs/car_catres_mapping.pkl --config config/carA_holo_adi_amp.json --output results/carA_holo_adi_amp_homologs_nac_candidates_modeller_2.csv
python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_holo_adi_amp_homologs_modeller_100_3 --apo_dir outputs/placer/carA_apo_homologs_modeller_100_1 --catres homologs/car_catres_mapping.pkl --config config/carA_holo_adi_amp.json --output results/carA_holo_adi_amp_homologs_nac_candidates_modeller_3.csv
python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_holo_adi_amp_homologs_modeller_100_4 --apo_dir outputs/placer/carA_apo_homologs_modeller_100_1 --catres homologs/car_catres_mapping.pkl --config config/carA_holo_adi_amp.json --output results/carA_holo_adi_amp_homologs_nac_candidates_modeller_4.csv
python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_holo_adi_amp_homologs_modeller_100_5 --apo_dir outputs/placer/carA_apo_homologs_modeller_100_1 --catres homologs/car_catres_mapping.pkl --config config/carA_holo_adi_amp.json --output results/carA_holo_adi_amp_homologs_nac_candidates_modeller_5.csv
python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_holo_adi_amp_homologs_modeller_100_6 --apo_dir outputs/placer/carA_apo_homologs_modeller_100_1 --catres homologs/car_catres_mapping.pkl --config config/carA_holo_adi_amp.json --output results/carA_holo_adi_amp_homologs_nac_candidates_modeller_6.csv
python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_holo_adi_amp_homologs_modeller_100_7 --apo_dir outputs/placer/carA_apo_homologs_modeller_100_1 --catres homologs/car_catres_mapping.pkl --config config/carA_holo_adi_amp.json --output results/carA_holo_adi_amp_homologs_nac_candidates_modeller_7.csv
python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_holo_adi_amp_homologs_modeller_100_8 --apo_dir outputs/placer/carA_apo_homologs_modeller_100_1 --catres homologs/car_catres_mapping.pkl --config config/carA_holo_adi_amp.json --output results/carA_holo_adi_amp_homologs_nac_candidates_modeller_8.csv


# Run docking analysis for the final output
python pipeline/run_docking_analysis.py --docking_dir /path/to/docking/output/dir --ref_root /path/to/placer/output/dir --candidates /path/to/input/csv/path --output /path/to/output/csv/path --ligand_resname LIG


conda deactivate; conda activate vina;
python pipeline/run_docking_analysis.py --docking_dir /path/to/docking/output/dir --ref_root /path/to/placer/output/dir --candidates /path/to/input/csv/path --output /path/to/output/csv/path --ligand_resname LIG

python pipeline/run_docking_analysis.py --docking_dir outputs/docking/carA_holo_adi_amp_homologs_modeller_100_1 --ref_root outputs/placer/carA_holo_adi_amp_homologs_modeller_100_1 --candidates results/carA_holo_adi_amp_homologs_nac_candidates_modeller_1_filtered.csv --output results/20260818_carA_holo_6oha_amp_homologs_nac_po_ds_candidates_modeller_1.csv --ligand_resname ADI
python pipeline/run_docking_analysis.py --docking_dir outputs/docking/carA_holo_adi_amp_homologs_modeller_100_2 --ref_root outputs/placer/carA_holo_adi_amp_homologs_modeller_100_2 --candidates results/carA_holo_adi_amp_homologs_nac_candidates_modeller_2_filtered.csv --output results/20260818_carA_holo_6oha_amp_homologs_nac_po_ds_candidates_modeller_2.csv --ligand_resname ADI
python pipeline/run_docking_analysis.py --docking_dir outputs/docking/carA_holo_adi_amp_homologs_modeller_100_3 --ref_root outputs/placer/carA_holo_adi_amp_homologs_modeller_100_3 --candidates results/carA_holo_adi_amp_homologs_nac_candidates_modeller_3_filtered.csv --output results/20260818_carA_holo_6oha_amp_homologs_nac_po_ds_candidates_modeller_3.csv --ligand_resname ADI
python pipeline/run_docking_analysis.py --docking_dir outputs/docking/carA_holo_adi_amp_homologs_modeller_100_4 --ref_root outputs/placer/carA_holo_adi_amp_homologs_modeller_100_4 --candidates results/carA_holo_adi_amp_homologs_nac_candidates_modeller_4_filtered.csv --output results/20260818_carA_holo_6oha_amp_homologs_nac_po_ds_candidates_modeller_4.csv --ligand_resname ADI
python pipeline/run_docking_analysis.py --docking_dir outputs/docking/carA_holo_adi_amp_homologs_modeller_100_5 --ref_root outputs/placer/carA_holo_adi_amp_homologs_modeller_100_5 --candidates results/carA_holo_adi_amp_homologs_nac_candidates_modeller_5_filtered.csv --output results/20260818_carA_holo_6oha_amp_homologs_nac_po_ds_candidates_modeller_5.csv --ligand_resname ADI
python pipeline/run_docking_analysis.py --docking_dir outputs/docking/carA_holo_adi_amp_homologs_modeller_100_6 --ref_root outputs/placer/carA_holo_adi_amp_homologs_modeller_100_6 --candidates results/carA_holo_adi_amp_homologs_nac_candidates_modeller_6_filtered.csv --output results/20260818_carA_holo_6oha_amp_homologs_nac_po_ds_candidates_modeller_6.csv --ligand_resname ADI
python pipeline/run_docking_analysis.py --docking_dir outputs/docking/carA_holo_adi_amp_homologs_modeller_100_7 --ref_root outputs/placer/carA_holo_adi_amp_homologs_modeller_100_7 --candidates results/carA_holo_adi_amp_homologs_nac_candidates_modeller_7_filtered.csv --output results/20260818_carA_holo_6oha_amp_homologs_nac_po_ds_candidates_modeller_7.csv --ligand_resname ADI
python pipeline/run_docking_analysis.py --docking_dir outputs/docking/carA_holo_adi_amp_homologs_modeller_100_8 --ref_root outputs/placer/carA_holo_adi_amp_homologs_modeller_100_8 --candidates results/carA_holo_adi_amp_homologs_nac_candidates_modeller_8_filtered.csv --output results/20260818_carA_holo_6oha_amp_homologs_nac_po_ds_candidates_modeller_8.csv --ligand_resname ADI


## Mutant sequence screening ##

# Run FastRelax for refinement of the wild-type structure
python pipeline/run_fastrelax.py -i inputs/rosetta/carA_single_target_holo_adi_amp_swiss -o inputs/placer/carA_single_target_holo_adi_amp_swiss -pr inputs/rosetta/subs/ADI.params -cm -pt A_YX

# Run saturation mutagenesis for target PDB
python pipeline/run_mutation.py -i inputs/placer/carA_single_target_holo_adi_amp_swiss/A0AB38D5A4_swiss.relax.pdb -o inputs/placer/carA_A0AB38D5A4_holo_adi_amp_mutants_single -pr inputs/rosetta/subs/ADI.params

python pipeline/run_mutation.py -i inputs/placer/carA_single_target_holo_adi_amp_swiss/A0A6G9XT36_swiss.relax.pdb -o inputs/placer/carA_A0A6G9XT36_holo_adi_amp_mutants_single -pr inputs/rosetta/subs/ADI.params

# Run PLACER for complex structures in specific directory
python pipeline/run_placer.py -i inputs/placer/carA_A0AB38D5A4_holo_adi_amp_mutants_single -o outputs/placer/carA_A0AB38D5A4_holo_adi_amp_mutants_single_100 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD
python pipeline/run_placer.py -i inputs/placer/carA_A0A6G9XT36_holo_adi_amp_mutants_single -o outputs/placer/carA_A0A6G9XT36_holo_adi_amp_mutants_single_100 -n 100 -l ADI:inputs/rosetta/subs/adi.mol2 AMP:CCD

python pipeline/run_placer.py -i inputs/placer/carA_A0AB38D5A4_holo_adi_amp_mutants_single -o outputs/placer/carA_A0AB38D5A4_apo_mutants_single_100 -n 100
python pipeline/run_placer.py -i inputs/placer/carA_A0A6G9XT36_holo_adi_amp_mutants_single -o outputs/placer/carA_A0A6G9XT36_apo_mutants_single_100 -n 100

# Run NAC_PO_analysis based on the PLACER outputs
python pipeline/run_nac_analysis.py --holo_dir /path/to/holo/output/dir --apo_dir /path/to/apo/output/dir --catres /path/to/res_mapping/file.pkl --config /path/to/confir/file.json --output /path/to/output/file.csv

python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_A0AB38D5A4_holo_adi_amp_mutants_single_100 --apo_dir outputs/placer/carA_A0AB38D5A4_apo_mutants_single_100 --config config/carA_A0AB38D5A4_holo_adi_amp_C.json --output results/test_2.csv
python pipeline/run_nac_analysis.py --holo_dir outputs/placer/carA_A0A6G9XT36_holo_adi_amp_mutants_single_100 --apo_dir outputs/placer/carA_A0A6G9XT36_apo_mutants_single_100 --config config/carA_A0A6G9XT36_holo_adi_amp_C.json --output results/test.csv


# Result descriptions:
1. carA-ADI-AMP complex conformation results: 
    - holo: carA_holo_adi_amp_homologs_100_2
    - apo: carA_apo_homologs_100
    - config: carA_holo_adi_amp_C.json
    - docking: carA_holo_adi_amp_homologs_2C
    - results_initial: carA_holo_adi_amp_homologs_nac_candidates_2C.csv
    - results_filtered: carA_holo_adi_amp_homologs_nac_candidates_2C_filtered.csv
    - results_final: 20260619_carA_holo_adi_amp_homologs_po_ds_candidates_2C.csv

2. carA-6ACA-AMP complex conformation results (6ACA == 6AC):
    - holo: carA_holo_6aca_amp_homologs_100
    - apo: carA_apo_homologs_100
    - config: carA_holo_6aca_amp_C.json
    - docking: carA_holo_6aca_amp_homologs
    - results_initial: carA_holo_6aca_amp_homologs_nac_candidates.csv
    - results_filtered: carA_holo_6aca_amp_homologs_nac_candidates_filtered.csv
    - results_final: 20260715_carA_holo_6aca_amp_homologs_nac_po_ds_candidates.csv

3. carA-6OHA-AMP complex conformation results (6OHA == 6OA):
    - holo: carA_holo_6oha_amp_homologs_100
    - apo: carA_apo_homologs_100
    - config: carA_holo_6oha_amp_C.json
    - docking: carA_holo_6oha_amp_homologs
    - results_initial: carA_holo_6oha_amp_homologs_nac_candidates.csv
    - results_filtered: carA_holo_6oha_amp_homologs_nac_candidates_filtered.csv
    - results_final: 20260715_carA_holo_6oha_amp_homologs_nac_po_ds_candidates.csv



## etc.
<PLACER CPU usage control>
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0

