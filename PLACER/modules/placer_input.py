import sys, os
import json
import time
import tempfile
import numpy as np
import pandas as pd
import copy


class PLACERinput():
    def __init__(self):
        ## Storage of user inputs
        self.__skip_ligands = []
        self.__bonds = None
        self.__fixed_ligand = None
        self.__fixed_ligand_noise = None  # default for this is set in PLACER instance
        self.__predict_ligand = None
        self.__ligand_reference = None
        self.__target_res = None
        self.__corruption_centers = None
        self.__crop_centers = None
        self.__ignore_hydrogens = False
        
        self.__pdb = None
        self.__cif_file = None
        self.__name = None

        self.__exclude_sm = False
        self.__mutate_dict = None
        self.__predict_multi = False
        self.__custom_entities = {}


    ### SETTERS / GETTERS ###
    def skip_ligands(self, ligands=None):
        """
        Ligand names (name3) that will be skipped during input structure parsing.
        Consider including common crystallography ligands such as SO4, PEG etc...

        Parameters
        ----------
        ligands : list, optional
            Ligand name3's'

        Returns
        -------
        list
            Returns list of stored ligand names if no input provided.
        """
        if ligands is None:
            return self.__skip_ligands
        else:
            self.__skip_ligands = ligands        
    
    def fixed_ligand(self, ligands=None):
        """
        # Ligands can be defined in 3 ways:
        # name3 - all ligands with this name will be fixed/predicted
        # (name3, resno) - ligands with this name and residue number will be fixed/predicted.
        #                  If there are multiple chains with the same residue numbering then all copies of the ligand will be fixed/predicted.
        # (chain, name3, resno) - ligand in this chain with this name and residue number will be fixed/predicted.

        Parameters
        ----------
        ligands : list, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        if no input, returns the stored fixed_ligands
        """
        if ligands is None:
            return self.__fixed_ligand
        else:
            self.__fixed_ligand = ligands

    def fixed_ligand_noise(self, noise=None):
        """
        Noise level that is used on the atoms of the "fixed_ligand" input.
        Default is the same as backbone atom noise, defined in the PLACER model (0.1).

        Parameters
        ----------
        noise : float, optional
            Noise level.

        """
        if noise is None:
            return self.__fixed_ligand_noise
        else:
            assert isinstance(noise, float)
            self.__fixed_ligand_noise = noise


    def bonds(self, bonds=None):
        """
        Any additional bonds that should be created between defined atoms in the system.
        Bonds are added as new edges in the graph.
        Input is a list where each items is a list: [ [(ch, resno, name3, atomname), (ch, resno, name3, atomname), bondlen], ]

        Parameters
        ----------
        bonds : list, optional
            List of atompairs that should be bonded to each other, and the bond length.
        """
        if bonds is None:
            return self.__bonds
        else:
            assert isinstance(bonds, list)
            for atompair in bonds:
                assert len(atompair) == 3
                assert isinstance(atompair[0], tuple)
                assert isinstance(atompair[1], tuple)
                assert isinstance(atompair[2], float)
            self.__bonds = bonds


    def predict_ligand(self, ligands=None):
        """
        Sets particular ligand(s) as to-be-predicted. All other ligands are treated as "fixed".
        If there are multiple ligands selected for "predict_ligand", then "predict_multi()" needs to be enabled to have all of them to be scored and used as corruption centers.
        If the two ligands are too far apart then selecting an appropriate crop can become difficult and the results may become meaningless.
        Ligands can be defined in 3 ways:
            (chain, name3, resno) - ligand in this chain with this name and residue number will be fixed/predicted. The safest option.
            (name3, resno) - ligands with this name and residue number will be fixed/predicted.
                           If there are multiple chains with the same residue numbering then all copies of the ligand will be fixed/predicted, if they end up on the same crop.
            name3 - all ligands with this name will be fixed/predicted.
            chain - all ligands under this chain letter will be predicted.


        Parameters
        ----------
        ligands : list, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        if no input, returns the stored predict_ligand
        """
        if ligands is None:
            return self.__predict_ligand
        else:
            self.__predict_ligand = ligands

    def ligand_reference(self, ligand_reference=None):
        """
        Provide reference SDF or MOL2 files (or their contents) for any residues and ligands.
        This can greatly improve PLACER performance in small molecule structure prediction from PDB inputs.
        SDF/MOL2 files are used to provide additional information for atom type, bonding and charge features.
        Input is a dictionary where keys are 3-letter ligand names, and values are paths to SDF/MOL2 files, or the contents of those files.

        The string 'PDB' is a special dictionary value that defines that the reference SDF information should be parsed from an internally stored native ligand database.

        Parameters
        ----------
        ligand_reference : dict, optional
            Dictionary where keys are 3-letter ligand names, and values are paths to SDF/MOL2 files, or the contents of those files, or the string 'PDB' (see above).
            {"LIG": "/path/to/LIG.sdf", "ATP": "PDB"}
            The default is None.

        """
        if ligand_reference is None:
            return self.__ligand_reference
        else:
            self.__ligand_reference = ligand_reference

    def ignore_ligand_hydrogens(self, ignore=None):
        """
        Affects ligand_reference().
        Ignores hydrogen atoms that are defined in the PDB and SDF/MOL2 files, and will not throw errors if the protonation states are different.
        Hydrogen atoms are not predicted with PLACER anyway.

        Parameters
        ----------
        ignore : bool, optional

        """
        if ignore is None:
            return self.__ignore_hydrogens
        else:
            assert isinstance(ignore, bool)
            self.__ignore_hydrogens = ignore


    def add_custom_residues(self, residue_dict):
        """
        User can provide a dictionary that defines the structure of a custom noncanonical amino acid or a ligand
        This information is used when mutating a residue to something that does not
        exist in the native RCSB entitiy dataset.
        
        The input dictionary must look like the following:
            {name3: {"sdf": [string representation of SDF file of the structure],
                     "atom_id": [atom names for each atom in SDF],
                     "leaving": [True/False depending if a particular atom should be deleted when part of a protein],
                     "pdbx_align": [integers 0/1 - has no effect on the job, add whatever you like]}}
        
        If you just want to add a ligand SDF to help PLACER parse that ligand, then use 'ligand_reference()'

        Parameters
        ----------
        residue_dict : dict
            DESCRIPTION.

        """
        assert isinstance(residue_dict, dict)
        for lg in residue_dict.keys():
            assert all([k in ['sdf', 'atom_id', 'leaving', 'pdbx_align'] for k in residue_dict[lg].keys()])
            assert isinstance(residue_dict[lg]["sdf"], str)
            assert len(residue_dict[lg]["atom_id"]) == len(residue_dict[lg]["leaving"])
        self.__custom_entities.update(residue_dict)
    
    def get_custom_residues(self):
        """
        Returns a verified dictioanry of user-provided entities that should be added to the internal residue database.
        """
        return self.__custom_entities

    def pdb(self, pdb=None):
        """
        Input structure in PDB format.
        Accepted inputs are:
          - path to a PDB file; or
          - string representation of the contents of a PDB file.
        """
        if pdb is None:
            return self.__pdb
        if os.path.exists(pdb):
            self.__pdb = open(pdb, "r").read()
        else:
            # assuming if input is not a filepath, then it's a PDBstring
            self.__pdb = pdb

    def cif(self, cif_file=None):
        """
        Only works if the input is from RCSB in PDBx/mmCIF format.
        Cifutils is not able to parse Rosetta or AF3 cif files correctly

        Parameters
        ----------
        cif_file : str, optional
            .cif or .cif.gz file.

        """
        if cif_file is None:
            return self.__cif_file
        self.__cif_file = cif_file

    def name(self, name=None):
        """
        Name given to the input object. Can be empty.
        """
        if name is None:
            return self.__name
        self.__name = name

    def exclude_sm(self, exclude=None):
        """
        Enables apo-structure analysis, if the input contains any ligand atoms.
        Omits any HETATM entities from the prediction.
        Does not return rmsd, kabsch and prmsd scores.
        """
        if exclude is None:
            return self.__exclude_sm
        else:
            assert isinstance(exclude, bool)
            self.__exclude_sm = exclude

    def mutate(self, mutate_dict=None):
        """
        Mutates a given residue at (chain, resno) to another residue <name3>.
        Input format is a dictionary:
        {(chain, resno): name3, ...}
        """
        if mutate_dict is None:
            return self.__mutate_dict
        else:
            assert isinstance(mutate_dict, dict)
            self.__mutate_dict = mutate_dict
            
    def predict_multi(self, predict_multi=None):
        """
        Enables multi-ligand predicion. All allowed ligands will be simultaneously predicted.
        predict_ligand() and fixed_ligand() inputs are respected.

        Currently, if you have multiple ligands, AND you're NOT using this option,
        then PLACER will randomly pick one ligand to predict and score.
        The other ligand will also be predicted, but not scored.

        Parameters
        ----------
        predict_multi : bool, optional

        """
        if predict_multi is None:
            return self.__predict_multi
        else:
            assert isinstance(predict_multi, bool)
            self.__predict_multi = predict_multi
        
    def target_res(self, target_res=None):
        """
        Defines or returns (if set) a non-ligand residue that will be used as corruption center

        Parameters
        ----------
        target_res : tuple, optional
            (chain, resno, name3) or (chain, resno). The default is None.

        """
        if target_res is None:
            return self.__target_res
        else:
            assert isinstance(target_res, tuple)
            self.__target_res = target_res


    def corruption_centers(self, points=None):
        """
        Atom names or XYZ coordinates from which corruption center(s) are randomly selected.
        Must have at least as many points as there are ligands in the system.

        The crop will be centerd around one of those points, AND the ligand is corrupted to that point as well.

        Atom name format (tuple): (chain, resno, name3, atom_name)

        """
        if points is None:
            return self.__corruption_centers
        else:
            assert isinstance(points, (list, np.array))
            self.__corruption_centers = points


    def crop_centers(self, points=None):
        """
        Atom names that will be used as CROP centers. This centers the crop to a particular part of the pocket, 
        but the ligands are still corrupted based on their input coordinates.
        Used for refining where the cropped sphere is. 
        This DOES NOT affect which atoms/ligands are selected for prediction. Use predict_ligand(...) for that. 
        One atom will be picked randomly from the provided set.

        Example: [('B', 200, 'HEM', 'FE'), ('B', 200, 'HEM', 'O1'), (1.0, 0.0, 2.0)]

        Atom name format (tuple): (chain: str, resno: int, name3: str, atom_name: str)
        Coordinate format (tuple): (x: float, y: float, z: float)
        """
        if points is None:
            return self.__crop_centers
        else:
            assert isinstance(points, (list, np.array))
            self.__crop_centers = points


    def create_from_dict(self, dct):
        """
        Create this input object from a dictionary where keys are the attribute names of this object, and keys are the set values.
        """
        for k,v in dct.items():
            getattr(self, k)(v)

    def copy(self):
        return copy.deepcopy(self)
