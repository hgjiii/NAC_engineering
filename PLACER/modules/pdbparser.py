import sys, os
import copy
import pandas as pd
import networkx as nx
from typing import List,Dict,Any
import torch
import random
import itertools
import numpy as np
from openbabel import openbabel

sys.path.append(os.path.dirname(__file__))
import obutils
import cifutils


class PDBParser:
    
    def __init__(self, skip_res=[], mols=None):

        self.Parser = cifutils.CIFParser(mols=mols)
        self.skip_res = skip_res

    def parseProtein(self, pdbstr : List[str]) -> Dict[str,cifutils.Chain]:

        # parse atoms
        atoms = [cifutils.Atom(name=(l[21],int(l[22:26].strip()),l[17:20].strip(),l[12:16].strip()),
                               xyz=[float(l[30:38]),float(l[38:46]),float(l[46:54])],
                               occ=float(l[54:60]),
                               bfac=float(l[60:66]),
                               leaving=False,
                               leaving_group=[],
                               parent=None,
                               element=None,
                               metal=None,
                               charge=None,
                               hyb=None,
                               nhyd=None,
                               align=0,
                               hvydeg=None,
                               hetero=False)
                 for l in pdbstr if l[:4]=="ATOM"]
        
        chains = {}
        for chid,chain_atoms in itertools.groupby(atoms, key=lambda a : a.name[0]):
            
            # collect residues
            chain_atoms = {a.name : a for a in chain_atoms}
            res_names = list({aname[:3] for aname in chain_atoms.keys()})
            res_names.sort(key= lambda x : x[1])
            chain_residues = [copy.deepcopy(self.Parser.getRes(name[2])['res'])._replace(name=name)
                              for name in res_names]
            chain_residues = {r.name : r for r in chain_residues}
            
            # populate residue atoms with the parsed data
            for rname,residue in chain_residues.items():
                for aname,atom in residue.atoms.items():
                    if (*rname,aname) in chain_atoms.keys():
                        leaving_group = [(*rname,a) for a in atom.leaving_group]
                        chain_residues[rname].atoms[aname] = chain_atoms[(*rname,aname)]._replace(
                            leaving_group=leaving_group,
                            element=atom.element,
                            metal=atom.metal,
                            charge=atom.charge,
                            hyb=atom.hyb,
                            nhyd=atom.nhyd,
                            align=atom.align,
                            hvydeg=atom.hvydeg
                        )


            # add bonds between residues along the sequence
            bonds = []
            skip_atoms = set()
            for ra,rb in zip(res_names[:-1],res_names[1:]):
                if rb[1]-ra[1]!=1:
                    continue

                # IK: update hybridization, nhyd and charge of backbone N
                chain_residues[rb].atoms["N"] = chain_residues[rb].atoms["N"]._replace(nhyd=chain_residues[rb].atoms["N"].nhyd-2, hyb=2, charge=0)

                ra = chain_residues[ra]
                rb = chain_residues[rb]
                skip_atoms.update(ra.atoms['C'].leaving_group + rb.atoms['N'].leaving_group)
                bonds.append(cifutils.Bond(
                    a=ra.atoms['C'].name,
                    b=rb.atoms['N'].name,
                    aromatic=False,
                    in_ring=False,
                    order=1,
                    intra=False,
                    length=1.5 #
                ))
                
            # collect atoms
            atoms = {(*rname,aname) : atom 
                     for rname,residue in chain_residues.items()
                     for aname,atom in residue.atoms.items()
                     if (*rname,aname) not in skip_atoms}
            a2i = {a:i for i,a in enumerate(atoms.keys())}
            i2a = {i:a for a,i in a2i.items()}

            # collect intra-residue bonds
            bonds_intra = [bond._replace(a=(*rname,bond.a),
                                         b=(*rname,bond.b))
                           for rname,residue in chain_residues.items()
                           for bond in residue.bonds]
            bonds_intra = [bond for bond in bonds_intra
                           if bond.a in atoms.keys() and bond.b in atoms.keys()]
            bonds.extend(bonds_intra)
            
            # relabel chirals and planars
            chirals = [[(*rname,c) for c in chiral] 
                       for rname,residue in chain_residues.items()
                       for chiral in residue.chirals]
            planars = [[(*rname,p) for p in planar] 
                       for rname,residue in chain_residues.items()
                       for planar in residue.planars]

            chirals = [c for c in chirals if all([ci in atoms.keys() for ci in c])]
            planars = [c for c in planars if all([ci in atoms.keys() for ci in c])]

            # relabel and filter out automorphisms
            automorphisms = [[[(*rname,a) for a in auto] for auto in residue.automorphisms] 
                             for rname,residue in chain_residues.items() 
                             if len(residue.automorphisms)>1]
            idx = [np.array([list(map(lambda x : a2i.get(x, -1),ai)) for ai in auto]) for auto in automorphisms]
            idx = [np.unique(i[:,(i>=0).all(axis=0)],axis=0) for i in idx]
            automorphisms = [[list(map(i2a.get,i)) for i in sub_idx] for sub_idx in idx if sub_idx.shape[1]>0]
            
            # put everything into a Chain
            chains[chid] = cifutils.Chain(id=chid,
                                          type='polypeptide(L)',
                                          sequence=None,
                                          atoms=atoms,
                                          bonds=bonds,
                                          chirals=chirals,
                                          planars=planars,
                                          automorphisms=automorphisms)

        return chains
    
        
    def parseLigand(self, 
                    obmol : openbabel.OBMol = None, 
                    smiles : str = None,
                    no_auto : bool = False,
                    pdbstr : str = None) -> Dict[str,cifutils.Chain]:

        # correct for pH and add hydrogens
        #obmol.CorrectForPH()
        #obmol.AddHydrogens()
        
        # IK: Re-parsing the ligand section in the PDB
        if pdbstr is not None:
            hetero = [l for l in pdbstr if l[:6]=='HETATM' if l[17:20].strip() not in self.skip_res]
            conect = [l for l in pdbstr if l[:6]=='CONECT']
            pdb_lig_dict = {int(l[6:11].strip()): (l[21], int(l[22:26].strip()), l[17:20], l[12:16].strip()) for l in hetero}  # (chain, resno, name3, atomname)
            pdbbonds = {}
            for l in conect:
                l = l.strip()
                lspl = ["CONECT"] + [l[6+n*5:6+n*5+5] for n in range(len(l[6:])//5)]  # splitting CONECT line based on official format
                a0 = lspl[1]
                for a1 in lspl[2:]:
                    pair = sorted([int(a0), int(a1)])
                    if pair[0] not in pdb_lig_dict.keys() or pair[1] not in pdb_lig_dict.keys():
                        continue
                    if f"{pair[0]}-{pair[1]}" not in pdbbonds:
                        pdbbonds[f"{pair[0]}-{pair[1]}"] = (pdb_lig_dict[pair[0]], pdb_lig_dict[pair[1]])


        # get atoms and their features
        if obmol.NumResidues()>0:
            get_atom_name  = lambda a : (a.GetResidue().GetChain().strip(),
                                         int(a.GetResidue().GetNumString().strip()),
                                         a.GetResidue().GetName().strip()[:3],
                                         a.GetResidue().GetAtomID(a).strip())

            atoms = [cifutils.Atom(name=get_atom_name(a),
                                   xyz=[a.GetX(),a.GetY(),a.GetZ()],
                                   occ=1.0,
                                   bfac=1.0,
                                   leaving=False,
                                   leaving_group=[],
                                   parent=None,
                                   element=a.GetAtomicNum(),
                                   metal=a.IsMetal(),
                                   charge=a.GetFormalCharge(),
                                   hyb=a.GetHyb(),
                                   nhyd=a.ExplicitHydrogenCount(),
                                   align=0,
                                   hvydeg=a.GetHvyDegree(),
                                   hetero=True)
                     for r in openbabel.OBResidueIter(obmol) 
                     for a in openbabel.OBResidueAtomIter(r)]


        else:
            get_atom_name  = lambda a : ('L',1,'LIG',self.Parser.i2a[a.GetAtomicNum()]+str(a.GetIdx()))

            atoms = [cifutils.Atom(name=get_atom_name(a),
                                   xyz=[a.GetX(),a.GetY(),a.GetZ()],
                                   occ=1.0,
                                   bfac=1.0,
                                   leaving=False,
                                   leaving_group=[],
                                   parent=None,
                                   element=a.GetAtomicNum(),
                                   metal=a.IsMetal(),
                                   charge=a.GetFormalCharge(),
                                   hyb=a.GetHyb(),
                                   nhyd=a.ExplicitHydrogenCount(),
                                   align=0,
                                   hvydeg=a.GetHvyDegree(),
                                   hetero=True)
                     for a in openbabel.OBMolAtomIter(obmol)]
        anames = np.array([a.name for a in atoms],dtype=tuple)

        chid = atoms[0].name[0]

        bonds = [(bond,bond.GetBeginAtom(),bond.GetEndAtom())
                 for bond in openbabel.OBMolBondIter(obmol)]

        # Fixing missing bonds in the OpenBabel molecule object
        # And removing any extra bonds that got created in the OpenBabel molecule
        # if they're not defined in the PDB file CONECT section.
        if pdbstr is not None:
            if len(bonds) < len(pdbbonds):
                _bonds_parsed = [(get_atom_name(i), get_atom_name(j)) for b,i,j in bonds]
                missing_bonds = []
                for k, pdb_bond in pdbbonds.items():
                    if (pdb_bond[0], pdb_bond[1]) in _bonds_parsed or (pdb_bond[1], pdb_bond[0]) in _bonds_parsed:
                        continue
                    else:
                        if (k, pdb_bond) not in missing_bonds:
                            missing_bonds.append((k, pdb_bond))
                if len(missing_bonds) != 0:
                    assert len(missing_bonds) == len(pdbbonds) - len(bonds), f"missing bonds: {missing_bonds}, PDB bonds: {pdbbonds}, oBabel bonds: {bonds}"
                    for k, pdb_bond in missing_bonds:
                        print(f"Adding missing bond between {pdb_bond}")
                        a1 = [a for r in openbabel.OBResidueIter(obmol) for a in openbabel.OBResidueAtomIter(r) if get_atom_name(a)[3] == pdb_bond[0][3]][0]
                        a2 = [a for r in openbabel.OBResidueIter(obmol) for a in openbabel.OBResidueAtomIter(r) if get_atom_name(a)[3] == pdb_bond[1][3]][0]
                        obmol.AddBond(a1.GetIdx(), a2.GetIdx(), 1)
                    bonds = [(bond,bond.GetBeginAtom(),bond.GetEndAtom())
                             for bond in openbabel.OBMolBondIter(obmol)]
            elif len(bonds) > len(pdbbonds) and len(conect) != 0:
                print("Openbabel thinks there are more bonds than what CONECT records show. Deleting the extra ones, unless they're from added hydrogens.")
                _bonds = []
                for bond in bonds:
                    if all([x not in pdbbonds.values() for x in [(get_atom_name(bond[1]), get_atom_name(bond[2])), (get_atom_name(bond[2]), get_atom_name(bond[1]))]]):
                        if obmol.HasHydrogensAdded() and any([ a.GetAtomicNum() == 1 for a in [bond[1], bond[2]]]):
                            _bonds.append(bond)
                            # print(f"    NOT Deleting bond: {get_atom_name(bond[1])} - {get_atom_name(bond[2])}")
                            continue
                        print(f"    Deleting bond: {get_atom_name(bond[1])} - {get_atom_name(bond[2])}")
                    else:
                        _bonds.append(bond)
                bonds = _bonds


        # IK: Fixing metal hybridization
        # for r in openbabel.OBResidueIter(obmol):
        #     for a in openbabel.OBResidueAtomIter(r):
        #         if a.IsMetal() == True:
        #             _bonds_metal = [b for b in bonds if a in b[1:]]
        #             _hyb = 6
        #             if len(_bonds_metal) <= 6:
        #                 _hyb = len(_bonds_metal)
        #             if a.GetHyb() == 1:
        #                 _hyb = 4

        #             if _hyb != a.GetHyb():
        #                 print(f"Changing hybridization of {get_atom_name(a)} from {a.GetHyb()} to {_hyb}")
        #                 a.SetHyb(_hyb)
        #                 for i, atom in enumerate(atoms):
        #                     if atom.name == get_atom_name(a):
        #                         atoms[i] = atom._replace(hyb=_hyb)

        bonds = [
            cifutils.Bond(
                a = get_atom_name(i),
                b = get_atom_name(j),
                aromatic=b.IsAromatic(),
                in_ring=b.IsInRing(),
                order=b.GetBondOrder(),
                intra=True,
                length=b.GetLength())
            for b,i,j in bonds
        ]

        # get automorphisms
        if no_auto==True:
            automorphisms = []
        else:
            automorphisms = obutils.FindAutomorphisms(obmol, heavy=True)
            automorphisms = self.Parser.AddQuasisymmetries(obmol, automorphisms)
            mask = (automorphisms[:1]==automorphisms).all(dim=0)
            automorphisms = automorphisms[:,~mask]
            if automorphisms.shape[1]>0:
                automorphisms = [[list(map(tuple,a)) for a in anames[automorphisms]]]
            else: 
                automorphisms = []

        # get chirals and planars
        chirals = obutils.GetChirals(obmol, heavy=True)
        planars = obutils.GetPlanars(obmol, heavy=True)
        
        chirals = [[tuple(anames[ci]) for ci in c] for c in chirals]
        planars = [[tuple(anames[pi]) for pi in p] for p in planars]

        # split into chains
        chains = {}
        for chid,group in itertools.groupby(atoms,lambda a : a.name[0]):
            chain_atoms = {a.name:a for a in list(group)}
            chain = cifutils.Chain(
                id=chid,
                type='nonpoly',
                sequence='',
                atoms=chain_atoms,
                bonds=[bnd for bnd in bonds if bnd.a in chain_atoms or bnd.b in chain_atoms],
                chirals=[chi for chi in chirals if any([a in chain_atoms for a in chi])],
                planars=[pl for pl in planars if any([a in chain_atoms for a in pl])],
                automorphisms=automorphisms
            )
            chains[chid] = chain

        return chains


    # def get_pdb_content(self, filename : str, reference_obmol_dict=None):
    #     """
    #     === DEPRECATED ===
    #     Parsed the PDB file into chains dictionary and openbabel Molecule

    #     Parameters
    #     ----------
    #     filename : str
    #         input PDB file.
    #     reference_obmol : dict, optional
    #         Dictionary with openbabel.OBMol objects parsed from user-provided SDF/MOL2 files.
    #         Is used to adjust the parsed ligand atom and bond properties. The default is None.

    #     Returns
    #     -------
    #     chains : dict
    #         DESCRIPTION.
    #     obmol : openbabel.OBMol
    #         DESCRIPTION.
    #     """
        
    #     lines = open(filename,'r').readlines()
        
    #     # parse the protein
    #     chains = self.parseProtein(lines)
        
    #     # parse the ligand
    #     hetero = [l for l in lines if l[:6]=='HETATM' if l[17:20].strip() not in self.skip_res]
    #     conect = [l for l in lines if l[:6]=='CONECT']
    #     sm_str = "".join(hetero+conect)

    #     obmol = openbabel.OBMol()
    #     obConversion = openbabel.OBConversion()
    #     obConversion.SetInFormat("pdb")
    #     obConversion.ReadString(obmol,sm_str)

    #     # obmol.AddHydrogens()

    #     if reference_obmol_dict is not None:
    #         for _r in openbabel.OBResidueIter(obmol):
    #             if _r.GetName() in reference_obmol_dict.keys():
    #                 print(_r.GetName())
    #                 # Finding the atom mapping between the parsed and reference OBMol objects
    #                 mapping = obutils.get_obmol_mapping(_r, reference_obmol_dict[_r.GetName()])
    #                 # Adjusting the atom properties in the obmol object
    #                 obutils.compare_pdbmol_sdfmol(obmol, reference_obmol_dict[_r.GetName()], mapping)

    #     # Converting the obmol object into chains
    #     chains.update(self.parseLigand(obmol, pdbstr=lines))

    #     return chains,obmol


    def parse_ligand_from_pdb_to_obmol(self, pdbstr, ligand_reference=None, ignore_hydrogens=False):
        """

        Parameters
        ----------
        pdbstr : str
            Contents of PDB file with HETATM and CONECT lines representing the ligand(s).
        ligand_reference : dict, optional
            Dictionary with parsed reference ligand obmol objects. The default is None.

        Returns
        -------
        obmol : TYPE
            openbabel object of ligand(s) as residues.

        """
        hetero = [l for l in pdbstr.split("\n") if l[:6]=='HETATM' if l[17:20].strip() not in self.skip_res]
        conect = [l for l in pdbstr.split("\n") if l[:6]=='CONECT']
        sm_str = "\n".join(hetero+conect)

        if len(hetero) == 0:
            return None

        obmol = openbabel.OBMol()
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("pdb")
        obConversion.ReadString(obmol,sm_str)

        if ligand_reference is not None:
            # Only add hydrogens if the molecule does not seem to have any
            # It's necessary because most reference structures would have hydrogens added,
            # but structures from the PDB do not.
            if not any([a.GetAtomicNum() == 1 for a in openbabel.OBMolAtomIter(obmol)]):
                obmol.AddHydrogens()
                print("Adding hydrogens to obmol")

            for res in openbabel.OBResidueIter(obmol):
                if res.GetName() in ligand_reference.keys():
                    # print(res.GetName())
                    # Finding the atom mapping between the parsed and reference OBMol objects
                    mapping = obutils.get_obmol_mapping(res, ligand_reference[res.GetName()], ignore_hydrogens)
                    # Adjusting the atom properties in the obmol object
                    obutils.compare_pdbmol_sdfmol(obmol, ligand_reference[res.GetName()], mapping)
        return obmol
