#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:48:48 2024

@author: ikalvet
"""
import torch
import numpy as np
import os
import itertools
_DIR = os.path.dirname(__file__)
# parse periodic table

i2a = [l.strip().split()[:2] for l in open(f'{_DIR}/../data/elements.txt','r').readlines()]
i2a = {int(i):a for i,a in i2a}


def get_plddt_pde(X,D,sel):
    '''interaction plDDT deraved from pDE'''
    mask = (torch.cdist(X,X)<15.0)[sel]
    pde = torch.nn.functional.softmax(D[sel],dim=-1)[mask]
    score = [pde[...,51-s:51+s].sum(-1).mean() for s in (5,10,20,40)]
    score = torch.stack(score).mean()
    return score
    

def get_plddt(logits,sel):
    '''plDDT of the corrupted part'''
    score = logits.argmax(-1).float()[sel]
    score = (score*0.02+0.01).mean()
    return score
    
    
def get_prmsd(sigma,sel):
    '''predicted RMSD from per-atom sigmas'''
    s = sigma[sel]
    score = ((s**2).mean())**0.5
    return score


def Pnear(rmsds : np.array, scores : np.array):
    '''
    Boltzmann−weighted discrimination score
    https://doi.org/10.1021/acs.jctc.6b00819
    Suppl Info, eq. S7 (p.16)
    '''
    
    k = 10
    
    bins = [0.50, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 
            2.50, 2.75, 3.00, 3.50, 4.00, 5.00, 6.0]
    Nj = len(bins)

    Emin,Emax = np.quantile(scores,(0.05,0.95))
    Enorm = Emax-Emin
    Emin = scores.min()
    
    Pi = np.exp(-k*(scores-Emin)/Enorm)
    Psum = Pi.sum()
    
    j = np.digitize(rmsds,bins)
    pnear = np.mean([Pi[j<=b].sum()/Psum for b in range(Nj)])
    
    return pnear


def fix_val_leu(G,X):
    '''fix atom naming in valines and leucines'''
    
    vals = {n[:3] for n in G.nodes if n[2]=='VAL'}
    vals = [[(*v,a) for a in ('CB','CA','CG1','CG2')] for v in vals]
    
    leus = {n[:3] for n in G.nodes if n[2]=='LEU'}
    leus = [[(*l,a) for a in ('CG','CB','CD1','CD2')] for l in leus]
    
    for oabc in vals+leus:
        if not all(i in G.nodes for i in oabc):
            continue
        o,a,b,c = [G.nodes[i]['index'] for i in oabc]
        abc = torch.dot(X[o]-X[a], torch.cross(X[o]-X[b], X[o]-X[c], dim=-1))

        # swap if left-handed
        if abc<0.0:
            X[[b,c]] = X[[c,b]]

    return


def mutate(res_lines, res_name, atoms):
    
    res_old = {l[12:16].strip():l for l in res_lines}
    out = []
    for aname,a in atoms.items():
        if a.element==1:
            continue
        l = res_old.get(aname)
        if l is not None:
            l = l[:17] + res_name + l[20:]
            out.append(l)
        elif aname!='OXT':
            l = res_old['CA']
            l = l[:17] + res_name + l[20:]
            l = l[:12] + "%-4s"%(' '*a.align+aname) + l[16:]
            out.append(l)
    
    return out


def create_pdbmodel(G, X, devs, sample_id=None):
    """

    Parameters
    ----------
    G : TYPE
        networkx graph of the crop.
    Xs : TYPE
        predicted coordinates of the crop.
    devs : TYPE
        predicted rmsds (deviations) of the crop. Added as bfactors
    sample_id : int, optional
        model number added to the top of the PDB as "MODEL <N>". The default is None.

    Returns
    -------
    mdl_str : str
        PDB representation of the model.

    """
    acount = 1
    a2i = {}

    fix_val_leu(G,X)  # does this change the G object outside of this function too?
    
    if sample_id is not None:
        mdl_str='MODEL %d\n'%(sample_id)
    else:
        mdl_str='MODEL %d\n'%(0)

    for (r,node),xyz,err in zip(G.nodes(data=True),X,devs):
        a = node['atom']
        element = i2a[a.element]
        mdl_str += "%-6s%5s %-4s %3s%2s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n"%(
            "HETATM" if a.hetero==True else "ATOM",
            acount, ' '*a.align+r[3], r[2], r[0], int(r[1]),
            xyz[0], xyz[1], xyz[2], a.occ, err, element, a.charge)
        a2i[r] = acount
        acount += 1
    for a,b in G.edges:
        a = G.nodes[a]['atom']
        b = G.nodes[b]['atom']
        if a.occ==0.0 or b.occ==0.0 or (a.hetero==False and b.hetero==False):
            continue
        mdl_str += "%-6s%5d%5d\n"%("CONECT", a2i[a.name], a2i[b.name])
    mdl_str += 'TER\n'
    mdl_str += 'ENDMDL\n'
    return mdl_str


def rank_outputs(outputs, rank_score):
    assert rank_score in ["prmsd", "plddt", "plddt_pde"]
    print(f"Ranking predictions based on {rank_score}. This will make the model numbers, and the CSV file look different from the above printed results.")
    _reverse = True
    if rank_score == "prmsd":
        _reverse = False

    for j,(i,dct) in enumerate(sorted(outputs.items(), key=lambda a: a[1][rank_score], reverse=_reverse)):
        outputs[j] = dct
        outputs[j]["model_idx"] = j+1
        outputs[j]["model"] = dct["model"].replace(f"MODEL {i+1}\n", f"MODEL {j+1}\n")  # is this safe?
    return outputs


def get_common_ligands():
    """
    Common ligands encountered in protein crystallography
    List obtained from AlphaFold3 supplementary methods:
    https://www.nature.com/articles/s41586-024-07487-w
    "Accurate structure prediction of biomolecular interactions with AlphaFold 3"
    J. Abramson et al, Nature 2024, 493-500.
    """
    ligands = ["144", "15P", "1PE", "2F2", "2JC", "3HR", "3SY", "7N5", "7PE", "9JE", "AAE", "ABA", "ACE", "ACN", "ACT", "ACY", "AZI", "BAM", "BCN",
               "BCT", "BDN", "BEN", "BME", "BO3", "BTB", "BTC", "BU1", "C8E", "CAD", "CAQ", "CBM", "CCN", "CIT", "CL", "CLR",
                "CM", "CMO", "CO3", "CXS", "D10", "DEP", "DIO", "DMS", "DN", "DOD", "DOX", "EDO", "EEE", "EGL", "EOH", "EOX", "EPE",
                "ETF", "FCY", "FJO", "FLC", "FMT", "FW5", "GOL", "GSH", "GTT", "GYF", "HED", "IHP", "IHS", "IMD", "IOD", "IPA", "IPH",
                "LDA", "MB3", "MEG", "MES", "MLA", "MLI", "MOH", "MPD", "MRD", "MSE", "MYR", "N", "NA", "NH2", "NH4", "NHE", "NO3", "O4B", "OHE", "OLA",
                "OLC", "OMB", "OME", "OXA", "P6G", "PE3", "PE4", "PEG", "PEO", "PEP", "PG0", "PG4", "PGE", "PGR",
                "PLM", "PO4", "POL", "POP", "PVO", "SAR", "SCN", "SEO", "SEP", "SIN", "SO4", "SPD", "SPM", "SR", "STE", "STO", "STU",
                "TAR", "TBU", "TME", "TPO", "TRS", "UNK", "UNL", "UNX", "UPL", "URE"]
    return ligands

