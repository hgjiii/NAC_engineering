import os, sys
import json
import random
import torch
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from openbabel import openbabel

sys.path.append(os.path.dirname(__file__))
import obutils

openbabel.OBMessageHandler().SetOutputLevel(0)

class CSDDataset():

    def __init__(self, 
                 csv : str,
                 ncpu : int,
                 world_size : int,
                 rank : int,
                 params : dict):
        
        # parse .csv
        df = pd.read_csv(csv, index_col=0)

        # sample from the set inversely proportional to the number of 1-hop neighbors
        weights = 1/(1+df.degree)
        sampler = DistributedWeightedSampler(weights = weights,
                                             nheavy = df.nheavy,
                                             maxatoms = params['maxatoms'],
                                             world_size = world_size,
                                             rank = rank)

        self.dataset = torch.utils.data.DataLoader(Dataset(df.hash.tolist(), params),
                                                   batch_sampler=sampler,
                                                   num_workers=ncpu,
                                                   pin_memory=True,
                                                   collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(sample):
        
        # remove samples with no frames
        sample = [s for s in sample if s['frames'].shape[0]>0]

        lens = [s['X'].shape[0] for s in sample]
        cumlens = np.cumsum([0]+lens[:-1])
        
        data = dict()
        data['idx'] = torch.cat([torch.full((l,),i) for i,l in enumerate(lens)])
        data['label'] = [s['label'] for s in sample]

        data['Y'] = torch.cat([s['Y'] for s in sample], dim=1)

        for key in ('X','f1d','bondlen'):
            data[key] = torch.cat([s[key] for s in sample], dim=0)

        n_f2d = sample[0]['f2d'].shape[-1]
        
        data['f2d'] = torch.stack([torch.block_diag(*[s['f2d'][...,i] for s in sample]) for i in range(n_f2d)], dim=-1)
        
        data['separation'] = torch.block_diag(*[s['separation'] for s in sample])

        data['frames'] = [s['frames'] for s in sample]
        
        for key in ('bonds','angles','dihedrals','chirals','planars'):
            to_cat = [s[key]+shift for s,shift in zip(sample,cumlens) if s[key].shape[0]>0]
            if len(to_cat)>0:
                data[key] = torch.cat(to_cat, dim=0)
            else:
                data[key] = torch.Tensor()

        '''
        for key in ('chirals','planars'):
            to_cat = [torch.cat([s[key][:,:-1]+shift,s[key][:,-1:]], dim=1) 
                      for s,shift in zip(sample,cumlens) if s[key].shape[0]>0]
            if len(to_cat)>0:
                data[key] = torch.cat(to_cat, dim=0)
            else:
                data[key] = torch.Tensor()
        '''

        data['mask'] = data['idx'][:,None]==data['idx'][None,:]
        
        return data

        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, labels, params):
        
        self.labels = labels
        self.params = params

        self.dims1d = (7, # group
                       18, # period
                       1, # is lanthanide
                       1, # is actinide
                       15, # group with lanthanides
                       15, # group with actinides
                       params['maxcharge']*2+1, # formal charge
                       params['maxhydr'], # number of hydrogens
                       params['maxhyb'], # hybridization state
                       1) # corrupted mask

        
        self.dims2d = (2,2,4,params['maxpath']+1)
        
        # parse the periodic table
        DIR = os.path.dirname(__file__)
        with open(f'{DIR}/data/PeriodicTableJSON.json') as file:
            data = json.load(file)['elements']
        self.elements = {
            d['number']:{
                "period":d["period"],
                "group":d["group"],
                "is_lanthanide":d["category"]=="lanthanide",
                "is_actinide":d["category"]=="actinide",
                } for d in data}

        # parse the quasi-symmetric groups table
        df = pd.read_csv(f'{DIR}/data/quasisym.csv')
        df.indices = df.indices.apply(lambda x : [int(xi) for xi in x.split(',')])
        df['matcher'] = df.apply(lambda x : openbabel.OBSmartsPattern(), axis=1)
        df.apply(lambda x : x.matcher.Init(x.smarts), axis=1)
        self.quasisym = {smarts:(matcher,torch.tensor(indices))
                         for smarts,matcher,indices 
                         in zip(df.smarts,df.matcher,df.indices)}

    
    def AddQuasisymmetries(self, 
                           obmol : openbabel.OBMol,
                           automorphisms : torch.Tensor) -> torch.Tensor:
        '''add quasisymmetries to automorphisms
        '''

        renum = []
        for smarts,(matcher,indices) in self.quasisym.items():
            res = openbabel.vectorvInt()
            if matcher.Match(obmol,res,0):
                res = torch.tensor(res)[:,indices]-1
                res = res.sort(-1)[0]
                res = torch.unique(res,dim=0)
                for res_i in res:
                    res_i = torch.tensor(list(itertools.permutations(res_i,indices.shape[0])))
                    renum.append(res_i)
                
        if len(renum)<1:
            return automorphisms
        elif len(renum)==1:
            renum = renum[0]
        else:
            random.shuffle(renum)
            renum = renum[:4]
            renum = torch.stack([torch.cat(ijk) for ijk in itertools.product(*renum)])

        L = automorphisms.shape[-1]
        modified = automorphisms[:,None].repeat(1,renum.shape[0],1)
        modified[...,renum[0]]=automorphisms[:,renum]
        modified = modified.reshape(-1,L)
        modified = torch.unique(modified, dim=0)
        
        return modified

    
    def get_atom_features(self, atom):
        ''' '''
        atomic_num = atom.GetAtomicNum()
        charge = atom.GetFormalCharge()
        nhyd = atom.GetTotalDegree()-atom.GetHvyDegree()
        if nhyd>self.params['maxhydr']:
            nhyd = self.params['maxhydr']
        hyb = atom.GetHyb()
        element = self.elements.get(atomic_num)
        if element is not None:
            period,group,is_lan,is_act = element.values()
            group_lan = ((atomic_num-56)%16)*is_lan
            group_act = ((atomic_num-88)%16)*is_act
            return (period, # element's period
                    group, # element's period
                    is_lan, # is the element a lanthanide?
                    is_act, # is the element an actinide?
                    group_lan, # element's group within lanthanides
                    group_act, # element's group within actinides
                    charge, # element's charge
                    nhyd, # element's number of hydrogens
                    hyb) # element's hybridization
        else:
            return (0,0,0,0,0,0,0,0,0)
        
        
    def OneHotF1D(self, 
                  f1d : torch.tensor) -> torch.tensor:
        '''one-hot-encode 1D features'''

        # element embeddings
        element = [
            torch.nn.functional.one_hot(f1d[:,0],self.dims1d[0]+1)[:,1:],
            torch.nn.functional.one_hot(f1d[:,1],self.dims1d[1]+1)[:,1:],
            f1d[:,2:3],
            f1d[:,3:4],
            torch.nn.functional.one_hot(f1d[:,4],self.dims1d[4]+1)[:,1:],
            torch.nn.functional.one_hot(f1d[:,5],self.dims1d[5]+1)[:,1:] ]

        charge,nhyd,hyb = f1d[:,-3:].T

        # charge embedding
        qmax = self.params['maxcharge']
        charge = torch.clamp(charge,min=-qmax,max=qmax)
        emb = torch.block_diag(torch.ones((qmax,qmax)).triu(),
                               torch.ones((1,1)),
                               torch.ones((qmax,qmax)).tril())
        charge = emb[charge+qmax]

        # number of hydrogens
        hmax = self.params['maxhydr']
        nhyd = torch.ones((hmax+1,hmax)).tril(diagonal=-1)[nhyd]

        # hybridization
        hmax = self.params['maxhyb']
        hyb = torch.eye(hmax)[torch.clamp(hyb,min=0,max=hmax-1)]*(hyb<hmax)[:,None] # zero out if off-range
        
        # stack all features together
        f1d = torch.cat(element + [charge,nhyd,hyb], dim=-1)
        
        return f1d

    
    def OneHotF2D(self, f2d : torch.tensor) -> torch.tensor:
        '''one-hot-encode 2D features'''
    
        f2d = torch.cat([torch.nn.functional.one_hot(f,d)
                         for f,d in zip(f2d.permute(2,0,1),self.dims2d)], dim=-1).float()
        
        return f2d

    
    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, index):

        label = self.labels[index]

        # load the selected molecule
        fname = f'{self.params["DIR"]}/{label[:2]}/{label[2:4]}/{label}.pt'
        data = torch.load(fname)

        obmol = openbabel.OBMol()
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("mol2")
        obConversion.ReadString(obmol, data['mol2'])
        L = obmol.NumAtoms()

        # get topology
        mol = obutils.GetTopology(obmol)
        mol['frames'] = mol['angles']

        # get automorphisms
        automorphisms = obutils.FindAutomorphisms(obmol, heavy=True)
        automorphisms = self.AddQuasisymmetries(obmol, automorphisms)

        ##### f1d #####
        f1d = torch.tensor([self.get_atom_features(a) 
                            for a in openbabel.OBMolAtomIter(obmol)])
        f1d = self.OneHotF1D(f1d)
        mol['f1d'] = torch.cat([f1d,torch.ones((L,1))],dim=-1)

        ##### f2d #####
        f2d = obutils.GetFeatures2D(obmol,self.params['maxpath'])
        mol['separation'] = f2d[...,-1]
        mol['f2d'] = self.OneHotF2D(f2d)
        
        # account for symmetries in the molecule,
        # standardize the number of conformers
        Y = data['xyz'][:,automorphisms].reshape(-1,L,3)
        maxperm = self.params['maxperm']
        B = Y.shape[0]
        if B>=maxperm:
            # subselect, if too many 
            Y = Y[torch.randperm(B)[:maxperm]]
        else:
            # fill in with conformer #0, if too few
            Y = torch.cat([Y,Y[:1].repeat(maxperm-Y.shape[0],1,1)],dim=0)
        
        mol['Y'] = Y
        mol['X'] = torch.randn_like(Y[0])*self.params['sigma']
        mol['label'] = label
        
        return mol

    
class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, nheavy, maxatoms, world_size, rank):

        self.weights = torch.tensor(weights).float()
        self.nheavy = torch.tensor(nheavy)
        self.world_size = world_size
        self.rank = rank
        self.maxatoms = maxatoms
        self.total_size = int(weights.sum())
        self.sample = self.set_epoch(0)
        self.sample_size = len(self.sample)
    
    
    def split_by_sum(self, numbers, indices, maxatoms):

        result = []
        current_sum = 0
        sub_list = []

        for number,index in zip(numbers,indices):
            if current_sum + number <= maxatoms:
                sub_list.append(index)
                current_sum += number
            else:
                result.append(sub_list)
                sub_list = [index]
                current_sum = number

        if sub_list:
            result.append(sub_list)

        return result


    def __iter__(self):
        return iter(self.sample)


    def __len__(self):
        return self.sample_size


    def set_epoch(self, epoch):
        
        self.epoch = epoch

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # sample indices
        indices = torch.multinomial(self.weights, self.total_size, 
                                    replacement=False, generator=g)

        # shuffle indices
        indices = indices[torch.randperm(self.total_size, generator=g)]
        
        # get the number of heavy atoms in selected molecules
        nheavy = self.nheavy[indices]

        # split the list of molecules into sub-lists such that 
        # the total number of heavy atoms in each of the sub-lists
        # does not exceed the cutoff
        splits = self.split_by_sum(nheavy.tolist(),indices.tolist(),self.maxatoms)

        # make sure that all workers get equal number of batches
        nsplits = len(splits)
        nsplits = nsplits - nsplits%self.world_size
        splits = splits[:nsplits]

        # select according to rank
        sample = splits[self.rank::self.world_size]
        self.sample_size = len(sample)
        self.sample = sample
        
        return sample
