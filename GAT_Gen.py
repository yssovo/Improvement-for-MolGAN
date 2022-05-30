# coding=utf-8
"""
Anonymous author
"""
import os
from string import printable
import sys
import numpy as np
import math
from time import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from GAT import MaskedGraphAF

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

import environment as env
#from environment import check_valency, convert_radical_electrons_to_hydrogens



def reward_target_molecule_similarity(mol, target, radius=2, nBits=2048,
                                      useChirality=True):
    """
    Reward for a target molecule similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule
    :param mol: rdkit mol object
    :param target: rdkit mol object
    :return: float, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target,
                                                            radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)




class GraphFlowModel(nn.Module):
    """
    Reminder:
        self.args: deq_coeff
                   deq_type

    Args:

    
    Returns:

    """
    def __init__(self, edge_unroll, batch_dim, vertexes, edges, nodes, bond_decoder_m, atom_decoder_m):
        super(GraphFlowModel, self).__init__()
        self.max_size = vertexes
        self.node_dim = nodes
        self.bond_dim = edges
        self.b_dim = edges
        self.edge_unroll = edge_unroll

        self.batch_size = batch_dim

        self.atom_decoder_m = atom_decoder_m
        self.bond_decoder_m = bond_decoder_m

        self.node_masks, self.adj_masks, \
            self.link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = self.node_masks.size(0)  # (max_size) + (max_edge_unroll - 1) / 2 * max_edge_unroll + (max_size - max_edge_unroll) * max_edge_unroll
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim
        print('latent node length: %d' % self.latent_node_length)
        print('latent edge length: %d' % self.latent_edge_length)

        self.constant_pi = nn.Parameter(torch.Tensor([3.1415926535]), requires_grad=False)
        #learnable
        self.prior_ln_var = nn.Parameter(torch.zeros([1])) # log(1^2) = 0
        nn.init.constant_(self.prior_ln_var, 0.)            

        self.dp = False
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 1:
            self.dp = True
            print('using %d GPUs' % num_gpus)
        
        self.flow_core = MaskedGraphAF(self.node_masks, self.adj_masks, 
                                       self.link_prediction_index, 
                                       num_flow_layer = 6,
                                       graph_size=self.max_size,
                                       num_node_type=self.node_dim,
                                       num_edge_type=self.bond_dim)
        if self.dp:
            self.flow_core = nn.DataParallel(self.flow_core)

    def forward(self, temperature=0.75, mute=False):
        #TODO: add dropout/normalize

        self.eval()
        with torch.no_grad():
            num2bond =  self.bond_decoder_m
            num2bond_symbol =  {0: '=', 1: '==', 2: '==='}
            # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            num2atom = self.atom_decoder_m
            num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4:'P', 5:'S', 6:'Cl', 7:'Br', 8:'I'}


            prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(), 
                                        temperature * torch.ones([self.node_dim]).cuda())
            prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(), 
                                        temperature * torch.ones([self.bond_dim]).cuda())

            ## Each: (B, V, N), (B, E, V, V)
            cur_node_features = torch.zeros([self.batch_size, self.max_size, self.node_dim]).cuda()
            cur_adj_features = torch.zeros([self.batch_size, self.bond_dim, self.max_size, self.max_size]).cuda()

            #mol_size = mol.GetNumAtoms()

            is_continue = [True] * self.batch_size
            total_resample = [0] * self.batch_size
            each_node_resample = np.zeros([self.batch_size,self.max_size])
            for b in range(self.batch_size):
                rw_mol = Chem.RWMol() # editable mol
                mol = None
                for i in range(self.max_size):
                    if not is_continue[b]:
                        break
                    if i < self.edge_unroll:
                        edge_total = i # edge to sample for current node
                        start = 0
                    else:
                        edge_total = self.edge_unroll
                        start = i - self.edge_unroll
                    # first generate node
                    ## reverse flow
                    latent_node = prior_node_dist.sample().view(1, -1) #(1, 9)

                    if self.dp:
                        latent_node = self.flow_core.module.reverse(cur_node_features[b:b+1], cur_adj_features[b:b+1], 
                                                latent_node, mode=0).view(-1) # (9, )
                    else:
                        latent_node = self.flow_core.reverse(cur_node_features[b:b+1], cur_adj_features[b:b+1], 
                                                latent_node, mode=0).view(-1) # (9, )
                    ## node/adj postprocessing
                    #print(latent_node.shape) #(38, 9)
                    feature_id = torch.argmax(latent_node).item()
                    #print(num2symbol[feature_id])
                    cur_node_features[b, i, feature_id] = 1.0
                    cur_adj_features[b, :, i, i] = 1.0
                    rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))
                    
                    # then generate edges
                    if i == 0:
                        is_connect = True
                    else:
                        is_connect = False
                    #cur_mol_size = mol.GetNumAtoms
                    for j in range(edge_total):
                        valid = False
                        resample_edge = 0
                        invalid_bond_type_set = set()
                        while not valid:
                            #TODO: add cache. Some atom can not get the right edge type and is stuck in the loop
                            #TODO: add cache. invalid bond set
                            if len(invalid_bond_type_set) < 3 and resample_edge <= 50: # haven't sampled all possible bond type or is not stuck in the loop
                                latent_edge = prior_edge_dist.sample().view(1, -1) #(1, 4)
                                if self.dp:
                                    latent_edge = self.flow_core.module.reverse(cur_node_features[b:b+1], cur_adj_features[b:b+1], latent_edge, 
                                                mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                                else:
                                    latent_edge = self.flow_core.reverse(cur_node_features[b:b+1], cur_adj_features[b:b+1], latent_edge, 
                                                mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                                edge_discrete_id = torch.argmax(latent_edge).item()
                            else:
                                if not mute:
                                    print('have tried all possible bond type, use virtual bond.')
                                assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                                edge_discrete_id = 0 # we have no choice but to choose not to add edge between (i, j+start)
                            cur_adj_features[b, edge_discrete_id, i, j + start] = 1.0
                            cur_adj_features[b, edge_discrete_id, j + start, i] = 1.0
                            if edge_discrete_id == 0: # virtual edge
                                valid = True
                            else: #single/double/triple bond
                                rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])                                                   
                                #TODO: check valency 
                                valid = env.check_valency(rw_mol)
                                if valid:
                                    is_connect = True
                                    #print(num2bond_symbol[edge_discrete_id])
                                else: #backtrack
                                    rw_mol.RemoveBond(i, j + start)
                                    cur_adj_features[b, edge_discrete_id, i, j + start] = 0.0
                                    cur_adj_features[b, edge_discrete_id, j + start, i] = 0.0
                                    total_resample[b] += 1.0
                                    each_node_resample[b][i] += 1.0
                                    resample_edge += 1

                                    invalid_bond_type_set.add(edge_discrete_id)
                    if is_connect: # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                        is_continue[b] = True
                    else:
                        is_continue[b] = False

        ## (B, E, V, V) to (B, V, V, E)

        cur_adj_features = cur_adj_features.permute(0,2,3,1)

        print(cur_adj_features.size(), cur_node_features.size())

        return cur_adj_features, cur_node_features

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).cuda()
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out
    
    #TODO: generate graph, finish this part
    def generate(self, temperature=0.75, mute=False, max_atoms=48, cnt=None):
        """
        inverse flow to generate molecule
        Args: 
            temp: temperature of normal distributions, we sample from (0, temp^2 * I)
        """
        generate_start_t = time()
        with torch.no_grad():
            num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
            num2bond_symbol =  {0: '=', 1: '==', 2: '==='}
            # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4:15, 5:16, 6:17, 7:35, 8:53}
            num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4:'P', 5:'S', 6:'Cl', 7:'Br', 8:'I'}


            prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(), 
                                        temperature * torch.ones([self.node_dim]).cuda())
            prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(), 
                                        temperature * torch.ones([self.bond_dim]).cuda())

            cur_node_features = torch.zeros([1, max_atoms, self.node_dim]).cuda()
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms]).cuda()

            rw_mol = Chem.RWMol() # editable mol
            mol = None
            #mol_size = mol.GetNumAtoms()

            is_continue = True
            total_resample = 0
            each_node_resample = np.zeros([max_atoms])
            for i in range(max_atoms):
                if not is_continue:
                    break
                if i < self.edge_unroll:
                    edge_total = i # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll
                # first generate node
                ## reverse flow
                latent_node = prior_node_dist.sample().view(1, -1) #(1, 9)
                if self.dp:
                    latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (9, )
                else:
                    latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (9, )
                ## node/adj postprocessing
                #print(latent_node.shape) #(38, 9)
                feature_id = torch.argmax(latent_node).item()
                #print(num2symbol[feature_id])
                cur_node_features[0, i, feature_id] = 1.0
                cur_adj_features[0, :, i, i] = 1.0
                rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))
                

                # then generate edges
                if i == 0:
                    is_connect = True
                else:
                    is_connect = False
                #cur_mol_size = mol.GetNumAtoms
                for j in range(edge_total):
                    valid = False
                    resample_edge = 0
                    invalid_bond_type_set = set()
                    while not valid:
                        #TODO: add cache. Some atom can not get the right edge type and is stuck in the loop
                        #TODO: add cache. invalid bond set
                        if len(invalid_bond_type_set) < 3 and resample_edge <= 50: # haven't sampled all possible bond type or is not stuck in the loop
                            latent_edge = prior_edge_dist.sample().view(1, -1) #(1, 4)
                            if self.dp:
                                latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                            else:
                                latent_edge = self.flow_core.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                            edge_discrete_id = torch.argmax(latent_edge).item()
                        else:
                            if not mute:
                                print('have tried all possible bond type, use virtual bond.')
                            assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                            edge_discrete_id = 3 # we have no choice but to choose not to add edge between (i, j+start)
                        cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                        cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                        if edge_discrete_id == 3: # virtual edge
                            valid = True
                        else: #single/double/triple bond
                            rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])                                                   
                            valid = env.check_valency(rw_mol)
                            if valid:
                                is_connect = True
                                #print(num2bond_symbol[edge_discrete_id])
                            else: #backtrack
                                rw_mol.RemoveBond(i, j + start)
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                total_resample += 1.0
                                each_node_resample[i] += 1.0
                                resample_edge += 1

                                invalid_bond_type_set.add(edge_discrete_id)
                if is_connect: # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                    is_continue = True
                    mol = rw_mol.GetMol()
                
                else:
                    is_continue = False

            #mol = rw_mol.GetMol() # mol backup
            assert mol is not None, 'mol is None...'

            #final_valid = check_valency(mol)
            final_valid = env.check_chemical_validity(mol)            
            assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'            

            final_mol = env.convert_radical_electrons_to_hydrogens(mol)
            smiles = Chem.MolToSmiles(final_mol, isomericSmiles=True)
            assert '.' not in smiles, 'warning: use is_connect to check stop action, but the final molecule is disconnected!!!'

            final_mol = Chem.MolFromSmiles(smiles)


            #mol = convert_radical_electrons_to_hydrogens(mol)
            num_atoms = final_mol.GetNumAtoms()
            num_bonds = final_mol.GetNumBonds()

            pure_valid = 0
            if total_resample == 0:
                pure_valid = 1.0
            if not mute:
                cnt = str(cnt) if cnt is not None else ''
                print('smiles%s: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (cnt, smiles, num_atoms, num_bonds, total_resample, time()-generate_start_t))
            return smiles, pure_valid, num_atoms

    def batch_generate(self, batch_size=100, temperature=0.75, mute=False):
        """
        inverse flow to generate one batch molecules
        Args: 
            temp: temperature of normal distributions, we sample from (0, temp^2 * I)
            mute: do not print some messages
        Returns:
            generated mol represented by smiles, valid rate (without check)
        """

        generate_start_t = time()
        with torch.no_grad():
            num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
            num2bond_symbol =  {0: '=', 1: '==', 2: '==='}
            # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4:15, 5:16, 6:17, 7:35, 8:53}
            num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4:'P', 5:'S', 6:'Cl', 7:'Br', 8:'I'}


            prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(), 
                                        temperature * torch.ones([self.node_dim]).cuda())
            prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(), 
                                        temperature * torch.ones([self.bond_dim]).cuda())

            cur_node_features = torch.zeros([batch_size, self.max_size, self.node_dim]).cuda()
            cur_adj_features = torch.zeros([batch_size, self.bond_dim, self.max_size, self.max_size]).cuda()

            rw_mol = Chem.RWMol() # editable mol
            mol = None
            #mol_size = mol.GetNumAtoms()

            is_continue = [True] * batch_size
            total_resample = [0] * batch_size
            each_node_resample = np.zeros([batch_size,self.max_size])
            for i in range(self.max_size):
                if not is_continue:
                    break
                if i < self.edge_unroll:
                    edge_total = i # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll
                # first generate node
                ## reverse flow
                latent_node = prior_node_dist.sample().view(1, -1) #(1, 9)
                if self.dp:
                    latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (9, )
                else:
                    latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (9, )
                ## node/adj postprocessing
                #print(latent_node.shape) #(38, 9)
                feature_id = torch.argmax(latent_node).item()
                #print(num2symbol[feature_id])
                cur_node_features[0, i, feature_id] = 1.0
                cur_adj_features[0, :, i, i] = 1.0
                rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))
                

                # then generate edges
                if i == 0:
                    is_connect = True
                else:
                    is_connect = False
                #cur_mol_size = mol.GetNumAtoms
                for j in range(edge_total):
                    valid = False
                    resample_edge = 0
                    invalid_bond_type_set = set()
                    while not valid:
                        #TODO: add cache. Some atom can not get the right edge type and is stuck in the loop
                        #TODO: add cache. invalid bond set
                        if len(invalid_bond_type_set) < 3 and resample_edge <= 50: # haven't sampled all possible bond type or is not stuck in the loop
                            latent_edge = prior_edge_dist.sample().view(1, -1) #(1, 4)
                            if self.dp:
                                latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                            else:
                                latent_edge = self.flow_core(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                            edge_discrete_id = torch.argmax(latent_edge).item()
                        else:
                            if not mute:
                                print('have tried all possible bond type, use virtual bond.')
                            assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                            edge_discrete_id = 3 # we have no choice but to choose not to add edge between (i, j+start)
                        cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                        cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                        if edge_discrete_id == 3: # virtual edge
                            valid = True
                        else: #single/double/triple bond
                            rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])                                                   
                            #TODO: check valency 
                            valid = env.check_valency(rw_mol)
                            if valid:
                                is_connect = True
                                #print(num2bond_symbol[edge_discrete_id])
                            else: #backtrack
                                rw_mol.RemoveBond(i, j + start)
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                total_resample += 1.0
                                each_node_resample[i] += 1.0
                                resample_edge += 1

                                invalid_bond_type_set.add(edge_discrete_id)
                if is_connect: # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                    is_continue = True
                    mol = rw_mol.GetMol()
                
                else:
                    is_continue = False

            #mol = rw_mol.GetMol() # mol backup
            assert mol is not None, 'mol is None...'
            mol = env.convert_radical_electrons_to_hydrogens(mol)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()

            smiles = Chem.MolToSmiles(mol)
            assert '.' not in smiles, 'warning: use is_connect to check stop action, but the final molecule is disconnected!!!'

            final_valid = env.check_valency(mol)
            assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'

            pure_valid = 0
            if total_resample == 0:
                pure_valid = 1.0
            if not mute:
                print('smiles: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (smiles, num_atoms, num_bonds, total_resample, time()-generate_start_t))
            return smiles, pure_valid   

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=12):
        """
        Args:
            max node unroll: maximal number of nodes in molecules to be generated (default: 38)
            max edge unroll: maximal number of edges to predict for each generated nodes (default: 12, calculated from zink250K data)
        Returns:
            node_masks: node mask for each step
            adj_masks: adjacency mask for each step
            is_node_update_mask: 1 indicate this step is for updating node features
            flow_core_edge_mask: get the distributions we want to model in adjacency matrix
        """
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (max_edge_unroll))
        num_mask_edge = int(num_masks - max_node_unroll)

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).byte()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).byte()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).byte()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).byte()        
        #is_node_update_masks = torch.zeros([num_masks]).byte()

        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()


        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).byte()

        #masks_edge = dict()
        cnt = 0
        cnt_node = 0
        cnt_edge = 0
        for i in range(max_node_unroll):
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            #is_node_update_masks[cnt] = 1
            cnt += 1
            cnt_node += 1

            edge_total = 0
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
            for j in range(edge_total):
                if j == 0:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node-1].clone()
                    adj_masks2[cnt_edge][i,i] = 1
                else:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge-1].clone()
                    adj_masks2[cnt_edge][i, start + j -1] = 1
                    adj_masks2[cnt_edge][start + j -1, i] = 1
                cnt += 1
                cnt_edge += 1
        assert cnt == num_masks, 'masks cnt wrong'
        assert cnt_node == max_node_unroll, 'node masks cnt wrong'
        assert cnt_edge == num_mask_edge, 'edge masks cnt wrong'

    
        cnt = 0
        for i in range(max_node_unroll):
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
        
            for j in range(edge_total):
                link_prediction_index[cnt][0] = start + j
                link_prediction_index[cnt][1] = i
                cnt += 1
        assert cnt == num_mask_edge, 'edge mask initialize fail'


        for i in range(max_node_unroll):
            if i == 0:
                continue
            if i < max_edge_unroll:
                start = 0
                end = i
            else:
                start = i - max_edge_unroll
                end = i 
            flow_core_edge_masks[i][start:end] = 1

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks = nn.Parameter(node_masks, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)
        
        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks