import itertools
import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
import copy
import networkx as nx
import math
import random
import time
#import matplotlib.pyplot as plt
import csv

from contextlib import contextmanager
import sys, os

# most of the codes here are taken from GCPN's open-source implementation
'''
Environment Usage
plogp: penalized_logp(mol)
logp: MolLogP(mol)
qed: qed(mol)
sa: calculateScore(mol)
mw: rdMolDescriptors.CalcExactMolWt(mol)
'''

def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        print('converting radical electrons to H')
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def check_chemical_validity(mol):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.
    :return: True if chemically valid, False otherwise
    """
    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False


def get_final_smiles(mol):
    """
    Returns a SMILES of the final molecule. Converts any radical
    electrons into hydrogens. Works only if molecule is valid
    :return: SMILES
    """
    m = convert_radical_electrons_to_hydrogens(mol)
    return Chem.MolToSmiles(m, isomericSmiles=True)


def get_final_mol(mol):
    """
    Returns a rdkit mol object of the final molecule. Converts any radical
    electrons into hydrogens. Works only if molecule is valid
    :return: SMILES
    """
    m = convert_radical_electrons_to_hydrogens(mol)
    return m


def add_hydrogen(mol):
    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    return Chem.MolFromSmiles(s)

def steric_strain_filter(mol, cutoff=0.82, max_attempts_embed=20, max_num_iters=200):
    """
    Flags molecules based on a steric energy cutoff after max_num_iters
    iterations of MMFF94 forcefield minimization. Cutoff is based on average
    angle bend strain energy of molecule
    :param mol: rdkit mol object
    :param cutoff: kcal/mol per angle . If minimized energy is above this
    threshold, then molecule fails the steric strain filter
    :param max_attempts_embed: number of attempts to generate initial 3d
    coordinates
    :param max_num_iters: number of iterations of forcefield minimization
    :return: True if molecule could be successfully minimized, and resulting
    energy is below cutoff, otherwise False
    """
    # check for the trivial cases of a single atom or only 2 atoms, in which
    # case there is no angle bend strain energy (as there are no angles!)
    if mol.GetNumAtoms() <= 2:
        return True

    # make copy of input mol and add hydrogens
    m = copy.deepcopy(mol)
    m_h = Chem.AddHs(m)

    # generate an initial 3d conformer
    try:
        flag = AllChem.EmbedMolecule(m_h, maxAttempts=max_attempts_embed)
        if flag == -1:
            # print("Unable to generate 3d conformer")
            return False
    except: # to catch error caused by molecules such as C=[SH]1=C2OC21ON(N)OC(=O)NO
        # print("Unable to generate 3d conformer")
        return False

    # set up the forcefield
    AllChem.MMFFSanitizeMolecule(m_h)
    if AllChem.MMFFHasAllMoleculeParams(m_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(m_h)
        try:    # to deal with molecules such as CNN1NS23(=C4C5=C2C(=C53)N4Cl)S1
            ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
        except:
            # print("Unable to get forcefield or sanitization error")
            return False
    else:
        # print("Unrecognized atom type")
        return False

    # minimize steric energy
    try:
        ff.Minimize(maxIts=max_num_iters)
    except:
        # print("Minimization error")
        return False

    # ### debug ###
    # min_e = ff.CalcEnergy()
    # print("Minimized energy: {}".format(min_e))
    # ### debug ###

    # get the angle bend term contribution to the total molecule strain energy
    mmff_props.SetMMFFBondTerm(False)
    mmff_props.SetMMFFAngleTerm(True)
    mmff_props.SetMMFFStretchBendTerm(False)
    mmff_props.SetMMFFOopTerm(False)
    mmff_props.SetMMFFTorsionTerm(False)
    mmff_props.SetMMFFVdWTerm(False)
    mmff_props.SetMMFFEleTerm(False)

    ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)

    min_angle_e = ff.CalcEnergy()
    # print("Minimized angle bend energy: {}".format(min_angle_e))

    # find number of angles in molecule
    # from molecule... This is too hacky
    num_atoms = m_h.GetNumAtoms()
    atom_indices = range(num_atoms)
    angle_atom_triplets = itertools.permutations(atom_indices, 3)  # get all
    # possible 3 atom indices groups. Currently, each angle is represented by
    #  2 duplicate groups. Should remove duplicates here to be more efficient
    double_num_angles = 0
    for triplet in list(angle_atom_triplets):
        if mmff_props.GetMMFFAngleBendParams(m_h, *triplet):
            double_num_angles += 1
    num_angles = double_num_angles / 2  # account for duplicate angles

    # print("Number of angles: {}".format(num_angles))

    avr_angle_e = min_angle_e / num_angles

    # print("Average minimized angle bend energy: {}".format(avr_angle_e))

    # ### debug ###
    # for i in range(7):
    #     termList = [['BondStretch', False], ['AngleBend', False],
    #                 ['StretchBend', False], ['OopBend', False],
    #                 ['Torsion', False],
    #                 ['VdW', False], ['Electrostatic', False]]
    #     termList[i][1] = True
    #     mmff_props.SetMMFFBondTerm(termList[0][1])
    #     mmff_props.SetMMFFAngleTerm(termList[1][1])
    #     mmff_props.SetMMFFStretchBendTerm(termList[2][1])
    #     mmff_props.SetMMFFOopTerm(termList[3][1])
    #     mmff_props.SetMMFFTorsionTerm(termList[4][1])
    #     mmff_props.SetMMFFVdWTerm(termList[5][1])
    #     mmff_props.SetMMFFEleTerm(termList[6][1])
    #     ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
    #     print('{0:>16s} energy: {1:12.4f} kcal/mol'.format(termList[i][0],
    #                                                  ff.CalcEnergy()))
    # ## end debug ###

    if avr_angle_e < cutoff:
        return True
    else:
        return False

### YES/NO filters ###
def zinc_molecule_filter(mol):
    """
    Flags molecules based on problematic functional groups as
    provided set of ZINC rules from
    http://blaster.docking.org/filtering/rules_default.txt.
    :param mol: rdkit mol object
    :return: Returns True if molecule is okay (ie does not match any of
    therules), False if otherwise
    """
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)