# -*- coding:utf-8 -*-
"""Sample training code
"""
import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# from sch import SchNetModel
# from mgcn import MGCNModel
# from mpnn import MPNNModel
# from torch.utils.data import DataLoader
# from Alchemy_dataset import TencentAlchemyDataset, batcher

parser = argparse.ArgumentParser()
parser.add_argument("--dataset1", help="Target dataset", default="")
parser.add_argument("--dataset2", help="input dataset", default="")
args = parser.parse_args()
dataset1 = args.dataset1
dataset2 = args.dataset2
dataset1= pd.read_csv(dataset1 + ".csv", skiprows=1,
                      names=['id', 'measured', 'predicted', 'SMILES'])
dataset2= pd.read_csv(dataset2 + ".csv", skiprows=1,
                      names=['id', 'measured', 'predicted', 'SMILES'])
fp_bulk = []
Tan_bulk = []
for idx, parms_x in dataset1.iterrows():
    try:
            smi_x = parms_x['SMILES']
            mol_x = Chem.MolFromSmiles(smi_x)
            AllChem.Compute2DCoords(mol_x)
            #fp1 = Chem.RDKFingerprint(mol_x)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_x, 2, 2048)
    except:
        print(smi_x + "was not valid SMILES\n")
        continue
    else:
        fp_bulk.append(fp1)
        
for idy, parms_y in dataset2.iterrows():

    try:
        smi_y = parms_y['SMILES']
        exp_solb = parms_y['measured']
        mol_y = Chem.MolFromSmiles(smi_y)
        AllChem.Compute2DCoords(mol_y)
        # fp2 = Chem.RDKFingerprint(mol_y)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_y, 2, 2048)
    except:
        print(smi_y + "was not valid SMILES\n")
        continue
    else:            
        Tan = DataStructs.BulkTanimotoSimilarity(fp2, fp_bulk)
        Tan = np.asarray(Tan)
        Tan_mean = Tan.mean()
        Tan_min = Tan.min()
        Tan_max = Tan.max()
        Tan_bulk.append([smi_y, exp_solb, Tan_mean, Tan_min, Tan_max])
        print("Tan=  %s"%(Tan))
Tan_pd = pd.DataFrame(Tan_bulk, columns=['smi','experimental_solubilty', 'smlty_mean', 'smlty_min', 'smlty_max'])
pd_file='%s_%s_smlty.csv'%(args.dataset1,args.dataset2)
Tan_pd.to_csv(pd_file, index=False)
print('mean_mean= %s'%(Tan_pd['smlty_mean'].mean()))
print('max_mean= %s'%(Tan_pd['smlty_max'].mean()))




