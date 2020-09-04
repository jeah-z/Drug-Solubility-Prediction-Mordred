from rdkit import Chem
# from mordred import Calculator, descriptors
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np



descriptor = pd.read_csv("descriptor.csv")

delaney = pd.read_csv("delaney.csv", skiprows=1,
                      names=['id', 'measured', 'predicted', 'SMILES'])

descriptor['Solubility'] = delaney['measured']


spearman = descriptor.corr(method='spearman')  
kendall = descriptor.corr(method='kendall') 

spearman.to_csv('spearman.csv', index=None)
kendall.to_csv('kendall.csv', index=None)