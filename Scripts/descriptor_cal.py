from rdkit import Chem
from mordred import Calculator, descriptors
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

def isnumber(x):
    # if str(x).isdigit():
    #     return x
    # else:
    #     return np.NaN
    try:
        float(str(x))
        return x
    except:
        return 'NaN'

def cal_mol(df):
    # if str(x).isdigit():
    #     return x
    # else:
    #     return np.NaN
    mols = []
    invalid = []
    for index, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            #AllChem(mol)
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
        except:
            print("%s is not a valid SMILES"%(row['SMILES']))
            invalid.append(index)
    df.drop(df.index[invalid], inplace = True)
    return mols, df


calc = Calculator(descriptors, ignore_3D=True)
# AllChem.Compute2DCoords(mol)

delaney = pd.read_csv("delaney_cross.csv", skiprows=1,
                      names=['id', 'measured', 'predicted', 'SMILES'])
# mol = Chem.MolFromSmiles('c1ccccc1')
# descriptor = calc(mol)
# print(descriptor)


# smis = delaney['SMILES']
mols, df_valid = cal_mol(delaney)
smis = df_valid['SMILES']
mols = [Chem.MolFromSmiles(smi) for smi in smis]
descrpitor_df = calc.pandas(mols)

print(descrpitor_df.applymap(isnumber))
df = descrpitor_df.applymap(isnumber)

# for key in descrpitor_df.keys():
#     df[key] = df[key].apply(lambda x: np.NaN if len(str(x))==0 else x)

df.to_csv('descriptor_delaney_cross.csv',index=None)
df_valid.to_csv('delaney_cross_valid.csv',index=None)
