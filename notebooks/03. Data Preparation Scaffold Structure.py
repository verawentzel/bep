from rdkit.Chem import AllChem
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

# Import online Data Frame
folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Dataframe met kernwaarden aanmaken
df_complete = pd.read_csv(f"{folder}v20.data.final_summary.txt", sep='\t')
df_scaffold_split = df_complete[[ 'cpd_smiles','apparent_ec50_umol']]

smiles_list = df_scaffold_split['cpd_smiles']
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
scaffolds = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
scaffold_smiles = [Chem.MolToSmiles(scaffold) for scaffold in scaffolds]
print(df_scaffold_split.head())

scaffold_smiles = list(map(str, scaffold_smiles))
#df_scaffold_split.loc[:, 'scaffold_smiles'] = scaffold_smiles


# Fingerprint aanmaken
molecule_scaffold = [Chem.MolFromSmiles(smile) for smile in scaffold_smiles]
ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024) for molecule in molecule_scaffold]
#ecfp_bit_vectors = [[int(bit) for bit in keys.ToBitString()] for keys in ecfp]

def get_unique_scaffolds(scaffold_smiles):
    # initialize a null list
    unique_scaffolds = []
    location=[]

    # traverse for all elements
    for scaffold in scaffold_smiles:
        # check if exists in unique_list or not
        if scaffold not in unique_scaffolds:
            unique_scaffolds.append(scaffold)
    return unique_scaffolds

list_unique_scaffolds=get_unique_scaffolds(scaffold_smiles)
print(len(list_unique_scaffolds))
print(len(mols))

from collections import defaultdict

res = defaultdict(list)
for ele in scaffold_smiles:
    res[ele].append(ele)
print("similar grouped dictionary : " + str(dict(res)))

def groepeer_locaties(lijst):
    locaties = defaultdict(list)
    for index, item in enumerate(lijst):
        if lijst.count(item) > 1:
            locaties[item].append(index)
    return dict(locaties)

resultaat = groepeer_locaties(scaffold_smiles)
print(resultaat)

count_list = []
seen_smiles = []
index=0

for smile in scaffold_smiles:
    if smile in seen_smiles:
        seen_smiles.append(scaffold_smiles[smile])
        index +=1
        seen_smiles[smile] += 1
        count_list.append(seen_smiles[smile])
    else:
        seen_smiles[smile] = 0
        count_list.append(seen_smiles[smile])

print(count_list)

