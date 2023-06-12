import numpy as nm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem as dc
import torch
import torch_geometric
import pytorch_lightning
import jax

#from rdkit.Chem.Scaffolds import ScaffoldSplitter
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

# Import online Data Frame
folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Dataframe met kernwaarden aanmaken
df_complete = pd.read_csv(f"{folder}v20.data.final_summary.txt", sep='\t')
smiles_list = df_complete['cpd_smiles']
y = nm.array(df_complete['ec50_mol'].values)
mols=[]
print(df_complete['ec50_mol'].isnull().sum())


for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    mols.append(mol)

dataset = dc.data.DiskDataset.from_numpy(X=mols,y=y,ids=mol)
scaffoldsplitter = dc.splits.ScaffoldSplitter()
train,test = scaffoldsplitter.train_test_split(dataset)
train


#core = MurckoScaffold.GetScaffoldForMol(mol)
#print('m=', Chem.MolToSmiles(core))

# Functie om de scaffold van een molecuul te berekenen
#def calculate_scaffold(smiles):
 #   mol = Chem.MolFromSmiles(smiles)
  #  scaffold = MurckoScaffold.GetScaffoldForMol(mol)
   # scaffold_smiles = Chem.MolToSmiles(scaffold)
    #return scaffold_smiles

# Lijst met scaffolds van d e moleculen
#scaffolds = [calculate_scaffold(smiles) for smiles in smiles_list]


# Train-test split op basis van de scaffolds
#splitter = ScaffoldSplitter()
#train_scaffolds, test_scaffolds = train_test_split(scaffolds, test_size=0.2, random_state=42)

# Train- en testdatasets op basis van de scaffoldsplitsing
#x_train = [smiles for smiles, scaffold in zip(smiles_list, scaffolds) if scaffold in train_scaffolds]
#x_test = [smiles for smiles, scaffold in zip(smiles_list, scaffolds) if scaffold in test_scaffolds]
