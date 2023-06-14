import numpy as nm
import pandas as pd
from rdkit.Chem import rdFMCS
from rdkit import Chem
import deepchem as dc
#from rdkit.Chem.Scaffolds import ScaffoldSplitter
#from rdkit.Chem.Scaffolds import MurckoScaffold
#from sklearn.model_selection import train_test_split

# Import online Data Frame
folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Dataframe met kernwaarden aanmaken
df_complete = pd.read_csv(f"{folder}v20.data.final_summary.txt", sep='\t')
smiles_list = df_complete['cpd_smiles']
y = nm.array(df_complete['ec50_mol'].values)

mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
scaffolds = [Chem.MolToSmiles(rdFMCS.FindMCS([mol])) for mol in mols]

dataset = dc.data.NumpyDataset(X=nm.array(mols), y=y, ids=nm.array(df_complete['master_cpd_id']))
scaffoldsplitter = dc.splits.ScaffoldSplitter()
train_dataset, valid_dataset, test_dataset = scaffoldsplitter.train_valid_test_split(dataset)

print('trainset', train_dataset)

#for smiles in smiles_list:
  #  mol = Chem.MolFromSmiles(smiles)
   # mols.append(mol)

#dataset = dc.data.DiskDataset.from_numpy(X=mols,y=y,ids=mol)
#scaffoldsplitter = dc.splits.ScaffoldSplitter()
#train,test = scaffoldsplitter.train_test_split(dataset)
#train


#core = MurckoScaffold.GetScaffoldForMol(mols)
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
#splitter = dc.ScaffoldSplitter()
#train_scaffolds, test_scaffolds = train_test_split(scaffolds, test_size=0.2, random_state=42)
