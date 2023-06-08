from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold

n_splits = 5

# Import online Data Frame
folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Dataframe met kernwaarden aanmaken
df_large = pd.read_csv(f"{folder}v20.data.full_data_summary.txt", sep='\t')
smiles=df_large['cpd_smiles']
df = pd.DataFrame({'Fingerprint': df_large['cpd_smiles'], 'Target': df_large['ec50_molair']})

# Functie om de scaffold van een molecuul te bepalen
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles

# Toevoegen van de Scaffold-kolom aan het dataframe
df['Scaffold'] = smiles.apply(get_scaffold)

# Unieke scaffolds bepalen
unique_scaffolds = df['Scaffold'].unique()

# Split van de unieke scaffolds in train en test
kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
for train_index, test_index in kf.split(unique_scaffolds):
    train_scaffolds = unique_scaffolds[train_index]
    test_scaffolds = unique_scaffolds[test_index]

# Filteren van de gegevens op basis van de train en test scaffolds
def get_ecfp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return list(ecfp)

train_data = df[df['Scaffold'].isin(train_scaffolds)]
train_data = train_data.apply(get_ecfp)

test_data = df[df['Scaffold'].isin(test_scaffolds)]
test_data = train_data.apply(get_ecfp)

# Verwijderen van de scaffold-kolom
train_data = train_data.drop('Scaffold', axis=1)
test_data = test_data.drop('Scaffold', axis=1)
