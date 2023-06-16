from rdkit.Chem import AllChem
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

# Import online Data Frame
folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Dataframe met kernwaarden aanmaken
df_complete = pd.read_csv(f"{folder}v20.data.final_summary.txt", sep='\t')
df_scaffold_split = df_complete[[ 'cpd_smiles','apparent_ec50_umol','MolWt']]

smiles_list = df_scaffold_split['cpd_smiles']
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
scaffolds = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
scaffold_smiles = [Chem.MolToSmiles(scaffold) for scaffold in scaffolds]
print(df_scaffold_split.head())

scaffold_smiles = list(map(str, scaffold_smiles))

# Fingerprint aanmaken
molecule_scaffold = [Chem.MolFromSmiles(smile) for smile in scaffold_smiles]
ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024) for molecule in molecule_scaffold]

def get_unique_scaffolds(scaffold_smiles):
    unique_scaffolds = []
    for scaffold in scaffold_smiles:
        if scaffold not in unique_scaffolds:
            unique_scaffolds.append(scaffold)
    return unique_scaffolds

list_unique_scaffolds=get_unique_scaffolds(scaffold_smiles)

location_scaffolds = defaultdict(list)
for scaffold in scaffold_smiles:
    location_scaffolds[scaffold].append(scaffold)
print("similar grouped dictionary : " + str(dict(location_scaffolds)))

def location_similar_scaffolds(lijst):
    locations = defaultdict(list)
    for index, item in enumerate(lijst):
        if lijst.count(item) > 1:
            locations[item].append(index)
    return dict(locations)

locations_similar_scaffolds = location_similar_scaffolds(scaffold_smiles)
print(locations_similar_scaffolds)

df_scaffolds_grouped = pd.DataFrame(columns=['Compound', 'Scaffold', 'Recurring'])
rows=[]
symbol_mapping = {}  # Scaffold symbool dictionary
symbol_counter = 0  # Aantal unieke scaffolds

for i, compound in enumerate(smiles_list):
    scaffold = scaffold_smiles[i]

    if scaffold in locations_similar_scaffolds:
        if scaffold not in symbol_mapping:
            symbol_counter += 1
            symbol_mapping[scaffold] = chr(64 + symbol_counter)

        recurring_symbol = symbol_mapping[scaffold]
        recurring = True
    else:
        recurring_symbol = 0
        recurring = False
    row = [compound, scaffold, recurring_symbol]
    rows.append(row)

df_scaffolds_grouped = pd.DataFrame(rows, columns=['cpd_smiles', 'spd_scaffold', 'recurring_scaffold'])

#Toevoegen van extra data die nodig is voor de modellen
df_scaffolds_grouped.insert(0,'apparent_ec50_umol',df_scaffold_split['apparent_ec50_umol'])
df_scaffolds_grouped.insert(1,'MolWt',df_scaffold_split['MolWt'])

df_scaffolds_grouped.to_csv(f"{folder}scaffold_split.txt", sep='\t', index=False)

