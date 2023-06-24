import numpy as nm
import pandas as pd
from rdkit.Chem import AllChem
from tqdm.auto import tqdm
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn import preprocessing as pre
from rdkit import Chem
from rdkit.Chem import MACCSkeys


# Import online Data Frame
folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Dataframe met kernwaarden aanmaken
df_large = pd.read_csv(f"{folder}v20.data.curves_post_qc.txt", sep='\t')

# Experiment_id aangeven van de te onderzoeken cellijn
## In Kahler onderzoek zijn de mogelijke cellijnen:
## exp. 419 | exp. 305 | exp. 538 | exp. 263 | exp. 260
experiment_id = 419


df_large=df_large[df_large['experiment_id'] == experiment_id]
df_summary = df_large[[ 'master_cpd_id','apparent_ec50_umol']]

df_smiles = pd.read_csv(f"{folder}v20.meta.per_compound.txt", sep="\t")
extracted_col = df_smiles[["master_cpd_id","cpd_smiles"]]
df_summary_sorted = pd.merge(df_summary, extracted_col, on='master_cpd_id', how='left')

# Aanmaken Mol Descriptors
def mol_descriptor(smiles: list, scale: bool = True) -> nm.ndarray:
    X = []
    for smi in tqdm(smiles):
        m = Chem.MolFromSmiles(smi)
        x = nm.array([Descriptors.TPSA(m),
                      Descriptors.MolLogP(m),
                      Descriptors.MolWt(m),
                      Descriptors.FpDensityMorgan2(m),
                      Descriptors.HeavyAtomMolWt(m),
                      Descriptors.MaxPartialCharge(m),
                      Descriptors.MinPartialCharge(m),
                      Descriptors.NumRadicalElectrons(m),
                      Descriptors.NumValenceElectrons(m),
                      rdMolDescriptors.CalcFractionCSP3(m),
                      rdMolDescriptors.CalcNumRings(m),
                      rdMolDescriptors.CalcNumRotatableBonds(m),
                      rdMolDescriptors.CalcNumLipinskiHBD(m),
                      rdMolDescriptors.CalcNumLipinskiHBA(m),
                      rdMolDescriptors.CalcNumHeterocycles(m),
                      rdMolDescriptors.CalcNumHeavyAtoms(m),
                      rdMolDescriptors.CalcNumAromaticRings(m),
                      rdMolDescriptors.CalcNumAtoms(m),
                      qed(m)])
        X.append(x)

    if scale:
        return pre.MinMaxScaler().fit_transform(nm.array(X))

    return nm.array(X)


# Dataframe met mol descriptors
smiles_column = df_summary_sorted['cpd_smiles']
descriptors = mol_descriptor(smiles_column)

df_summary_sorted[['TPSA', 'MolLogP', 'MolWt', 'FpDensityMorgan2', 'HeavyAtomMolWt',
               'MaxPartialCharge', 'MinPartialCharge', 'NumRadicalElectrons',
               'NumValenceElectrons', 'CalcFractionCSP3', 'CalcNumRings',
               'CalcNumRotatableBonds', 'CalcNumLipinskiHBD', 'CalcNumLipinskiHBA',
               'CalcNumHeterocycles', 'CalcNumHeavyAtoms', 'CalcNumAromaticRings',
               'CalcNumAtoms', 'qed']] = descriptors

df_summary_sorted['ec50_mol'] = df_summary_sorted['apparent_ec50_umol'] / 1000000
df_summary_sorted['ec50_mol']=df_summary_sorted['ec50_mol'].replace(0, 1e-10)
df_summary_sorted['ec50_molair'] = df_summary_sorted['ec50_mol']/ df_summary_sorted['MolWt']
df_summary_sorted.to_csv(f"{folder}v20.data.final_summary_{experiment_id}.txt", sep='\t', index=False)

#####################################################################################

# Fingerprint Data Frame aanmaken
#df_fingerprints = df_summary_sorted[['master_cpd_id','ec50_mol', 'ec50_molair']]
#molecules = [Chem.MolFromSmiles(smiles) for smiles in df_summary_sorted['cpd_smiles'].tolist()]

## ECFP Aanmaken
#ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule,2,nBits=1024) for molecule in molecules]
#ecfp_bit_vectors = [[int(bit) for bit in keys.ToBitString()] for keys in ecfp]
#df_fingerprints['ECFP'] = ecfp_bit_vectors
#df_fingerprints.to_csv(f"{folder}v20.data.fingerprints.txt", sep='\t', index=False)

## MACCS key Aanmaken
#maccs_keys = [MACCSkeys.GenMACCSKeys(molecule) for molecule in molecules]
#maccs_bit_vectors = [[int(bit) for bit in keys.ToBitString()] for keys in maccs_keys]
#df_fingerprints['MACCS Keys'] = maccs_bit_vectors

## Conjoint key aanmaken
#def combine_lists(row):
#    return row['ECFP'] + row['MACCS Keys']
#df_fingerprints['Conjoint Keys']= df_fingerprints.apply(combine_lists, axis=1)
#df_fingerprints.to_csv(f"{folder}v20.data.fingerprints.txt", sep='\t', index=False)

#####################################################################################

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
symbol_mapping = {}
symbol_counter = 0

for i, compound in enumerate(smiles_list):
    scaffold = scaffold_smiles[i]

    if scaffold in locations_similar_scaffolds:
        if scaffold not in symbol_mapping:
            symbol_counter += 1
            symbol_mapping[scaffold] = symbol_counter

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

df_scaffolds_grouped_sorted = df_scaffolds_grouped.sort_values('recurring_scaffold', ascending=False)

df_scaffolds_grouped_sorted.to_csv(f"{folder}scaffold_split.txt", sep='\t', index=False)


