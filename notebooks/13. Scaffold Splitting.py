import matplotlib.pyplot as plt
import numpy as nm
import pandas as pd
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''ECFP RANDOM FOREST'''

''' Dit is het bestand voor het laatste, meest up-to-date
ECFP random forest model dat ik heb gemaakt en toegepast
heb op de uiteindelijke cellijn.'''

folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Import data
complete_df = pd.read_csv(f"{folder}scaffold_split.txt", sep="\t")
complete_df.fillna(complete_df.mean(), inplace=True)

# Fingerprint aanmaken
molecules = [Chem.MolFromSmiles(smile) for smile in complete_df['cpd_smiles'].tolist()]
ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule,2,nBits=1024) for molecule in molecules]
complete_df['ecfp_bit_vectors'] = [[int(bit) for bit in keys.ToBitString()] for keys in ecfp]
complete_df['ECFP'] = [''.join(str(value) for value in row) for row in complete_df['ecfp_bit_vectors']]

# Doelvariabele transformeren & limieten stellen dataframe
complete_df['ec50_mol'] = complete_df['apparent_ec50_umol'] / 1000000
complete_df['ec50_mol']=complete_df['ec50_mol'].replace(0, 1e-10)
complete_df['ec50_molair'] = complete_df['ec50_mol']/ complete_df['MolWt']
complete_df['ec50_molair_transformed'] = -nm.log10(complete_df['ec50_molair'])
condition = (complete_df['ec50_molair_transformed'] < 1 ) | (complete_df['ec50_molair_transformed'] > 10)
complete_df=complete_df[~condition]

# Dependent & Independent variable
x = nm.array(complete_df['ecfp_bit_vectors'].tolist())
y = complete_df['ec50_molair_transformed'].values
z = complete_df['recurring_scaffold'].values

df_unique_scaffolds = complete_df[complete_df['recurring_scaffold'] == 0]
df_grouped_scaffolds = complete_df[complete_df['recurring_scaffold'] > 0]

x_grouped_scaffolds = df_grouped_scaffolds['cpd_smiles']
y_grouped_scaffolds = df_grouped_scaffolds['ec50_molair_transformed']

from sklearn.model_selection import GroupKFold

gkf_cv = GroupKFold(n_splits=2)

train_locations = pd.DataFrame()
test_locations = pd.DataFrame()

for split, (ix_train, ix_test) in enumerate(gkf_cv.split(x_grouped_scaffolds, y_grouped_scaffolds, groups=df_grouped_scaffolds['recurring_scaffold'])):
    print(f'SPLIT {split+1}')
    print(f'TRAIN INDEXES: {ix_train}, TEST INDEXES: {ix_test}\n')
    ix_train_list = nm.array(ix_train).tolist()
    print(ix_train_list)
    ix_test_list = nm.array(ix_test).tolist()
    print(ix_test_list)
    for i in ix_train_list:
        x_train = x_grouped_scaffolds.iloc[ix_train_list[i]]
        y_train = y_grouped_scaffolds.iloc[ix_train_list[i]]
        train_locations.append(pd.DataFrame({'cpd_smiles': x_train, 'ec50_molair_transformed': y_train}), ignore_index=True)
    for j in ix_test_list:
        x_test = x_grouped_scaffolds.iloc[ix_test_list[j]]
        y_test = y_grouped_scaffolds.iloc[ix_test_list[j]]
        test_locations.append(pd.DataFrame({'cpd_smiles': x_test, 'ec50_molair_transformed': y_test}), ignore_index=True)

#total_length_compounds = len(complete_df)
#print(total_length_compounds)
#size_train_set = total_length_compounds * 0.8

#shuffled_df = df_unique_scaffolds.sample(frac=1)
#x_unique_scaffolds = df_unique_scaffolds['cpd_smiles']
#y_unique_scaffolds = df_unique_scaffolds['ec50_molair_transformed']

