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
df_large = pd.read_csv(f"{folder}v20.data.curves_post_qc_419.txt", sep='\t')
df_summary = df_large[[ 'master_cpd_id','apparent_ec50_umol']]

df_smiles = pd.read_csv(f"{folder}v20.meta.per_compound.txt", sep="\t")
extracted_col = df_smiles[["master_cpd_id","cpd_smiles"]]

#df_all = pd.merge(df_summary, extracted_col, on='master_cpd_id', how='left')
#df_summary_sorted = df_all.sort_values(by=['apparent_ec50_umol'])
df_summary_sorted = pd.merge(df_summary, extracted_col, on='master_cpd_id', how='left')

# Aanmaken Mol Descriptors
def mol_descriptor(smiles: list[str], scale: bool = True) -> nm.ndarray:
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
#df_summary_sorted['ec50_molair']=df_summary_sorted['ec50_molair'].replace(0, 1e-10)
df_summary_sorted.to_csv(f"{folder}v20.data.final_summary.txt", sep='\t', index=False)


#####################################################################################

# Fingerprint Data Frame aanmaken
df_fingerprints = df_summary_sorted[['master_cpd_id','ec50_mol', 'ec50_molair']]
molecules = [Chem.MolFromSmiles(smiles) for smiles in df_summary_sorted['cpd_smiles'].tolist()]

## ECFP Aanmaken
ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule,2,nBits=1024) for molecule in molecules]
ecfp_bit_vectors = [[int(bit) for bit in keys.ToBitString()] for keys in ecfp]
df_fingerprints['ECFP'] = ecfp_bit_vectors
df_fingerprints.to_csv(f"{folder}v20.data.fingerprints.txt", sep='\t', index=False)

## MACCS key Aanmaken
maccs_keys = [MACCSkeys.GenMACCSKeys(molecule) for molecule in molecules]
maccs_bit_vectors = [[int(bit) for bit in keys.ToBitString()] for keys in maccs_keys]
df_fingerprints['MACCS Keys'] = maccs_bit_vectors

## Conjoint key aanmaken
def combine_lists(row):
    return row['ECFP'] + row['MACCS Keys']
df_fingerprints['Conjoint Keys']= df_fingerprints.apply(combine_lists, axis=1)
df_fingerprints.to_csv(f"{folder}v20.data.fingerprints.txt", sep='\t', index=False)

#####################################################################################
