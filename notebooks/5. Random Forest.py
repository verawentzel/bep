import numpy as nm
import matplotlib as mtp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import random
from tqdm.auto import tqdm
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn import preprocessing as pre

folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

df_large = pd.read_csv(f"{folder}v20.data.curves_post_qc_419.txt", sep='\t')
df_summary = df_large[[ 'master_cpd_id','apparent_ec50_umol']]

df_smiles = pd.read_csv(f"{folder}v20.meta.per_compound.txt", sep="\t")
extracted_col = df_smiles[["master_cpd_id","cpd_smiles"]]

df_all = pd.merge(df_summary, extracted_col, on='master_cpd_id', how='left')
df_summary_sorted = df_all.sort_values(by=['apparent_ec50_umol'])
df_summary_sorted.to_csv(f"{folder}v20.data.final_summary.txt", sep='\t', index=False)

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


smiles_column = df_summary_sorted['cpd_smiles']

# Call the mol_descriptor function with the 'cpd_smiles' values
descriptors = mol_descriptor(smiles_column)

# Add the descriptors to your existing DataFrame
df_summary_sorted[['TPSA', 'MolLogP', 'MolWt', 'FpDensityMorgan2', 'HeavyAtomMolWt',
               'MaxPartialCharge', 'MinPartialCharge', 'NumRadicalElectrons',
               'NumValenceElectrons', 'CalcFractionCSP3', 'CalcNumRings',
               'CalcNumRotatableBonds', 'CalcNumLipinskiHBD', 'CalcNumLipinskiHBA',
               'CalcNumHeterocycles', 'CalcNumHeavyAtoms', 'CalcNumAromaticRings',
               'CalcNumAtoms', 'qed']] = descriptors
df_summary_sorted['ec50_mol'] = df_summary_sorted['apparent_ec50_umol'] / 1000000
df_summary_sorted['ec50_molair'] = df_summary_sorted['ec50_mol']/ df_summary_sorted['MolWt']
df_summary_sorted['molecule']=df_summary_sorted['cpd_smiles'].apply(lambda x: Chem.MolFromSmiles(x))
df_summary_sorted['ECFP']=df_summary_sorted['molecule'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024))
df_summary_sorted['ECFP'] = df_summary_sorted['ECFP'].apply(lambda x: int(x.ToBitString(),2))



df_summary_sorted.to_csv(f"{folder}v20.data.full_data_summary.txt", sep='\t', index=False)

#######################################################################
