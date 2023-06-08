from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef
#from sklearn.externals import joblib
import pandas as pd

folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'
df_large = pd.read_csv(f"{folder}v20.data.final_summary.txt", sep='\t')

smiles = df_large['cpd_smiles']
mols = [Chem.MolFromSmiles(s) for s in smiles]
y = df_large['ec50_molair']

fp = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
print(fp)
def rdkit_numpy_convert(fp):
    output = []
    for f in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)

x = rdkit_numpy_convert(fp)
print(x.shape)

dataset_balance = sum(y)/len(y)
print(dataset_balance)