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
complete_df = pd.read_csv(f"{folder}v20.data.final_summary_260.txt", sep="\t")
complete_df.fillna(complete_df.mean(), inplace=True)

# Fingerprint aanmaken
molecules = [Chem.MolFromSmiles(smile) for smile in complete_df['cpd_smiles'].tolist()]
ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule,2,nBits=1024) for molecule in molecules]
complete_df['ecfp_bit_vectors'] = [[int(bit) for bit in keys.ToBitString()] for keys in ecfp]
complete_df['ECFP'] = [''.join(str(value) for value in row) for row in complete_df['ecfp_bit_vectors']]

# Adding moldescriptors to the Morgan Fingerprint
import numpy as np
from tqdm.auto import tqdm
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn import preprocessing as pre


def mol_descriptor(smiles: list[str], scale: bool = True) -> np.ndarray:
    descriptors_list=[]
    for smi in tqdm(smiles):

        m = Chem.MolFromSmiles(smi)
        descriptors = np.array([Descriptors.TPSA(m),
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
        descriptors.append(descriptors)

    if scale:
        descriptors_list = pre.MinMaxScaler().fit_transform(np.array(descriptors_list))
    else:
        return np.array(descriptors_list)

descriptors_list = mol_descriptor(complete_df['cpd_smiles'])

complete_df[['TPSA', 'MolLogP', 'MolWt', 'FpDensityMorgan2', 'HeavyAtomMolWt',
           'MaxPartialCharge', 'MinPartialCharge', 'NumRadicalElectrons',
           'NumValenceElectrons', 'CalcFractionCSP3', 'CalcNumRings',
           'CalcNumRotatableBonds', 'CalcNumLipinskiHBD', 'CalcNumLipinskiHBA',
           'CalcNumHeterocycles', 'CalcNumHeavyAtoms', 'CalcNumAromaticRings',
           'CalcNumAtoms', 'qed']] = descriptors_list


# Doelvariabele transformeren & limieten stellen dataframe
complete_df['ec50_mol'] = complete_df['apparent_ec50_umol'] / 1000000
complete_df['ec50_mol']=complete_df['ec50_mol'].replace(0, 1e-10)
complete_df['ec50_molair'] = complete_df['ec50_mol']/ complete_df['MolWt']
complete_df['ec50_molair_transformed'] = -nm.log10(complete_df['ec50_molair'])
condition = (complete_df['ec50_molair_transformed'] < 2 ) | (complete_df['ec50_molair_transformed'] > 8)  # Vraag waarom en wat de grenzen zijn
complete_df=complete_df[~condition]

# Dependent & Independent variable
#x = nm.array(complete_df['ecfp_bit_vectors'].tolist())
x= complete_df[['TPSA', 'MolLogP', 'MolWt', 'FpDensityMorgan2', 'HeavyAtomMolWt',
               'MaxPartialCharge', 'MinPartialCharge', 'NumRadicalElectrons',
               'NumValenceElectrons', 'CalcFractionCSP3', 'CalcNumRings',
               'CalcNumRotatableBonds', 'CalcNumLipinskiHBD', 'CalcNumLipinskiHBA',
               'CalcNumHeterocycles', 'CalcNumHeavyAtoms', 'CalcNumAromaticRings',
               'CalcNumAtoms', 'qed']]
y = complete_df['ec50_molair_transformed'].values

# Split Test & Train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, min_samples_split=5,min_samples_leaf=4,max_depth=10,bootstrap=True)
regressor.fit(x_train,y_train)

# Predict Test result
y_pred = regressor.predict(x_test)

# Visualisatie y_train ## Vooral voor eigen begrip
plt.hist(y_train, alpha=0.5, label='y_train')
plt.xlabel('Waarden EC50')
plt.ylabel('Frequentie')
plt.title('Histogram van y_train')
plt.legend()
plt.show()

# Visualisatie y_test ## Vooral voor eigen begrip
plt.hist(y_test, alpha=0.5, label='y_test')
plt.xlabel('Waarden EC50')
plt.ylabel('Frequentie')
plt.title('Histogram van y_test')
plt.legend()
plt.show()

# Fit Regressor
## Momenteel o.b.v. hyperparameters die in een oude versie van het model zijn berekend
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators=200, min_samples_split=5,min_samples_leaf=4,max_depth=10,bootstrap=True)
#regressor.fit(x_train,y_train)

# Predict Test result
#y_pred = regressor.predict(x_test)

# Visualisatie y_pred ## Vooral voor eigen begrip
plt.hist(y_pred, alpha=0.5, label='y_pred')
plt.xlabel('Waarden EC50')
plt.ylabel('Frequentie')
plt.title('Histogram van y_pred')
plt.legend()
plt.show()

# Errors berekenen
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mean Absolute Error
mae=mean_absolute_error(y_test, y_pred)
print('mean absolute error is ', mae)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print('mean squared error is ', mse)

# Root Mean Squared Error
rmse = math.sqrt(mse)
print('root mean squared error is ', rmse)

# R2 berekenen
slope, intercept = nm.polyfit(y_test,y_pred,1)
line = slope * nm.array(y_test)+ intercept
r2 = r2_score(y_test, y_pred)
print('r2 is', r2)

# R2 Visualisatie                                              
plt.scatter(y_test,y_pred)
plt.plot(y_test, line, color='red', label='line of current best fit')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.xlim(1,10)
plt.ylim(1,10)
plt.title('Scatterplot with Line of Best Fit (R2 = {:.2f})'.format(r2))
plt.show()

residuen = y_test - y_pred
plt.violinplot(residuen, showmedians=True)
iqr = nm.percentile(residuen, 75) - nm.percentile(residuen, 25)
plt.axhline(y=nm.percentile(residuen, 25), color='green', linestyle='--', label='25th Percentile')
plt.axhline(y=nm.percentile(residuen, 75), color='green', linestyle='--', label='75th Percentile')
plt.axhline(y=nm.percentile(residuen, 25) - 1.5 * iqr, color='orange', linestyle='--', label='1.5*IQR Range')
plt.axhline(y=nm.percentile(residuen, 75) + 1.5 * iqr, color='orange', linestyle='--')


# Toon de plot
plt.show()