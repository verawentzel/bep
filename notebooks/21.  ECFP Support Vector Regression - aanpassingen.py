import matplotlib.pyplot as plt
import numpy as nm
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split

folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

#importing datasets
complete_df = pd.read_csv(f"{folder}v20.data.final_summary.txt", sep="\t")
complete_df.fillna(complete_df.mean(), inplace=True)

# Fingerprint aanmaken
molecules = [Chem.MolFromSmiles(smile) for smile in complete_df['cpd_smiles'].tolist()]
ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule,2,nBits=1024) for molecule in molecules]
complete_df['ecfp_bit_vectors'] = [[int(bit) for bit in keys.ToBitString()] for keys in ecfp]
complete_df['ECFP'] = [''.join(str(value) for value in row) for row in complete_df['ecfp_bit_vectors']]

complete_df['ec50_mol'] = complete_df['apparent_ec50_umol'] / 1000000
complete_df['ec50_mol']=complete_df['ec50_mol'].replace(0, 1e-10)
complete_df['ec50_molair'] = complete_df['ec50_mol']/ complete_df['MolWt']

# Grenswaarden aangeven
complete_df['ec50_molair_transformed'] = -nm.log10(complete_df['ec50_molair'])
condition = (complete_df['ec50_molair_transformed'] < 2) | (complete_df['ec50_molair_transformed'] > 9)
complete_df=complete_df[~condition]

x = nm.array(complete_df['ecfp_bit_vectors'].tolist())
y = complete_df['ec50_molair_transformed'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = SVR(C=1, epsilon=0.1, kernel='rbf')
model.fit(X_train,y_train)
y_pred  = model.predict(X_test)


# Evalueren
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

mae=mean_absolute_error(y_test, y_pred)
print('mean absolute error is ', mae)

mse = mean_squared_error(y_test, y_pred)
import math
rmse = math.sqrt(mse)
print('mean squared error is ', mse)
print('root mean squared error is ', rmse)

slope, intercept = nm.polyfit(y_test,y_pred,1)
line = slope * nm.array(y_test)+ intercept
r2 = r2_score(y_test, y_pred)
print('r2 is ', r2)

# Visualiseren
plt.scatter(y_test,y_pred)
plt.plot(y_test, line, color='red', label='line of current best fit')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.xlim(1,10)
plt.ylim(1,10)
plt.title('Scatterplot with Line of Best Fit (R2 = {:.2f})'.format(r2))
plt.show()