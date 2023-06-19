import matplotlib.pyplot as plt
import numpy as nm
import pandas as pd
import math
from rdkit import Chem
from rdkit.Chem import AllChem


folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Import data
complete_df_263 = pd.read_csv(f"{folder}v20.data.final_summary_263.txt", sep="\t")
complete_df_263.fillna(complete_df_263.mean(), inplace=True)

complete_df_419 = pd.read_csv(f"{folder}v20.data.final_summary_419.txt", sep="\t")
complete_df_419.fillna(complete_df_419.mean(), inplace=True)

# Fingerprint aanmaken
molecules = [Chem.MolFromSmiles(smile) for smile in complete_df_263['cpd_smiles'].tolist()]
ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule,2,nBits=1024) for molecule in molecules]
complete_df_263['ecfp_bit_vectors'] = [[int(bit) for bit in keys.ToBitString()] for keys in ecfp]
complete_df_263['ECFP'] = [''.join(str(value) for value in row) for row in complete_df_263['ecfp_bit_vectors']]

# Doelvariabele transformeren & limieten stellen dataframe
complete_df_263['ec50_mol'] = complete_df_263['apparent_ec50_umol'] / 1000000
complete_df_263['ec50_mol']=complete_df_263['ec50_mol'].replace(0, 1e-10)
complete_df_263['ec50_molair'] = complete_df_263['ec50_mol']/ complete_df_263['MolWt']
complete_df_263['ec50_molair_transformed_263'] = -nm.log10(complete_df_263['ec50_molair'])
condition = (complete_df_263['ec50_molair_transformed_263'] < 4 ) | (complete_df_263['ec50_molair_transformed_263'] > 10)  # Vraag waarom en wat de grenzen zijn
complete_df_263=complete_df_263[~condition]

# Doelvariabele transformeren & limieten stellen dataframe
complete_df_419['ec50_mol'] = complete_df_419['apparent_ec50_umol'] / 1000000
complete_df_419['ec50_mol']=complete_df_419['ec50_mol'].replace(0, 1e-10)
complete_df_419['ec50_molair'] = complete_df_419['ec50_mol']/ complete_df_419['MolWt']
complete_df_419['ec50_molair_transformed_419'] = -nm.log10(complete_df_419['ec50_molair'])
condition = (complete_df_419['ec50_molair_transformed_419'] < 4 ) | (complete_df_419['ec50_molair_transformed_419'] > 10)  # Vraag waarom en wat de grenzen zijn
complete_df_419=complete_df_419[~condition]

extracted_col_419 = complete_df_419[["master_cpd_id","ec50_molair_transformed_419"]]
df_summary_sorted = pd.merge(complete_df_263, extracted_col_419, on='master_cpd_id', how='left')

# Dependent & Independent variable
x = nm.array(df_summary_sorted['ecfp_bit_vectors'].tolist())

y_pivot = df_summary_sorted.pivot(index='master_cpd_id', columns=['ec50_molair_transformed_263', 'ec50_molair_transformed_419'])
y_pivot = y_pivot.dropna()
y = y_pivot.values

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
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, min_samples_split=5,min_samples_leaf=4,max_depth=10,bootstrap=True)
regressor.fit(x_train,y_train)

# Predict Test result
y_pred = regressor.predict(x_test)

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
plt.title('Scatterplot with Line of Best Fit (R2 = {:.2f})'.format(r2))
plt.show()

