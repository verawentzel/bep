import matplotlib.pyplot as plt
import numpy as nm
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import math
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

'''ECFP XGBOOST TEST'''

''' Dit is de code waarin ik de algemene dataset
laat runnen door mijn gemaakte ECFP XGBoost om te
zien of hij vergelijkbaar scoort met de cellijn
data.'''

folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Import data
complete_df = pd.read_csv(f"{folder}chembl234_ki.csv", sep=",")
complete_df.fillna(complete_df.mean(), inplace=True)

# Fingerprint aanmaken
molecules = [Chem.MolFromSmiles(smile) for smile in complete_df['smiles'].tolist()]
ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule,2,nBits=1024) for molecule in molecules]
complete_df['ecfp_bit_vectors'] = [[int(bit) for bit in keys.ToBitString()] for keys in ecfp]
complete_df['ECFP'] = [''.join(str(value) for value in row) for row in complete_df['ecfp_bit_vectors']]

# Data selecteren om een vergelijkbare data grootte te creeeren
#random_indices = random.sample(range(len(complete_df)), 400)
#selected_data = complete_df.iloc[random_indices]

# Dependent & Independent variable
#x = nm.array(selected_data['ecfp_bit_vectors'].tolist()) #Voorheen: complete_df['ecfp_bit_vectors']
#y = selected_data['y'].values #Voorheen: complete_df['y']

x = nm.array(complete_df['ecfp_bit_vectors'].tolist()) #Voorheen: complete_df['ecfp_bit_vectors']
y = complete_df['y'].values #Voorheen: complete_df['y']

# Split Test & Train
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit XGBoost-regressiemodel
model = xgb.XGBRegressor()
grid = xgb.XGBRegressor(colsample_bytree= 0.8, learning_rate= 0.01, max_depth= 3, n_estimators= 500, subsample= 0.8)
grid.fit(X_train,y_train)

# Predict Test result
y_pred=grid.predict(X_test)

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

# R2 visualisatie scatterplot
slope, intercept = nm.polyfit(y_test,y_pred,1)
line = slope * nm.array(y_test)+ intercept
plt.scatter(y_test,y_pred)
plt.plot(y_test, line, color='red', label='line of current best fit')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Scatterplot with Line of Best Fit (R2 = {:.2f})'.format(r2))
plt.show()
