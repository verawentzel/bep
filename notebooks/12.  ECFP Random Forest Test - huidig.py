from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as nm
import math
import pandas as pd
import matplotlib.pyplot as plt
import random

'''ECFP RANDOM FOREST TEST'''

''' Dit is de code waarin ik de algemene dataset
laat runnen door mijn gemaakte ECFP Random Forest
om te zien of hij vergelijkbaar scoort met de cellijn
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
random_indices = random.sample(range(len(complete_df)), 400)
selected_data = complete_df.iloc[random_indices]

# Dependent & Independent variable
x = nm.array(selected_data['ecfp_bit_vectors'].tolist()) #Voorheen: complete_df['ecfp_bit_vectors']
y = selected_data['y'].values #Voorheen: complete_df['y']

#x = nm.array(complete_df['ecfp_bit_vectors'].tolist()) #Voorheen: complete_df['ecfp_bit_vectors']
#y = complete_df['y'].values #Voorheen: complete_df['y']

# Split Test & Train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

# Visualisatie y_train ## Vooral voor eigen begrip
plt.hist(y_train, alpha=0.5, label='y_train')
plt.xlabel('Waarden Test y')
plt.ylabel('Frequentie')
plt.title('Histogram van y_train')
plt.legend()
plt.show()

# Visualisatie y_test ## Vooral voor eigen begrip
plt.hist(y_test, alpha=0.5, label='y_test')
plt.xlabel('Waarden Test y')
plt.ylabel('Frequentie')
plt.title('Histogram van y_test')
plt.legend()
plt.show()

# Fit Regressor
## Momenteel o.b.v. hyperparameters die het andere model ook worden gebruikt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, min_samples_split=5,min_samples_leaf=4,max_depth=10,bootstrap=True)
regressor.fit(x_train,y_train)

# Predict Test result
y_pred = regressor.predict(x_test)

# Visualisatie y_pred ## Vooral voor eigen begrip
plt.hist(y_pred, alpha=0.5, label='y_pred')
plt.xlabel('Waarden Test y')
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
