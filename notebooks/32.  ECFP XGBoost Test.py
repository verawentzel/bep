import matplotlib.pyplot as plt
import numpy as nm
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
import random
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

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

#extracting independent and dependent variable
x = nm.array(selected_data['ecfp_bit_vectors'].tolist()) #Voorheen: complete_df['ecfp_bit_vectors']
y = selected_data['y'].values #Voorheen: complete_df['y']

# Split de gegevens in een trainingset en een testset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# XGBoost-regressiemodel
model = xgb.XGBRegressor()

# Toepassen optimale grid
grid = xgb.XGBRegressor(colsample_bytree= 0.8, learning_rate= 0.01, max_depth= 3, n_estimators= 500, subsample= 0.8)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)

# Evaluatie
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = nm.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('Mean Absolute Error: {:.2f}'.format(mae))
print('Mean Squared Error: {:.2f}'.format(mse))
print('Root Mean Squared Error: {:.2f}'.format(rmse))
print('R^2 Score: {:.2f}'.format(r2))

# Visualisatie scatterplot
slope, intercept = nm.polyfit(y_test,y_pred,1)
line = slope * nm.array(y_test)+ intercept
plt.scatter(y_test,y_pred)
plt.plot(y_test, line, color='red', label='line of current best fit')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Scatterplot with Line of Best Fit (R2 = {:.2f})'.format(r2))
plt.show()
