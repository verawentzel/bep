import matplotlib.pyplot as plt
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

#importing datasets
complete_df = pd.read_csv(f"{folder}v20.data.full_data_summary.txt", sep="\t")
complete_df.fillna(complete_df.mean(), inplace=True)

complete_df['ec50_molair'] = -nm.log10(complete_df['ec50_molair'])
condition = (complete_df['ec50_molair'] < 1) | (complete_df['ec50_molair'] > 10)
complete_df=complete_df[~condition]

#extracting independent and dependent variable
x=complete_df.iloc[:,3:22]
y = complete_df['ec50_molair']
#y = -nm.log10(y) ## Let op: bij interpertreren en evalueren moet er eerst worden teruggeschaald met omgekeerde log-transformatie (np.expm1()).
plt.plot(y)
plt.show()

plt.boxplot(y)
#plt.hist(y)
plt.ylabel('Log transformed EC50 value')
plt.show()


 # Bereik/spreiding achterhalen vd doelvariabele
std_dev = y.std()
print("Standaarddeviatie:", std_dev)

data_min = y.min()
data_max = y.max()
data_range = data_max - data_min
print("Bereik (min-max):", data_range)

q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1
print("Interkwartielafstand (IQR):", iqr)

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33, random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

# Fitting Decision Tree classifier to the training set | friedman_mse
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42) #nestimators is requorednumber of trees in the trandom forest
regressor.fit(x_train,y_train)


# Predicting the test result
y_pred = regressor.predict(x_test)

# Errors berekenen
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


plt.scatter(y_test,y_pred)
plt.plot(y_test, line, color='red', label='line of current best fit')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Scatterplot with Line of Best Fit (R2 = {:.2f})'.format(r2))
plt.show()


print('r2 score is ', r2)

