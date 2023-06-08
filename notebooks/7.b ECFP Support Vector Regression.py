import matplotlib.pyplot as plt
import numpy as nm
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

#importing datasets
complete_df = pd.read_csv(f"{folder}v20.data.fingerprints.txt", sep="\t")
complete_df.fillna(complete_df.mean(), inplace=True)

complete_df['ec50_mol'] = -nm.log10(complete_df['ec50_mol'])
condition = (complete_df['ec50_mol'] < 1) | (complete_df['ec50_mol'] > 10)
complete_df=complete_df[~condition]

ECFP_string = complete_df['ECFP']
ECFP_list = []
import ast
for string in complete_df['ECFP']:
    ECFP_single_list = ast.literal_eval(string)
    ECFP_list.append(ECFP_single_list)

#extracting independent and dependent variable
x = ECFP_list
y = complete_df['ec50_mol'].values

# Vorm van X wijzigen naar (n_samples, 1)
# x = nm.ravel(x)
# x = nm.reshape(x, (1, -1))

# Verdeel de gegevens in trainings- en testsets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Maak het SVR-model aan
svr = SVR(kernel='rbf')

# Parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# Train het model
model=GridSearchCV(svr,param_grid, cv=5)
model.fit(X_train, y_train)

# Beste parameters en score weergeven
print("Beste parameters: ", model.best_params_)
print("Beste score: ", model.best_score_)

# Maak voorspellingen op de testset
y_pred  = model.predict(X_test)
print(y_pred)
#print(y_pred_final)

# Bereken de RMSE (Root Mean Squared Error) op de testset
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

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