import matplotlib.pyplot as plt
import numpy as nm
import pandas as pd

folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

#importing datasets
complete_df = pd.read_csv(f"{folder}v20.data.fingerprints.txt", sep="\t")
complete_df.fillna(complete_df.mean(), inplace=True)

#grenswaarden ec50 aangeven
complete_df['ec50_molair'] = -nm.log10(complete_df['ec50_molair'])
condition = (complete_df['ec50_molair'] < 4) | (complete_df['ec50_molair'] > 10)
complete_df=complete_df[~condition]

#ECFP uitlezen
ECFP_string = complete_df['ECFP']
ECFP_list = []
import ast
for string in complete_df['ECFP']:
    ECFP_single_list = ast.literal_eval(string)
    ECFP_list.append(ECFP_single_list)

#extracting independent and dependent variable
x=ECFP_list
y = complete_df['ec50_molair'].values
print('y=', y)

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

#from sklearn.preprocessing import StandardScaler
#st_x = StandardScaler()
#x_train=st_x.fit_transform(x_train)
#x_test=st_x.transform(x_test)



from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators=1024, random_state=42) #nestimators is requorednumber of trees in the trandom forest
#estimator = RandomForestRegressor(random_state = 42), param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
regressor = RandomForestRegressor(n_estimators=200, min_samples_split=5,min_samples_leaf=4,max_depth=10,bootstrap=True)
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
plt.xlim(1,10)
plt.ylim(1,10)
plt.title('Scatterplot with Line of Best Fit (R2 = {:.2f})'.format(r2))
plt.show()
