import matplotlib.pyplot as plt
import numpy as nm
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

'''ECFP SUPPORT VECTOR REGRESSION'''

''' Dit is mijn eerdere variant van het model voor
ECFP Support Vector Regression, hier staat o.a. data
preparation niet in. De meest up-to-date versie is 21. ECFP 
Support Vector Regression - aanpassingen'''

folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

#importing datasets
complete_df = pd.read_csv(f"{folder}v20.data.fingerprints.txt", sep="\t")
complete_df.fillna(complete_df.mean(), inplace=True)

complete_df['transformed_ec50_mol'] = -nm.log10(complete_df['ec50_mol'])
condition = (complete_df['transformed_ec50_mol'] < 2) | (complete_df['transformed_ec50_mol'] > 9)
complete_df=complete_df[~condition]

ECFP_string = complete_df['ECFP']
ECFP_list = []
import ast
for string in complete_df['ECFP']:
    ECFP_single_list = ast.literal_eval(string)
    ECFP_list.append(ECFP_single_list)

x = ECFP_list
y = complete_df['transformed_ec50_mol'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Optimale grid SVR via CV
#svr = SVR()
#parameters = {
 #   'C': [0.1, 1, 10],
  #  'epsilon': [0.01, 0.1, 1],
   # 'kernel': ['linear', 'rbf']
#}

#model=GridSearchCV(svr,parameters, cv=5)
#model.fit(X_train, y_train)


# Toepassen optimale grid
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