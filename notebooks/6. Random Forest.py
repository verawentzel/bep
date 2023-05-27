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

#extracting independent and dependent variable
x=complete_df.iloc[:,[3,21]]
y=complete_df.iloc[:,1]

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33, random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

print(x_train)
print(x_test)

# Fitting Decision Tree classifier to the training set | friedman_mse
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy") #nestimators is requorednumber of trees in the trandom forest
print(classifier)
classifier.fit(x_train,y_train)


# Predicting the test result
y_pred = classifier.predict(x_test)

#Creating the confusion matrix
from sklearn.metrics    import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the training set result
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1,x2 = nm.meshgrid(nm.arange(start=x_set[:,0].min() - 1, stop=[:,0].max()+1, step=0.01),
nm.arange(start=x_set[:,1].mix()-1,stop=x_set[:,1].max()+1, step=0.01))
mtp.contourf(x1,x2,classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('purple','green')))
mtp.xlim(x1.min(),x1.max())
mtp.ylim(x2.min(). x2.max())
for i, j in enumerate(nm.unique(y_set)):
    mtp.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                c=ListedColormap(('purple','green'))(i),label=j)
mtp.title('Random Forest Algorithm (Training set)')
mtp.legend()
mtp.show()