import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#colors = [(0.773, 0.890, 0.933),  (0.055, 0.408, 0.596)]# Lichtgroen naar middelgroen
colors = [(0.055, 0.408, 0.596), (0.773, 0.890, 0.933)]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)


# Voorbeeldgegevens van een 5x6 tabel
#r2 = [[0.22, 0.24, 0.23],
 #      [0.08, 0.22, 0.21],
  #     [0.06, 0.08, 0.14],
   #     [0.19, 0.17, 0.19],
    #    [0.20, 0.27, 0.22]]

rmse = [[0.99, 1.05, 1.06],
        [0.90,1.26,1.27],
        [1.16,1.20,1.16],
        [1.05,1.04,1.03],
        [1.01,1.08,1.12]]

data=np.array(rmse)

y_labels = ['MM1s', 'MOLP2','EJM', 'KMS34', 'KSM18']
x_labels = ['RF', 'SVR', 'XGBoost']

# Maak een figuur en een subplot voor de heatmap
fig, ax = plt.subplots()

# Maak de heatmap met behulp van de imshow-functie
heatmap = ax.imshow(data, cmap=cmap)

# Voeg een kleurenbalk toe aan de rechterzijde van de heatmap
cbar = fig.colorbar(heatmap)

# Stel de labels in voor de assen
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))

# Stel de labels in voor de tick-posities
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

# Draai de tick-labels op de x-as
plt.xticks(rotation=0)

# Voeg de waarden toe aan de heatmap
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black')

# Titel
ax.set_title('RMSE Values\n Testing Cell Lines & Computational Models\n',fontweight='bold')


ax.set_ylabel('MM Cell Lines\n ')
#ax.set_xlabel('Computational Models')

# Toon de plot
plt.show()
