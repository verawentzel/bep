from rdkit.Chem import AllChem
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold


# Import online Data Frame
folder = 'C:\\Users\\vswen\\Documents\\1. Biomedische Technologie\\BMT JAAR 5\\Kwart 4\\4. Data\\CTRPv2.0_2015_ctd2_ExpandedDataset\\'

# Dataframe met kernwaarden aanmaken
complete_df = pd.read_csv(f"{folder}chembl234_ki.csv", sep=",")
smiles_list = complete_df['smiles']

mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
scaffolds = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
scaffold_smiles = [Chem.MolToSmiles(scaffold) for scaffold in scaffolds]

# Fingerprint aanmaken
molecule_scaffold = [Chem.MolFromSmiles(smile) for smile in scaffold_smiles]
ecfp = [AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024) for molecule in molecule_scaffold]
#ecfp_bit_vectors = [[int(bit) for bit in keys.ToBitString()] for keys in ecfp]

def tanimoto_distance_matrix(efcp_list):
    dissimilarity_matrix =[]
    for i in range (1, len(efcp_list)):
        similarities = DataStructs.BulkTanimotoSimilarity(efcp_list[i], efcp_list[:i])
        dissimilarity_matrix.extend([1-x for x in similarities])
    return dissimilarity_matrix

sim = DataStructs.TanimotoSimilarity(ecfp[0], ecfp[1])
print(f"Tanimoto similarity: {sim:.2f}, distance: {1-sim:.2f}")

n = len(ecfp)
elem_triangular_matrix = (n * (n-1))/2
print(
    f"Elements in the triangular matrix ({elem_triangular_matrix:.0f}) ==",
    f"tanimoto_distance_matrix(fingerprints) ({len(tanimoto_distance_matrix(ecfp))})",
)

def cluster_fingerprints(efcp_list, cutoff=0.2):
    distance_matrix = tanimoto_distance_matrix(efcp_list)
    clusters = Butina.ClusterData(distance_matrix, len(efcp_list), cutoff, isDistData=True)
    clusters=sorted(clusters, key=len, reverse=True)
    return clusters

for cutoff in numpy.arange(0.0, 1.0, 0.1):
    clusters = cluster_fingerprints(ecfp, cutoff=cutoff)
    num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
    num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
    num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
    num_clust_g100 = sum(1 for c in clusters if len(c) > 100)

    print("Given te cutoff:", cutoff, " and the total # clusters: ", len(clusters))
    print("# clusters with only 1 compound: ", num_clust_g1)
    print("# clusters with >5 compounds: ", num_clust_g5)
    print("# clusters with >25 compounds: ", num_clust_g25)
    print("# clusters with >100 compounds: ", num_clust_g100)

    amb_plot = plt.subplots(figsize=(15, 4))
    plt.title(f"Threshold: {cutoff:3.1f}")
    plt.xlabel("Cluster index")
    plt.ylabel("Number of molecules")
    plt.bar(range(1, len(clusters) + 1), [len(c) for c in clusters], lw=5)
    plt.show()

cutoff = 0.3
clusters = cluster_fingerprints(ecfp, cutoff=cutoff)
print(f"Number of clusters: {len(clusters)} from {len(scaffold_smiles)} molecules at distance cut-off {cutoff:.2f}")
print("Number of molecules in largest cluster:", len(clusters[0]))
print(f"Similarity between two random points in same cluster: {DataStructs.TanimotoSimilarity(ecfp[clusters[0][0]], ecfp[clusters[0][1]]):.2f}")
print(f"Similarity between two random points in different cluster: {DataStructs.TanimotoSimilarity(ecfp[clusters[0][0]], ecfp[clusters[1][0]]):.2f}")

# Moleculen grootste cluster
scaffold_smiles = [(scaffold, smile) for scaffold, smile in zip(scaffolds, scaffold_smiles)]
moleculen_cluster_1= Draw.MolsToGridImage(
    [scaffold_smiles[i][0] for i in clusters[0][:10]],
    legends=[scaffold_smiles[i][1] for i in clusters[0][:10]],
    molsPerRow=5)
moleculen_cluster_1.save("moleculen_cluster_1.png")