from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
warnings.filterwarnings('ignore')
script_dir = Path(__file__).parent
data_dir = script_dir / 'data' / 'Simulation'
rna_file = data_dir / 'adata_RNA.h5ad'
protein_file = data_dir / 'adata_ADT.h5ad'
adata_rna = sc.read_h5ad(rna_file)
adata_protein = sc.read_h5ad(protein_file)

sc.pp.filter_genes(adata_rna, min_cells=5)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)

from sklearn.decomposition import PCA
if hasattr(adata_rna.X, 'toarray'):
    feat_rna = adata_rna.X.toarray()
else:
    feat_rna = adata_rna.X
feat_rna = PCA(n_components=100).fit_transform(feat_rna)

def seurat_clr(x):
    x = np.array(x)
    x_pos = x[x > 0]
    if len(x_pos) > 0:
        s = np.sum(np.log1p(x_pos))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)
    return np.zeros_like(x)

if hasattr(adata_protein.X, 'toarray'):
    feat_protein = np.apply_along_axis(seurat_clr, 1, adata_protein.X.toarray())
else:
    feat_protein = np.apply_along_axis(seurat_clr, 1, adata_protein.X)

from scipy.spatial.distance import cdist

def construct_adj(coords, n_neighbors=3):
    dist_matrix = cdist(coords, coords)
    n = dist_matrix.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[1:n_neighbors+1]
        adj[i, neighbors] = 1
    adj = adj + adj.T
    adj = np.clip(adj, 0, 1)
    return adj, adj

coords = adata_rna.obsm['spatial']
adj_rna, graph_rna = construct_adj(coords, n_neighbors=3)
adj_protein, graph_protein = construct_adj(coords, n_neighbors=3)

def add_contrastive_label(adata):
    n = adata.n_obs
    label = np.zeros((n, 2))
    label[:, 0] = 1.0
    label[:, 1] = 0.0
    return label

label_rna = add_contrastive_label(adata_rna)
label_protein = add_contrastive_label(adata_protein)

omics1_data = {
    'feat': feat_rna,
    'adj': adj_rna,
    'graph_neigh': graph_rna,
    'label_CSL': label_rna,
    'spatial_locations': coords
}
omics2_data = {
    'feat': feat_protein,
    'adj': adj_protein,
    'graph_neigh': graph_protein,
    'label_CSL': label_protein
}
from main import train
train(omics1_data, omics2_data, 'Simulation', out_dim=55, negative_mode='similarity')

result = np.load(r'../result/Simulation/combine_emb.npy')
adata_result = AnnData(result)
adata_result.obsm['spatial'] = coords

sc.pp.pca(adata_result, n_comps=20)
sc.pp.neighbors(adata_result, n_neighbors=50)
sc.tl.leiden(adata_result, resolution=0.4)
sc.tl.umap(adata_result)

if 'spfac' in adata_rna.obsm:
    spfac = adata_rna.obsm['spfac']
    ground_truth = np.sum([(i+1) * spfac[:, i] for i in range(min(4, spfac.shape[1]))], axis=0)
else:
    ground_truth = pd.factorize(adata_rna.obs.get('cell_type', np.random.randint(0, 5, adata_rna.shape[0])))[0]

cluster_labels = pd.factorize(adata_result.obs['leiden'])[0]
ari = adjusted_rand_score(ground_truth, cluster_labels)
nmi = normalized_mutual_info_score(ground_truth, cluster_labels)
homo = homogeneity_score(ground_truth, cluster_labels)
umap_coords = adata_result.obsm['X_umap']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
adata_rna.obs['annotation'] = pd.Categorical(ground_truth)
sc.pl.spatial(adata_rna, color='annotation', spot_size=0.12, ax=axes[0], show=False, title='Ground Truth')
sc.pl.spatial(adata_result, color='leiden', spot_size=0.12, ax=axes[1], show=False, title='SpaCCS Clustering')
sc.pl.umap(adata_result, color='leiden', ax=axes[2], show=False, title='UMAP Visualization')
plt.tight_layout()
plt.savefig('spaccs_results.png', dpi=300, bbox_inches='tight')
plt.show()

