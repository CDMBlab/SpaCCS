import os
import h5py
import torch
import sklearn
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from utils import fix_seed
from scipy.spatial import distance_matrix
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14
sns.set_style("white")
from sklearn.feature_extraction.text import TfidfTransformer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fix_seed(2024)
from pathlib import Path

script_dir = Path(__file__).parent
def construct_adj_custom(adata, n_neighbors=6, spatial_coords=None):

    dist_matrix = distance_matrix(spatial_coords, spatial_coords)
    adj = np.zeros((len(spatial_coords), len(spatial_coords)), dtype=np.float32)
    for i in range(len(spatial_coords)):
        nearest_idx = np.argsort(dist_matrix[i])[1:n_neighbors + 1]
        adj[i, nearest_idx] = 1
        adj[nearest_idx, i] = 1
    np.fill_diagonal(adj, 0)

    graph_neigh = [np.where(adj[i] == 1)[0].tolist() for i in range(adj.shape[0])]
    return adj, graph_neigh

def leiden_clustering_force_k(embeddings, target_k=10, n_neighbors=15, n_pcs=30):

    adata = AnnData(embeddings)
    n_pcs_use = min(n_pcs, embeddings.shape[1])

    sc.pp.pca(adata, n_comps=n_pcs_use, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs_use, use_rep='X_pca')
    best_res = None
    for res in np.arange(0.1, 3.0, 0.05):
        sc.tl.leiden(adata, resolution=res, key_added='temp', flavor="igraph", n_iterations=5, directed=False)
        if len(adata.obs['temp'].unique()) == target_k:
            best_res = res
            break
    if best_res is None:
        min_diff = float('inf')
        for res in np.arange(0.1, 3.0, 0.05):
            sc.tl.leiden(adata, resolution=res, key_added='temp', flavor="igraph", n_iterations=5, directed=False)
            diff = abs(len(adata.obs['temp'].unique()) - target_k)
            if diff < min_diff:
                min_diff, best_res = diff, res

    sc.tl.leiden(adata, resolution=best_res, key_added='leiden', flavor="igraph", n_iterations=5, directed=False)
    labels = adata.obs['leiden'].astype(int).values

    return labels

if __name__ == "__main__":
    data_dir = script_dir / 'data' / 'Mouse Embryo E10'
    data_file = data_dir / 'DBiT_Seq Mouse Embryo_RNA_Protein.h5'
    data_mat = h5py.File(data_file, 'r')
    rna_data = data_mat['X_gene'][:]
    protein_data = data_mat['X_protein'][:]
    spatial_coords = data_mat['pos'][:]
    cell_names = data_mat['cell'][:]
    cell_names = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in cell_names]

    min_cells = min(rna_data.shape[0], protein_data.shape[0], spatial_coords.shape[0])
    rna_data = rna_data[:min_cells]
    protein_data = protein_data[:min_cells]
    spatial_coords = spatial_coords[:min_cells]
    cell_names = cell_names[:min_cells]

    adata_rna = AnnData(rna_data)
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    if adata_rna.shape[1] > 3000:
        sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000)
        adata_rna = adata_rna[:, adata_rna.var['highly_variable']]
    X_rna = adata_rna.X.toarray() if hasattr(adata_rna.X, 'toarray') else adata_rna.X
    feat_rna = sklearn.decomposition.PCA(n_components=min(100, X_rna.shape[1], X_rna.shape[0])).fit_transform(X_rna)

    adata_protein = AnnData(protein_data)
    sc.pp.filter_genes(adata_protein, min_cells=10)
    X_protein = adata_protein.X.toarray() if hasattr(adata_protein.X, 'toarray') else adata_protein.X
    X = TfidfTransformer(norm=None).fit_transform(X_protein)
    X = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X = np.log1p(X * 1e4)
    n_comp = min(51, X.shape[1], X.shape[0])
    X_lsi = sklearn.utils.extmath.randomized_svd(X, n_comp)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    feat_protein = X_lsi[:, 1:] if X_lsi.shape[1] > 1 else X_lsi
    adj_rna, _ = construct_adj_custom(adata_rna, n_neighbors=6, spatial_coords=spatial_coords)
    adj_protein, _ = construct_adj_custom(adata_protein, n_neighbors=6, spatial_coords=spatial_coords)

    from preprocess import add_contrastive_label
    omics1_data = {'feat': feat_rna, 'adj': adj_rna, 'graph_neigh': adj_rna, 'label_CSL': add_contrastive_label(adata_rna)}
    omics2_data = {'feat': feat_protein, 'adj': adj_protein, 'graph_neigh': adj_protein,
                   'label_CSL': add_contrastive_label(adata_protein)}

    result_dir = '../result/E10'
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, 'spatial_coords.npy'), spatial_coords)
    try:
        from main import train
        train(omics1_data, omics2_data, 'E10', negative_mode='similarity', out_dim=64)
    except Exception as e:
        print(f"error: {e}")
    data_mat.close()

    result_path = os.path.join(result_dir, 'combine_emb.npy')
    if not os.path.exists(result_path):
        result_path = os.path.join(result_dir, 'embeddings.npy')

    embeddings = np.load(result_path)
    target_k = 10
    labels = leiden_clustering_force_k(
        embeddings, target_k=target_k, n_neighbors=15, n_pcs=30
    )
    actual_k = len(np.unique(labels))
    print(f"k={target_k}->{actual_k}")
    pd.DataFrame({
        'cell_id': cell_names[:len(embeddings)],
        'cluster_id': labels,
        'x': spatial_coords[:len(embeddings), 0],
        'y': spatial_coords[:len(embeddings), 1]
    }).to_csv(os.path.join(result_dir, f'labels_k{target_k}.csv'), index=False)
    adata_viz = AnnData(embeddings)
    adata_viz.obs['cluster'] = labels.astype(str)
    adata_viz.obsm['spatial'] = spatial_coords[:len(embeddings)]
    sc.pp.neighbors(adata_viz, n_neighbors=30, use_rep='X')
    sc.tl.umap(adata_viz, min_dist=0.3, spread=1.0)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    palette = {str(i): colors[i % 20] for i in np.unique(labels)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'DBiT-Seq | k={target_k}')
    sc.pl.spatial(adata_viz, color='cluster', spot_size=1.0, ax=axes[0],
                  title='Spatial', show=False, palette=palette)
    sc.pl.umap(adata_viz, color='cluster', ax=axes[1], title='UMAP', s=100, show=False, palette=palette)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'dual_plot_k{target_k}.png'), dpi=300, bbox_inches='tight')
    np.save(os.path.join(result_dir, 'embeddings.npy'), embeddings)
    plt.show(block=True)