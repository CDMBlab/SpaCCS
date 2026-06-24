import ot
import h5py
import sklearn
import episcanpy
import scipy.sparse
import sklearn.decomposition
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
def preprocess_rna(rna_path, gene_n_neighbors):
    with h5py.File(rna_path, 'r') as file:
        gene_counts = file['X'][:]
        cell_barcodes = file['cell'][:]
        gene_names = file['gene'][:]
        spatial_locations = file['pos'][:]

    adata_gene = AnnData(X=gene_counts,
                         obs=pd.DataFrame(index=cell_barcodes),
                         var=pd.DataFrame(index=gene_names))
    adata_gene.obsm['spatial'] = spatial_locations

    sc.pp.filter_genes(adata_gene, min_cells=3)
    sc.pp.normalize_total(adata_gene, target_sum=1e4)
    sc.pp.log1p(adata_gene)
    sc.pp.highly_variable_genes(adata_gene, n_top_genes=1800)
    adata_gene_copy = adata_gene[:, adata_gene.var['highly_variable']]
    if isinstance(adata_gene_copy.X, csc_matrix) or isinstance(adata_gene_copy.X, csr_matrix):
        feat = adata_gene_copy.X.toarray()[:, ]
    else:
        feat = adata_gene_copy.X[:, ]
    pca = sklearn.decomposition.PCA(n_components=50)
    feat = pca.fit_transform(feat)

    adj, graph_neigh = construct_adj(adata_gene, n_neighbors=gene_n_neighbors)

    # label for contrastive learning
    label_CSL = add_contrastive_label(adata_gene)

    return {
        'feat': feat,
        'adj': adj,
        'graph_neigh': graph_neigh,
        'label_CSL': label_CSL,
        'spatial_locations': adata_gene.obsm['spatial']
    }


def preprocess_atac(atac_path, peak_n_neighbors, coords):

    # read MISAR(.h5) file
    with h5py.File(atac_path, 'r') as file:
        peak_counts = file['X'][:]
        cell_barcodes = file['cell'][:]
        peak_names = file['peak'][:]

    adata_peak = AnnData(X=peak_counts,
                         obs=pd.DataFrame(index=cell_barcodes),
                         var=pd.DataFrame(index=peak_names))
    adata_peak.obsm['spatial'] = coords


    adata_peak_copy = adata_peak.copy()
    episcanpy.pp.binarize(adata_peak_copy)
    episcanpy.pp.filter_features(adata_peak_copy, min_cells=np.ceil(0.005 * adata_peak.shape[0]))
    count_mat = adata_peak_copy.X.copy()
    X = tfidf(count_mat)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, 51)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    X_lsi = X_lsi[:, 1:]
    if isinstance(X_lsi, csc_matrix) or isinstance(X_lsi, csr_matrix):
        feat = X_lsi.toarray()[:, ]
    else:
        feat = X_lsi[:, ]


    adj, graph_neigh = construct_adj(adata_peak, n_neighbors=peak_n_neighbors)

    label_CSL = add_contrastive_label(adata_peak)

    return {
        'feat': feat,
        'adj': adj,
        'graph_neigh': graph_neigh,
        'label_CSL': label_CSL
    }


def preprocess_adt(adt_path, adt_n_neighbors):


    adata_adt = sc.read_h5ad(adt_path)
    sc.pp.filter_genes(adata_adt, min_cells=50)

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    adata_adt.X = np.apply_along_axis(
        seurat_clr, 1, (adata_adt.X.A if scipy.sparse.issparse(adata_adt.X) else np.array(adata_adt.X))
    )
    feat = adata_adt.X


    adj, graph_neigh = construct_adj(adata_adt, n_neighbors=adt_n_neighbors)

    label_CSL = add_contrastive_label(adata_adt)

    return {
        'feat': feat,
        'adj': adj,
        'graph_neigh': graph_neigh,
        'label_CSL': label_CSL
    }


def tfidf(X):


    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def construct_adj(adata, n_neighbors=3):
    coordinate = adata.obsm['spatial']
    distance_matrix = ot.dist(coordinate, coordinate, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    graph_neigh = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    return adj, graph_neigh


def permutation(feature, mode='similarity', **kwargs):
    n_spots = feature.shape[0]
    similarity_threshold = kwargs.get('similarity_threshold', 0.4)

    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(feature)

    new_ids = np.zeros(n_spots, dtype=int)

    for i in range(n_spots):
        similarities = sim_matrix[i]
        low_sim_indices = np.where(
            (similarities < similarity_threshold) &
            (np.arange(n_spots) != i)
        )[0]

        if len(low_sim_indices) > 0:
            j = np.random.choice(low_sim_indices)
        else:
            sorted_indices = np.argsort(similarities)
            candidate_indices = sorted_indices[sorted_indices != i]
            if len(candidate_indices) > 0:
                j = candidate_indices[0]
            else:
                j = i

        new_ids[i] = j

    return feature[new_ids]


def add_contrastive_label(adata):

    n_spot = adata.n_obs
    label_CSL = np.zeros([n_spot, 2])


    label_CSL[:, 0] = 1.0
    label_CSL[:, 1] = 0.0

    return label_CSL