import os
import torch
import random
import numpy as np
import scanpy as sc
from anndata import AnnData
from torch.backends import cudnn

def evaluate_multiview_consistency(model, omics_data, device):

    model.eval()


    if 'feat_views' in omics_data:
        feat_views = [torch.FloatTensor(view).to(device) for view in omics_data['feat_views']]
        adj = torch.FloatTensor(omics_data['adj'] + np.eye(omics_data['adj'].shape[0])).to(device)
        graph_neigh = torch.FloatTensor(omics_data['graph_neigh'] + np.eye(omics_data['adj'].shape[0])).to(device)


        with torch.no_grad():
            if hasattr(model, 'multiview_loss_fn'):

                omics_embs, _, _ = model.omics1_encoder(
                    feat_views[0], feat_views[1], adj, graph_neigh, return_multiview=True
                )

                similarities = []
                for i in range(len(omics_embs)):
                    for j in range(i + 1, len(omics_embs)):
                        cos_sim = F.cosine_similarity(omics_embs[i], omics_embs[j], dim=1)
                        similarities.append(cos_sim.mean().item())

                avg_similarity = np.mean(similarities) if similarities else 0
                print(f"Average multiview similarity: {avg_similarity:.4f}")
                return avg_similarity

    return 0.0

def leiden_cluster(result_embedding, spatial_coords):

    X = np.load(result_embedding)
    adata = AnnData(X)
    adata.obsm['spatial'] = spatial_coords

    sc.pp.pca(adata, n_comps=20)
    sc.pp.neighbors(adata, n_neighbors=50, use_rep='X')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1)
    sc.pl.spatial(adata, color='leiden', spot_size=1)

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
