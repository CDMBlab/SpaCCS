import os.path
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import SpaCCS
from utils import fix_seed
from preprocess import permutation
import torch
import os


def cmcl_loss(z1, z2, temperature=1.0):

    batch_size = z1.size(0)

    # 归一化嵌入向量
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)


    sim_matrix = torch.matmul(z1, z2.T) / temperature


    labels = torch.arange(batch_size).to(z1.device)


    loss_12 = F.cross_entropy(sim_matrix, labels)  # omics1 -> omics2
    loss_21 = F.cross_entropy(sim_matrix.T, labels)  # omics2 -> omics1


    loss = (loss_12 + loss_21) / 2

    return loss



def train(omics1_data, omics2_data, dataset, out_dim=64, dropout_rate=0.15, weight_decay=0.0, random_seed=2024, negative_mode='shuffle',           # 负样本模式
          similarity_threshold=0.4,
          **kwargs):
    fix_seed(random_seed)

    # 强制使用GPU 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # omics1 data
    omics1_feat = torch.FloatTensor(omics1_data['feat']).to(device)
    omics1_feat_shuffle = None
    omics1_label_CSL = torch.FloatTensor(omics1_data['label_CSL']).to(device)
    omics1_graph_neigh = torch.FloatTensor(omics1_data['graph_neigh'] + np.eye(omics1_data['adj'].shape[0])).to(device)
    omics1_input_dim = omics1_data['feat'].shape[1]
    omics1_adj = torch.FloatTensor(omics1_data['adj'] + np.eye(omics1_data['adj'].shape[0])).to(device)

    # omics2 data
    omics2_feat = torch.FloatTensor(omics2_data['feat']).to(device)
    omics2_feat_shuffle = None
    omics2_label_CSL = torch.FloatTensor(omics2_data['label_CSL']).to(device)
    omics2_graph_neigh = torch.FloatTensor(omics2_data['graph_neigh'] + np.eye(omics2_data['adj'].shape[0])).to(device)
    omics2_input_dim = omics2_data['feat'].shape[1]
    omics2_adj = torch.FloatTensor(omics2_data['adj'] + np.eye(omics2_data['adj'].shape[0])).to(device)


    spatial_coords = None
    if 'spatial_locations' in omics1_data:
        spatial_coords = omics1_data['spatial_locations']
    elif 'spatial_locations' in omics2_data:
        spatial_coords = omics2_data['spatial_locations']
    # parameters for different datasets
    if dataset == 'MISAR':
        learning_rate = 0.001
        epochs = 800
        omics1_n_neighbors = 3
        omics2_n_neighbors = 3
        factors = [1, 1, 10, 15, 3]
        cmcl_temperature = 0.5

    elif dataset == 'Mouse_Brain_P22':
        learning_rate = 0.001
        epochs = 200
        omics1_n_neighbors = 6
        omics2_n_neighbors = 6
        factors = [1, 1, 10, 15, 5]
        cmcl_temperature = 1.0

    elif dataset == "E10":
        learning_rate = 0.0001
        epochs = 200
        omics1_n_neighbors = 5
        omics2_n_neighbors = 5
        factors = [1, 1, 8, 15, 3]
        cmcl_temperature = 1.0

    else:
        learning_rate = 0.001
        epochs = 200
        omics1_n_neighbors = 3
        omics2_n_neighbors = 3
        factors = [1, 1, 5, 5, 2]
        cmcl_temperature =1.0

    model = SpaCCS(omics1_input_dim, omics2_input_dim, out_dim, dropout_rate,
                  omics1_feat, omics1_adj, omics1_graph_neigh, omics1_label_CSL,
                  omics2_feat, omics2_adj, omics2_graph_neigh, omics2_label_CSL)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

    model.train()

    loss_rec = nn.MSELoss()
    loss_CSL = nn.BCEWithLogitsLoss()

    train_losses = []
    cmcl_similarities = []
    individual_losses = {
        'csl1': [], 'csl2': [],
        'rec1': [], 'rec2': [],
        'cmcl': []
    }

    for epoch in tqdm(range(epochs)):
        model.train()
        omics1_feat_np = omics1_feat.cpu().numpy()
        omics2_feat_np = omics2_feat.cpu().numpy()
        omics1_feat_shuffle = permutation(
            feature=omics1_feat_np,
            mode='similarity',
            similarity_threshold=similarity_threshold,
        )
        omics1_feat_shuffle = torch.FloatTensor(omics1_feat_shuffle).to(device)

        omics2_feat_shuffle = permutation(
            feature=omics2_feat_np,
            mode='similarity',
            similarity_threshold=similarity_threshold,

        )
        omics2_feat_shuffle = torch.FloatTensor(omics2_feat_shuffle).to(device)

        (omics1_emb, omics1_rec, omics1_ret, omics1_ret_a,
         omics2_emb, omics2_rec, omics2_ret, omics2_ret_a,
         combine_emb, omics_weight) = model(omics1_feat_shuffle, omics2_feat_shuffle)

        loss_omics1_csl1 = loss_CSL(omics1_ret, omics1_label_CSL)
        loss_omics1_csl2 = loss_CSL(omics1_ret_a, omics1_label_CSL)
        loss_omics1_CSL = loss_omics1_csl1 + loss_omics1_csl2

        loss_omics2_csl1 = loss_CSL(omics2_ret, omics2_label_CSL)
        loss_omics2_csl2 = loss_CSL(omics2_ret_a, omics2_label_CSL)
        loss_omics2_CSL = loss_omics2_csl1 + loss_omics2_csl2


        loss_omics1_rec = loss_rec(omics1_feat, omics1_rec)
        loss_omics2_rec = loss_rec(omics2_feat, omics2_rec)

        loss_cmcl = cmcl_loss(omics1_emb, omics2_emb, cmcl_temperature)

        with torch.no_grad():
            cmcl_sim = F.cosine_similarity(omics1_emb, omics2_emb, dim=1).mean().item()
            cmcl_similarities.append(cmcl_sim)

        loss = (factors[0] * loss_omics1_CSL +
                factors[1] * loss_omics2_CSL +
                factors[2] * loss_omics1_rec +
                factors[3] * loss_omics2_rec +
                factors[4] * loss_cmcl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        individual_losses['csl1'].append(loss_omics1_CSL.item())
        individual_losses['csl2'].append(loss_omics2_CSL.item())
        individual_losses['rec1'].append(loss_omics1_rec.item())
        individual_losses['rec2'].append(loss_omics2_rec.item())
        individual_losses['cmcl'].append(loss_cmcl.item())


        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.4f}, "
                  f"CMCL Loss = {loss_cmcl.item():.4f}, "
                  f"CMCL Similarity = {cmcl_sim:.4f}")

    print('Model training completed!')


    with torch.no_grad():
        model.eval()
        torch.save(model.state_dict(), f'../result/{dataset}/model.pt')
        omics1_emb, omics1_rec, _, _, omics2_emb, omics2_rec, _, _, combine_emb, _ = model(omics1_feat_shuffle,
                                                                                           omics2_feat_shuffle)

        rec_omics1 = omics1_rec.clone().detach().cpu().numpy()
        rec_omics2 = omics2_rec.clone().detach().cpu().numpy()
        emb_omics1 = omics1_emb.clone().detach().cpu().numpy()
        emb_omics2 = omics2_emb.clone().detach().cpu().numpy()
        emb_combine = combine_emb.clone().detach().cpu().numpy()

        np.save(f'../result/{dataset}/combine_emb.npy', emb_combine)
        np.save(f'../result/{dataset}/omics1_emb.npy', emb_omics1)
        np.save(f'../result/{dataset}/omics2_emb.npy', emb_omics2)


        np.save(f'../result/{dataset}/train_losses.npy', np.array(train_losses))
        np.save(f'../result/{dataset}/cmcl_similarities.npy', np.array(cmcl_similarities))
        np.save(f'../result/{dataset}/individual_losses.npy', individual_losses)

        print(f"Results saved to ../result/{dataset}/")


    fig, axes = plt.subplots(2, 2, figsize=(12, 10))


    axes[0, 0].plot(train_losses, label='Total Loss', color='blue', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(cmcl_similarities, label='CMCL Similarity', color='red', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Similarity')
    axes[0, 1].set_title('Cross-Modal Similarity (Cosine)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(individual_losses['csl1'], label='CSL1 Loss', alpha=0.7, linewidth=1)
    axes[1, 0].semilogy(individual_losses['csl2'], label='CSL2 Loss', alpha=0.7, linewidth=1)
    axes[1, 0].semilogy(individual_losses['rec1'], label='Recon1 Loss', alpha=0.7, linewidth=1)
    axes[1, 0].semilogy(individual_losses['rec2'], label='Recon2 Loss', alpha=0.7, linewidth=1)
    axes[1, 0].semilogy(individual_losses['cmcl'], label='CMCL Loss', alpha=0.7, linewidth=1)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Individual Loss Components')
    axes[1, 0].legend(loc='upper right', fontsize='small')
    axes[1, 0].grid(True, alpha=0.3)

    if epochs > 100:
        recent_epochs = 100
        axes[1, 1].plot(range(epochs - recent_epochs, epochs),
                        train_losses[-recent_epochs:],
                        label='Total Loss', color='blue', linewidth=1.5)
        axes[1, 1].plot(range(epochs - recent_epochs, epochs),
                        individual_losses['cmcl'][-recent_epochs:],
                        label='CMCL Loss', color='red', linewidth=1.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Recent 100 Epochs')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough epochs\nfor recent analysis',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Recent Epochs Analysis')

    plt.tight_layout()
    plt.savefig(f'../result/{dataset}/training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Epochs ({dataset})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'../result/{dataset}/loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("This is the main training script for SpaCCS with CMCL.")
    print("Please import this module and call train() with appropriate parameters.")