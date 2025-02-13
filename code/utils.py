import world
import torch
import numpy as np
from torch import LongTensor
import torch.nn.functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from dataloader import Loader
device = world.device
config = world.config


def Full_Sampling(dataset:Loader):
    """
    With Normlized Sampling on Graph
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    weights = dataset.sampling_weights
    batch_size = config['bpr_batch_size']
    mini_batch = []
    sampler = WeightedRandomSampler(
        weights,
        num_samples=train_edge_index.size(1)
    )
    train_loader = DataLoader(
        range(train_edge_index.size(1)),
        sampler=sampler,
        batch_size=batch_size
    )
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = torch.randint(0, num_items,(index.numel(), ), device=device)
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch
    
    
    

def Fast_Sampling(dataset:Loader):
    """
    With Uniformal Sampling on Graph
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    batch_size = config['bpr_batch_size']
    mini_batch = []
    train_loader = DataLoader(
            range(train_edge_index.size(1)),
            shuffle=True,
            batch_size=batch_size)
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = torch.randint(0, num_items,(index.numel(), ), device=device)
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch
        
    



def edge_drop(edge_index:LongTensor,drop_ratio = 0.1):
        num_drop = int(drop_ratio * edge_index.size(1))
        drop_index = np.random.randint(0,edge_index.size(1),(num_drop,))
        drop_index = torch.tensor(drop_index).to(device)
        mask = torch.ones_like(edge_index[0],dtype=torch.bool,device=edge_index.device)
        mask[drop_index[:num_drop]] = False
        edge_index_new = edge_index[:,mask]
        return edge_index_new

def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

def early_stopping(recall,
                   ndcg,
                   best,
                   patience,
                   model):
    if patience < 100: 
        if recall + ndcg > best: 
            patience = 0
            print('[BEST]')
            best = recall + ndcg
            # torch.save(model.state_dict(), save_file_name)
            # torch.save(model.state_dict(),'./models/' + save_file_name)
        else:
            patience += 1
        return 0,best,patience
    else:
        return 1,best,patience # Perform Early Stopping 

def eval(node_count,topk_index,logits,ground_truth,k):
    isin_mat = ground_truth.gather(1, topk_index)
    # Calculate recall
    recall = float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
    # Calculate NDCG
    log_positions = torch.log2(torch.arange(2, k + 2, device=logits.device).float())
    dcg = (isin_mat / log_positions).sum(dim=-1)
    ideal_dcg = torch.zeros_like(dcg)
    for i in range(len(dcg)):
        ideal_dcg = (1.0 / log_positions[:node_count[i].clamp(max=k).int()]).sum()
    ndcg = float((dcg / ideal_dcg.clamp(min=1e-6)).sum())
    return recall,ndcg

# ====================end Metrics=============================
# =========================================================
