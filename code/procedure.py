import torch
from torch import Tensor
import numpy as np
from torch_geometric.utils import degree
import world
import utils
import multiprocessing
from model import RecModel
device = world.device
config = world.config

"""
define evaluation metrics here
Already implemented:[Recall@K,NDCG@K]
"""
def train_bpr(dataset,model:RecModel,opt):
    model = model
    model.train()
    S = utils.Full_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        opt.zero_grad()
        pos_rank,neg_rank = model(edge_label_index)
        bpr_loss,reg_loss = model.recommendation_loss(pos_rank,neg_rank,edge_label_index)
        loss = bpr_loss + reg_loss
        loss.backward()
        opt.step()   
        aver_loss += (loss)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"


def train_bpr_sgl(dataset,
                  model,
                  opt,
                  edge_index1,
                  edge_index2):
    model = model
    model.train()
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        pos_rank,neg_rank = model(edge_label_index)
        bpr_loss = model.bpr_loss(pos_rank,neg_rank)
        ssl_loss = model.ssl_loss(edge_index1,edge_index2,edge_label_index)
        L2_reg = model.L2_reg(edge_label_index)
        loss = bpr_loss + ssl_loss + L2_reg 
        opt.zero_grad()
        loss.backward()
        opt.step()    
        aver_loss += (bpr_loss + ssl_loss + L2_reg)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"


def train_bpr_simgcl(dataset,
                  model,
                  opt):
    model = model
    model.train()
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        pos_rank,neg_rank = model(edge_label_index)
        bpr_loss = model.bpr_loss(pos_rank,neg_rank)
        ssl_loss = model.ssl_loss(edge_label_index)
        L2_reg = model.L2_reg(edge_label_index)
        loss = bpr_loss + ssl_loss + L2_reg
        opt.zero_grad()
        loss.backward()
        opt.step()    
        aver_loss += (bpr_loss + ssl_loss + L2_reg)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"


@torch.no_grad()
def test(k_values:list,
         model,
         train_edge_index,
         test_edge_index,
         num_users,
         ):
    model.eval()
    recall = {k: 0 for k in k_values}
    ndcg = {k: 0 for k in k_values}
    total_examples = 0
    for start in range(0, num_users, 2048):
        end = start + 2048
        if end > num_users:
            end = num_users
        src_index=torch.arange(start,end).long().to(device)
        logits = model.link_prediction(src_index=src_index,dst_index=None)

        # Exclude training edges:
        mask = ((train_edge_index[0] >= start) &
                (train_edge_index[0] < end))
        masked_interactions = train_edge_index[:,mask]
        logits[masked_interactions[0] - start,masked_interactions[1]] = float('-inf')
        # Generate ground truth matrix
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((test_edge_index[0] >= start) &
                (test_edge_index[0] < end))
        masked_interactions = test_edge_index[:,mask]
        ground_truth[masked_interactions[0] - start,masked_interactions[1]] = True
        node_count = degree(test_edge_index[0, mask] - start,
                            num_nodes=logits.size(0))
        topk_indices = logits.topk(max(k_values),dim=-1).indices
        for k in k_values:
            topk_index = topk_indices[:,:k]
            isin_mat = ground_truth.gather(1, topk_index)
            # Calculate recall
            recall[k] += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
            # Calculate NDCG
            log_positions = torch.log2(torch.arange(2, k + 2, device=logits.device).float())
            dcg = (isin_mat / log_positions).sum(dim=-1)
            ideal_dcg = torch.zeros_like(dcg)
            for i in range(len(dcg)):
                ideal_dcg[i] = (1.0 / log_positions[:node_count[i].clamp(0, k).int()]).sum()
            ndcg[k] += float((dcg / ideal_dcg.clamp(min=1e-6)).sum())

        total_examples += int((node_count > 0).sum())

    recall = {k: recall[k] / total_examples for k in k_values}
    ndcg = {k: ndcg[k] / total_examples for k in k_values}

    return recall,ndcg

