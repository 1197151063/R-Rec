from model import RecModel
from torch import LongTensor,Tensor
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from dataloader import Loader
import world
from procedure import test
import utils
import torch.nn.functional as F
from torch_sparse import SparseTensor,matmul
import time


if world.config['dataset'] == 'yelp2018':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':0.01,#INIT WEIGHT
        'K':2,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-5,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'ssl_tmp':0.2,#TEMPERATURE
        'ssl_decay':1e-5,#SSL_STRENGTH
        'aug_ratio':0.1,#ADDING EDGE RATIO
        'prune_threshold': 0.02
    }
if world.config['dataset'] == 'amazon-book':
    config = {
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'ssl_tmp':0.2,#TEMPERATURE
        'ssl_decay':0.5,#SSL_STRENGTH
        'drop_ratio':0.1,#EDGE_DROP_RATIO
    }
if world.config['dataset'] == 'gowalla':
    config = {
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'ssl_tmp':0.2,#TEMPERATURE
        'ssl_decay':0.05,#SSL_STRENGTH
        'drop_ratio':0.2,#EDGE_DROP_RATIO
    }

class RGCF(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config):
        super().__init__(num_users=num_users,
                         num_items=num_items,
                         edge_index=edge_index,
                         config=config)
        self.K = config['K']
        self.num_interactions = edge_index.size(1)
        self.edge_index = edge_index  
        self.alpha= 1./ (1 + self.K)
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['dim'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['dim'])
        self.init_weight()
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K + 1
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K + 1))
        self.prune_threshold = config['prune_threshold']
        self.ssl_tmp = config['ssl_tmp']
        self.ssl_decay = config['ssl_decay'] 
        self.aug_ratio = config['aug_ratio'] 
        self.ssl_decay = config['ssl_decay']  
        self.value = None
        self.aug_graph = None
        self.edge_index_sp = self.get_sparse_graph(edge_index)
        self.edge_index_norm = None
        print('Go RGCF')
        print(f"params settings: \n emb_size:{config['dim']}\n L2 reg:{config['decay']}\n layer:{self.K}")
        print(f" ssl_tmp:{config['ssl_tmp']}\n ssl_decay:{config['ssl_decay']}\n pruning threshold:{config['prune_threshold']}")

    def init_weight(self):
        if config['init'] == 'normal':
            nn.init.normal_(self.user_emb.weight,std=config['init_weight'])
            nn.init.normal_(self.item_emb.weight,std=config['init_weight'])
        else:
            nn.init.xavier_uniform_(self.user_emb.weight,gain=config['init_weight'])
            nn.init.xavier_uniform_(self.item_emb.weight,gain=config['init_weight'])
    
    def get_hidden_emb(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        x = self.propagate(edge_index=self.edge_index_sp,x=x)
        out_u,out_i = torch.split(x,[self.num_users,self.num_items])
        return out_u, out_i
    
    def cal_cos_sim_sp(self,edge_index,a,b,eps=1e-8,CHUNK_SIZE=65536):
        a_n, b_n = a.norm(dim=1)[:,None],b.norm(dim=1)[:,None]
        a_norm = a / torch.max(a_n,eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n,eps * torch.ones_like(b_n))
        sims = torch.zeros(edge_index.size(1),dtype=a.dtype).to(device)
        for idx in range(0,edge_index.size(1),CHUNK_SIZE):
            batch_row_index = edge_index[0][idx:idx+CHUNK_SIZE]
            batch_col_index = edge_index[1][idx:idx+CHUNK_SIZE]
            a_batch = torch.index_select(a_norm, 0, batch_row_index)
            b_batch = torch.index_select(b_norm, 0, batch_col_index)
            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods
        return sims
    
    def cal_cos_sim(self,u_idx, i_idx, CHUNK_SIZE=65536):
        user_feature,item_feature = self.get_hidden_emb()
        L = u_idx.shape[0]
        sims = torch.zeros(L, dtype=user_feature.dtype).to(world.device)
        for idx in range(0, L, CHUNK_SIZE):
            a_batch = torch.index_select(user_feature, 0, u_idx[idx:idx + CHUNK_SIZE])
            b_batch = torch.index_select(item_feature, 0, i_idx[idx:idx + CHUNK_SIZE])
            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods
        return sims
    
    def graph_denoising(self):
        with torch.no_grad():
            hidden_user,hidden_item = self.get_hidden_emb()
            cos_sim = self.cal_cos_sim_sp(self.edge_index,hidden_user,hidden_item)
            cos_sim = (cos_sim + 1) / 2 
            cos_sim[cos_sim < self.prune_threshold] = 0
            self.value = cos_sim
            self.edge_index_norm = gcn_norm(self.edge_index_to_sparse_tensor(self.edge_index,cos_sim))
        
    def graph_augmentation(self):
        with torch.no_grad():
            num_edges = self.edge_index.size(1)
            aug_user = torch.randint(0, self.num_users,(int(self.aug_ratio * num_edges),), device=device)
            aug_item = torch.randint(0, self.num_items,(int(self.aug_ratio * num_edges),), device=device)
            cos_sim = self.cal_cos_sim(aug_user,aug_item)
            _, idx = torch.topk(cos_sim, int(self.aug_ratio * num_edges))
            aug_user = aug_user[idx].long()
            aug_item = aug_item[idx].long()
            aug_edge_index = torch.stack([aug_user,aug_item])
            aug_value = torch.ones_like(aug_user,device=device) * torch.median(self.value)
            aug_edge_index = torch.cat([self.edge_index, aug_edge_index], dim=1)
            aug_value = torch.cat([self.value,aug_value],dim=0)
            self.aug_graph = gcn_norm(self.edge_index_to_sparse_tensor(aug_edge_index,aug_value))
        
    def edge_index_to_sparse_tensor(self,edge_index,value):
        r,c = edge_index
        row_index = torch.cat([r , c + self.num_users])
        col_index = torch.cat([c + self.num_users ,r])
        value = torch.cat([value,value])
        return SparseTensor(row=row_index,col=col_index,value=value,sparse_sizes=(self.num_items + self.num_users , self.num_users + self.num_items))
    
    def forward(self):
        edge_index_C = self.edge_index_norm
        out = self.get_ssl_embedding(edge_index=edge_index_C)
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        return out_u, out_i
    
    def ssl_forward(self):
        edge_index_C = self.aug_graph
        out = self.get_ssl_embedding(edge_index=edge_index_C)
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        return out_u, out_i

    def get_embedding(self):
        x_u,x_i = self.get_hidden_emb()
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index_norm,x=x)
            out = out + x * self.alpha[i + 1]
        return out
    
    def bpr_loss(self,out_u,out_i,edge_label_index):
        out_src = out_u[edge_label_index[0]]
        out_dst = out_i[edge_label_index[1]]
        out_dst_neg = out_i[edge_label_index[2]]
        pos_rank = (out_src * out_dst).sum(dim=-1)
        neg_rank = (out_src * out_dst_neg).sum(dim=-1) 
        return torch.nn.functional.softplus(neg_rank - pos_rank).mean()

    def get_ssl_embedding(self,edge_index):
        x_u,x_i = self.get_hidden_emb()
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=edge_index,x=x)
            out = out + x * self.alpha[i + 1]
        return out

    def ssl_triple_loss(self, z1: torch.Tensor, z2: torch.Tensor, all_emb: torch.Tensor):
        norm_emb1 = F.normalize(z1)
        norm_emb2 = F.normalize(z2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.mul(norm_emb1, norm_emb2).sum(dim=1)
        ttl_score = torch.matmul(norm_emb1, norm_all_emb.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.ssl_tmp)
        ttl_score = torch.exp(ttl_score / self.ssl_tmp).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return self.ssl_decay * ssl_loss
    
    
    def L2_reg(self,edge_label_index):
        user_emb,item_emb = self.get_hidden_emb()
        u_idx,i_idx_pos,i_idx_neg = edge_label_index
        userEmb0 = user_emb[u_idx]
        posEmb0 = item_emb[i_idx_pos]
        negEmb0 = item_emb[i_idx_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2)) / edge_label_index.size(1)
        regularization = self.config['decay'] * reg_loss
        return regularization
    
device = world.device
dataset = Loader()
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = RGCF(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.

def train_bpr_rgcf(dataset,
                  model:RGCF,
                  opt):
    model = model
    model.train()
    S = utils.Full_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        out_u,out_i = model.forward()
        aug_u,_ = model.ssl_forward()
        all_u,_ = model.get_hidden_emb()
        bpr_loss = model.bpr_loss(out_u,out_i,edge_label_index)
        ssl_loss = model.ssl_triple_loss(out_u[edge_label_index[0]],aug_u[edge_label_index[0]],all_u)
        L2_reg = model.L2_reg(edge_label_index)
        loss = bpr_loss + ssl_loss + L2_reg 
        opt.zero_grad()
        loss.backward()
        opt.step()    
        aver_loss += (bpr_loss + ssl_loss + L2_reg)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"


for epoch in range(1, 1001):
    edge_index = train_edge_index
    model.graph_denoising()
    model.graph_augmentation()
    start_time = time.time()
    loss = train_bpr_rgcf(dataset=dataset,
                         model=model,
                         opt=opt)
    end_time = time.time()
    recall,ndcg = test([20],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f}, N@20: {ndcg[20]:.4f}, '
          f'time:{end_time-start_time:.2f} seconds')