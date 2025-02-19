from model import RecModel
from torch import LongTensor,Tensor
import torch.nn as nn
from torch_geometric.typing import Adj
import torch
from dataloader import Loader
import world
from procedure import train_bpr,test
import utils
import time
import torch.nn.functional as F
from torch_sparse import sum, mul, fill_diag, remove_diag,SparseTensor

if world.config['dataset'] == 'yelp2018':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':0.1,#INIT WEIGHT
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'lambda':3,
    }

if world.config['dataset'] == 'amazon-book':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':0.1,#INIT WEIGHT
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'lambda':1
    }

if world.config['dataset'] == 'gowalla':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':0.1,#INIT WEIGHT
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'lambda':4,
    }

if world.config['dataset'] == 'iFashion':
    config = {
        'init':'uniform',#UNIFORM DISTRIBUTION
        'init_weight':1,#INIT WEIGHT
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'lambda':4
    }

class GTN(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            config=config,
            edge_index=edge_index
        )
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['dim'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['dim'])
        self.init_weight()
        self.K = config['K']
        edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=None)
        # self.edge_index = gcn_norm(edge_index)
        self.incident_matrix = self.get_incident_matrix(edge_index)
        self.incident_matrix = self.inc_norm(self.incident_matrix,edge_index,add_self_loops=True)
        self.init_z = torch.zeros((self.incident_matrix.sizes()[0], config['dim'])).cuda()
        self.lambda1 = config['lambda']
        print('Go GTN')
        print(f"params settings: \n emb_size:{config['dim']}\n L2 reg:{config['decay']}\n layer:{self.K}\n lambda:{config['lambda']}")

    def init_weight(self):
        if config['init'] == 'normal':
            nn.init.normal_(self.user_emb.weight,std=config['init_weight'])
            nn.init.normal_(self.item_emb.weight,std=config['init_weight'])
        else:
            nn.init.xavier_uniform_(self.user_emb.weight,gain=config['init_weight'])
            nn.init.xavier_uniform_(self.item_emb.weight,gain=config['init_weight'])
        
    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        hh = x
        z = self.init_z.detach()
        for i in range(self.K):
            grad = x - hh
            smoo = x - grad
            temp = z + 0.5 * (self.incident_matrix @ (smoo - (self.incident_matrix.t() @ z)))
            z = self.proximal_l1_conjugate(x=temp, lambda2=self.lambda1)
            ctz = self.incident_matrix.t() @ z
            x = smoo - ctz
            x = F.dropout(x, p=0.1, training=self.training)
        return x
    
    def forward(self,
                    edge_label_index:Tensor):
            out = self.get_embedding()
            out_u,out_i = torch.split(out,[self.num_users,self.num_items])
            out_src = out_u[edge_label_index[0]]
            out_dst = out_i[edge_label_index[1]]
            out_dst_neg = out_i[edge_label_index[2]]
            return (out_src * out_dst).sum(dim=-1),(out_src * out_dst_neg).sum(dim=-1)

    def proximal_l1_conjugate(self, x: Tensor, lambda2):
        x = torch.clamp(x, min=-lambda2, max=lambda2)
        return x
    
    def inc_norm(self, inc, edge_index, add_self_loops):
        if add_self_loops:
            edge_index = fill_diag(edge_index, 1.0)
        else:
            edge_index = remove_diag(edge_index)
        deg = sum(edge_index, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        inc = mul(inc, deg_inv_sqrt.view(1, -1))
        return inc
    
    def get_incident_matrix(self, edge_index: Adj):
        size = edge_index.sizes()[1]
        row_index = edge_index.storage.row()
        col_index = edge_index.storage.col()
        mask = row_index >= col_index
        row_index = row_index[mask]
        col_index = col_index[mask]
        edge_num = row_index.numel()
        row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)]).cuda()
        col = torch.cat([row_index, col_index])
        value = torch.cat([torch.ones(edge_num), -1 * torch.ones(edge_num)]).cuda()
        inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
                           sparse_sizes=(edge_num, size))
        return inc
    

device = world.device
dataset = Loader()
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = GTN(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
for epoch in range(1, 1001):
    start_time = time.time()
    loss = train_bpr(dataset=dataset,model=model,opt=opt)
    end_time = time.time()
    recall,ndcg = test([20],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f}, N@20: {ndcg[20]:.4f}, '
          f'time:{end_time-start_time:.2f} seconds')