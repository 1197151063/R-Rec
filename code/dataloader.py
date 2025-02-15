from torch.utils.data import Dataset
from world import cprint
import world
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.utils import degree
seed = world.seed
    
class Loader(Dataset):
    """
    Loading data from datasets
    supporting:['gowalla','amazon-book','yelp2018','Alibaba-iFashion']
    """
    def __init__(self,config=world.config,path='../data/'):
        dir_path = path + config['dataset']
        cprint(f'loading from {dir_path}')
        self.n_user = 0
        self.n_item = 0
        train_file = dir_path + '/train.txt'
        test_file = dir_path + '/test.txt'
        train_edge_index = []
        test_edge_index = []
        testUser = []
        testItem = []
        trainUser = []
        trainItem = []
       
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    all = l.strip('\n').split(' ')
                    uid = int(all[0])
                    train_items = [int(i) for i in all[1:]]
                    for item in train_items:
                        train_edge_index.append([uid,item])
                    trainUser.extend([uid] * len(train_items))
                    trainItem.extend(train_items)
        train_edge_index = torch.LongTensor(np.array(train_edge_index).T)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    all = l.strip('\n').split(' ')
                    uid = int(all[0])
                    try:
                        items = [int(i) for i in all[1:]]
                        for item in items:
                            test_edge_index.append([uid,item])
                    except Exception:
                        continue
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
        test_edge_index = torch.LongTensor(np.array(test_edge_index).T)


        edge_index = torch.cat((train_edge_index,test_edge_index),1)
        self.train_edge_index = train_edge_index
        self.edge_index = edge_index
        self.n_user = edge_index[0].max() + 1
        self.n_item = edge_index[1].max() + 1
        self.test_edge_index = test_edge_index
        self.sampling_weights = self.get_edge_weights(train_edge_index)
        print(f"{world.dataset} is ready to go")

    @property
    def num_users(self):
        return self.n_user
    @property
    def num_items(self):
        return self.n_item

    

    '''
    A = |0   R|
        |R^T 0|
    R : user-item bipartite graph
    '''
    def getSparseGraph(self):
        cprint("generate Adjacency Matrix A")
        user_index = self.train_edge_index[0]
        item_index = self.train_edge_index[1]
        row_index = torch.cat([user_index,item_index+self.n_user])
        col_index = torch.cat([item_index+self.n_user,user_index])
        return SparseTensor(row=row_index,col=col_index,sparse_sizes=(self.n_item+self.n_user,self.n_item+self.n_user))

    def getSparseBipartite(self):
        user_index = self.train_edge_index[0]
        item_index = self.train_edge_index[1]
        return SparseTensor(row=user_index,col=item_index,sparse_sizes=(self.num_users,self.num_items))
    
    def get_edge_weights(self,edge_index):
        user_degree = degree(edge_index[0])
        sampling_weight = 1 / user_degree[edge_index[0]]
        return sampling_weight

