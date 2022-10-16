'''
Created on Oct 4, 2020

@author: Yinfeng Li
'''

import world
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp


def loss_dependence(emb1, emb2):
    dim = m = emb1.size(0)
    R = torch.eye(dim).to(world.device)  - (1/dim) * torch.ones(dim, dim).to(world.device) 
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC / ((dim-1)**2)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    
    def attnw(self):
        return self.attnw
    
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        self.attnw = beta
        return (beta * z).sum(1)



class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getRating(self, users, locations, times):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, locations, times, pos, neg):
        """
        Parameters:
            users: users list 
            locations: locations list
            times: times list
            pos: positive items for corresponding Scenes
            neg: negative items for corresponding Scenes
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class DisenHCN(BasicModel):
    def __init__(self, 
                 config:dict
                 ):
        """
        Disentangled Hypergraph Convolutional Networks for spatiotemporal activity predction.
        config: parameters
        """
        super(DisenHCN, self).__init__()
        self.config = config
        self.num_users = self.config['num_users']
        self.num_locations = self.config['num_locations']
        self.num_times = self.config['num_times']
        self.num_activities  = self.config['num_activities']
        self.latent_dim = self.config['emb_size']   # the embedding size of location, time and activity are d/3
        self.n_layers = self.config['layer']
        self.use_drop = self.config['use_drop']
        self.keep_prob = 1 - self.config['droprate']
        self.dropout = self.config['droprate']
        self.cor_flag = 1
        self.n_factors = 3

        self.split_sections = [self.latent_dim//self.n_factors] * self.n_factors

        
        # Load hypergraph agg matrix: norm_vtoe & norm_etov from location, time and activity aspect
        norm_VtoE_l = sp.load_npz(world.DATA_PATH + '/norm_vtoe_l.npz')
        self.VtoE_l = self._convert_sp_mat_to_sp_tensor(norm_VtoE_l)
        self.VtoE_l = self.VtoE_l.coalesce().to(world.device) 
        norm_VtoE_t = sp.load_npz(world.DATA_PATH + '/norm_vtoe_t.npz')
        self.VtoE_t = self._convert_sp_mat_to_sp_tensor(norm_VtoE_t)
        self.VtoE_t = self.VtoE_t.coalesce().to(world.device) 
        norm_VtoE_a = sp.load_npz(world.DATA_PATH + '/norm_vtoe_a.npz')
        self.VtoE_a = self._convert_sp_mat_to_sp_tensor(norm_VtoE_a)
        self.VtoE_a = self.VtoE_a.coalesce().to(world.device) 

        norm_EtoV_l = sp.load_npz(world.DATA_PATH + '/norm_etov_l.npz')
        self.EtoV_l = self._convert_sp_mat_to_sp_tensor(norm_EtoV_l)
        self.EtoV_l = self.EtoV_l.coalesce().to(world.device) 
        norm_EtoV_t = sp.load_npz(world.DATA_PATH + '/norm_etov_t.npz')
        self.EtoV_t = self._convert_sp_mat_to_sp_tensor(norm_EtoV_t)
        self.EtoV_t = self.EtoV_t.coalesce().to(world.device) 
        norm_EtoV_a = sp.load_npz(world.DATA_PATH + '/norm_etov_a.npz')
        self.EtoV_a = self._convert_sp_mat_to_sp_tensor(norm_EtoV_a)
        self.EtoV_a = self.EtoV_a.coalesce().to(world.device) 

        # Load User Similar Sub-hypergraph: E^L(Norm_L), E^L(Norm_T), E^L(Norm_A), E^L(Norm_LT), E^L(Norm_LA), E^L(Norm_TA), E^L(Norm_LTA)
        Norm_L = sp.load_npz(world.DATA_PATH + '/Norm_L.npz')
        self.Norm_L = self._convert_sp_mat_to_sp_tensor(Norm_L)
        self.Norm_L = self.Norm_L.coalesce().to(world.device)

        Norm_T = sp.load_npz(world.DATA_PATH + '/Norm_T.npz')
        self.Norm_T = self._convert_sp_mat_to_sp_tensor(Norm_T)
        self.Norm_T = self.Norm_T.coalesce().to(world.device)

        Norm_A = sp.load_npz(world.DATA_PATH + '/Norm_A.npz')
        self.Norm_A = self._convert_sp_mat_to_sp_tensor(Norm_A)
        self.Norm_A = self.Norm_A.coalesce().to(world.device)

        Norm_LT = sp.load_npz(world.DATA_PATH + '/Norm_LT.npz')
        self.Norm_LT = self._convert_sp_mat_to_sp_tensor(Norm_LT)
        self.Norm_LT = self.Norm_LT.coalesce().to(world.device)

        Norm_LA = sp.load_npz(world.DATA_PATH + '/Norm_LA.npz')
        self.Norm_LA = self._convert_sp_mat_to_sp_tensor(Norm_LA)
        self.Norm_LA = self.Norm_LA.coalesce().to(world.device)

        Norm_TA = sp.load_npz(world.DATA_PATH + '/Norm_TA.npz')
        self.Norm_TA = self._convert_sp_mat_to_sp_tensor(Norm_TA)
        self.Norm_TA = self.Norm_TA.coalesce().to(world.device)

        Norm_LTA = sp.load_npz(world.DATA_PATH + '/Norm_LTA.npz')
        self.Norm_LTA = self._convert_sp_mat_to_sp_tensor(Norm_LTA)
        self.Norm_LTA = self.Norm_LTA.coalesce().to(world.device)

        self.__init__weight()

    def __init__weight(self):
        
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_base = torch.nn.Embedding(
            num_embeddings=self.num_locations, embedding_dim=self.latent_dim//3)
        self.embedding_time = torch.nn.Embedding(
            num_embeddings=self.num_times, embedding_dim=self.latent_dim//3)
        self.embedding_app  = torch.nn.Embedding(
            num_embeddings=self.num_activities,  embedding_dim=self.latent_dim//3)
        if self.config['pretrain'] == 0:
            """
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_base.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_time.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_app.weight,  gain=1)
            print('use xavier initilizer')
            """
        # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_base.weight, std=0.1)
            nn.init.normal_(self.embedding_time.weight, std=0.1)
            nn.init.normal_(self.embedding_app.weight,  std=0.1)
            world.cprint('use NORMAL distribution initilizer')
            
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_base.weight.data.copy_(torch.from_numpy(self.config['base_emb']))
            self.embedding_time.weight.data.copy_(torch.from_numpy(self.config['time_emb']))
            self.embedding_app.weight.data.copy_(torch.from_numpy(self.config['activity_emb']))
            print('use pretarined data')
        

        self.__init__attn()

        self.f = nn.Sigmoid() # for final prediction
        self.disen_l = nn.Linear(self.latent_dim, self.latent_dim//3, bias=True)
        self.disen_t = nn.Linear(self.latent_dim, self.latent_dim//3, bias=True)
        self.disen_a = nn.Linear(self.latent_dim, self.latent_dim//3, bias=True)
        self.ln = nn.LayerNorm(normalized_shape=self.latent_dim//3, elementwise_affine=False)
        
        
        print(f"DisenHCN is already to go(use_dropout:{self.config['use_drop']})")

    def __init__attn(self):
        # user hypergraph
        print('init Inter-type Attention weight...')
        self.attention_l = Attention(hidden_size=self.latent_dim//3)
        self.attention_t = Attention(hidden_size=self.latent_dim//3)
        self.attention_a = Attention(hidden_size=self.latent_dim//3)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        
        return self.__dropout_x(self.VtoE_l, keep_prob), self.__dropout_x(self.VtoE_t, keep_prob), self.__dropout_x(self.VtoE_a, keep_prob), \
               self.__dropout_x(self.EtoV_l, keep_prob), self.__dropout_x(self.EtoV_t, keep_prob), self.__dropout_x(self.EtoV_a, keep_prob), \
               self.__dropout_x(self.Norm_L, keep_prob), \
               self.__dropout_x(self.Norm_T, keep_prob), self.__dropout_x(self.Norm_A, keep_prob), self.__dropout_x(self.Norm_LT, keep_prob), \
               self.__dropout_x(self.Norm_LA, keep_prob), self.__dropout_x(self.Norm_TA, keep_prob), self.__dropout_x(self.Norm_LTA, keep_prob)
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def computer(self):
        """
        Calculate Att-HGConv on Hypergraph: intra-type -> inter-type
        """      
        users_emb = self.embedding_user.weight
        locations_emb = self.embedding_base.weight
        times_emb = self.embedding_time.weight
        activities_emb  = self.embedding_app.weight
        
        if self.use_drop:
            if self.training:
                #print("droping")
                VtoE_l, VtoE_t, VtoE_a, EtoV_l, EtoV_t, EtoV_a, Norm_L,  Norm_T,  Norm_A,  Norm_LT,  Norm_LA,  Norm_TA,  Norm_LTA= self.__dropout(self.keep_prob)
            
        else:
            VtoE_l, VtoE_t, VtoE_a, EtoV_l, EtoV_t, EtoV_a, Norm_L,  Norm_T,  Norm_A,  Norm_LT,  Norm_LA,  Norm_TA,  Norm_LTA = \
            self.VtoE_l, self.VtoE_t, self.VtoE_a, self.EtoV_l, self.EtoV_t, self.EtoV_a, \
            self.Norm_L,  self.Norm_T,  self.Norm_A,  self.Norm_LT,  self.Norm_LA,  self.Norm_TA,  self.Norm_LTA

        # Disentangled Embedding Transformation
        '''
        u_emb_l = self.ln(self.disen_l(users_emb))
        u_emb_t = self.ln(self.disen_t(users_emb))
        u_emb_a = self.ln(self.disen_a(users_emb))
        '''
        # slice chunk for disentanglement
        user_emb_split = torch.split(users_emb, self.split_sections, dim=-1)
        u_emb_l = user_emb_split[0]
        u_emb_t = user_emb_split[1]
        u_emb_a = user_emb_split[2]
          
        u_emb_l_all = [u_emb_l]
        u_emb_t_all = [u_emb_t]
        u_emb_a_all = [u_emb_a]

        l_emb = locations_emb
        l_emb_all = [l_emb]
        t_emb = times_emb
        t_emb_all = [t_emb]
        a_emb = activities_emb
        a_emb_all = [a_emb]

      
        for k in range(self.n_layers):
            # Intra-type propagation

            # User Similarity Hyperedges with Efficient HGConv
            # E^L
            u_emb_l1 = torch.sparse.mm(Norm_L, u_emb_l)
            # E^T
            u_emb_t1 = torch.sparse.mm(Norm_T, u_emb_t)
            # E^A
            u_emb_a1 = torch.sparse.mm(Norm_A, u_emb_a)
            # E^LT
            u_emb_l2 = torch.sparse.mm(Norm_LT, u_emb_l)
            u_emb_t2 = torch.sparse.mm(Norm_LT, u_emb_t)
            # E^LA
            u_emb_l3 = torch.sparse.mm(Norm_LA, u_emb_l)
            u_emb_a2 = torch.sparse.mm(Norm_LA, u_emb_a)
            # E^TA
            u_emb_t3 = torch.sparse.mm(Norm_TA, u_emb_t)
            u_emb_a3 = torch.sparse.mm(Norm_TA, u_emb_a)
            # E^LTA
            u_emb_l4 = torch.sparse.mm(Norm_LTA, u_emb_l)
            u_emb_t4 = torch.sparse.mm(Norm_LTA, u_emb_t)
            u_emb_a4 = torch.sparse.mm(Norm_LTA, u_emb_a)

            # User Interaction Hyperedges with two-step HGConv
            # node to hyperedge
            u_emb_l5 = torch.sparse.mm(VtoE_l, l_emb)
            u_emb_t5 = torch.sparse.mm(VtoE_t, t_emb)
            u_emb_a5 = torch.sparse.mm(VtoE_a, a_emb) 

            # hyperedge to node 
            l_emb = torch.sparse.mm(EtoV_l, u_emb_l)
            t_emb = torch.sparse.mm(EtoV_t, u_emb_t)
            a_emb = torch.sparse.mm(EtoV_a, u_emb_a)

            l_emb_all.append(l_emb)
            t_emb_all.append(t_emb)
            a_emb_all.append(a_emb)

            # inter-type propagation
            '''
            u_emb_l = self.attention_l(torch.stack([u_emb_l1, u_emb_l2, u_emb_l3, u_emb_l4, u_emb_l5], dim=1))
            u_emb_t = self.attention_t(torch.stack([u_emb_t1, u_emb_t2, u_emb_t3, u_emb_t4, u_emb_t5], dim=1))
            u_emb_a = self.attention_a(torch.stack([u_emb_a1, u_emb_a2, u_emb_a3, u_emb_a4, u_emb_a5], dim=1))
            '''
            u_emb_l = self.attention_l(torch.stack([u_emb_l1, u_emb_l, u_emb_l, u_emb_l2, u_emb_l3, u_emb_l, u_emb_l4, u_emb_l5], dim=1))
            u_emb_t = self.attention_t(torch.stack([u_emb_t, u_emb_t1, u_emb_t, u_emb_t2, u_emb_t, u_emb_t3, u_emb_t4, u_emb_t5], dim=1))
            u_emb_a = self.attention_a(torch.stack([u_emb_a, u_emb_a, u_emb_a1, u_emb_a, u_emb_a2, u_emb_a3, u_emb_a4, u_emb_a5], dim=1))

            u_emb_l_all.append(u_emb_l)
            u_emb_t_all.append(u_emb_t)
            u_emb_a_all.append(u_emb_a)
            
            
        users_l = torch.mean(torch.stack(u_emb_l_all, dim=1), dim=1)
        users_t = torch.mean(torch.stack(u_emb_t_all, dim=1), dim=1)
        users_a = torch.mean(torch.stack(u_emb_a_all, dim=1), dim=1)
        
        locations = torch.mean(torch.stack(l_emb_all, dim=1), dim=1)
        times = torch.mean(torch.stack(t_emb_all, dim=1), dim=1)
        activities = torch.mean(torch.stack(a_emb_all, dim=1), dim=1)
        
        return users_l, users_t, users_a, locations, times, activities 

    
    def getRating(self, users, locations, times):
        all_users_l, all_users_t, all_users_a, all_locations, all_times, all_activities = self.computer()
        users = users.long()
        locations = locations.long()
        times = times.long()
        users_emb_l = all_users_l[users]
        users_emb_t = all_users_t[users]
        users_emb_a = all_users_a[users]
        locations_emb = all_locations[locations]
        times_emb = all_times[times]
        activities_emb  = all_activities
        scores = torch.sum(users_emb_l*locations_emb + users_emb_t*times_emb, dim=1, keepdim=True) + torch.matmul(users_emb_a, activities_emb.t())
        
        return self.f(scores)

    def getEmbedding(self, users, locations, times, pos, neg):
        all_users_l, all_users_t, all_users_a, all_locations, all_times, all_activities = self.computer()
        users_emb_l = all_users_l[users]
        users_emb_t = all_users_t[users]
        users_emb_a = all_users_a[users]
        locations_emb = all_locations[locations]
        times_emb = all_times[times]
        pos_emb = all_activities[pos]
        neg_emb = all_activities[neg]
        

        # calculate loss_independent
        loss_dep = self.create_cor_loss(users_emb_l, users_emb_t, users_emb_a)

        users_emb_ego = self.embedding_user(users)
        locations_emb_ego = self.embedding_base(locations)
        times_emb_ego = self.embedding_time(times)
        pos_emb_ego   = self.embedding_app(pos)
        neg_emb_ego   = self.embedding_app(neg)

        return users_emb_l, users_emb_t, users_emb_a, locations_emb, times_emb, pos_emb, neg_emb, users_emb_ego, locations_emb_ego, times_emb_ego, pos_emb_ego, neg_emb_ego, loss_dep
    
    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            r = torch.sum(torch.square(X), dim=1, keepdim=True)
            D = torch.sqrt(torch.max(r - 2 * torch.matmul(X,X.t()) + r.t(), torch.zeros(1).cuda()) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) \
                + torch.mean(D)
            return D
        
        def _create_distance_covariance(D1, D2):
            #calculate distance covariance between D1 and D2
            n_samples = D1.shape[0]
            dcov = torch.sqrt(torch.max(torch.sum(D1 * D2) / (n_samples * n_samples), torch.zeros(1).cuda()) + 1e-8)
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        #calculate the distance correlation
        dcor = dcov_12 / (torch.sqrt(torch.max(dcov_11 * dcov_22, torch.zeros(1).cuda())) + 1e-10)
        return dcor


    def create_cor_loss(self, users_emb_l, users_emb_t, users_emb_a):
        cor_loss = torch.zeros(1)
        if self.cor_flag == 0:
            return cor_loss
        
        cor_loss = self._create_distance_correlation(users_emb_l, users_emb_t) + self._create_distance_correlation(users_emb_l, users_emb_a) + \
                   self._create_distance_correlation(users_emb_t, users_emb_a)
        
        cor_loss /= ((self.n_factors + 1.0) * self.n_factors/2)

        return cor_loss

    
    def bpr_loss(self, users, locations, times, pos, neg):
        (users_emb_l, users_emb_t, users_emb_a, locations_emb, times_emb, pos_emb, neg_emb, 
        userEmb0, baseEmb0, timeEmb0, posEmb0, negEmb0, loss_dep) = self.getEmbedding(users.long(), locations.long(),times.long(), pos.long(), neg.long())

        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         baseEmb0.norm(2).pow(2) + 
                         timeEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))

        pos_scores= torch.sum(users_emb_l*locations_emb + users_emb_t*times_emb + users_emb_a*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb_l*locations_emb + users_emb_t*times_emb + users_emb_a*neg_emb, dim=1) #[2048]
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss += self.config['gamma']*loss_dep[0]   

        return loss, reg_loss
    
    def forward(self, users, locations, times, activities):
        """
        Input:(users, locations, times, activities)
        """
        # compute embedding
        all_users_l, all_users_t, all_users_a, all_locations, all_times, all_activities = self.computer()
        users_emb_l = all_users_l[users]
        users_emb_t = all_users_t[users]
        users_emb_a = all_users_a[users]
        locations_emb = all_locations[locations]
        times_emb = all_times[times]
        activities_emb  = all_activities[activities]
        inner_pro = users_emb_l*locations_emb + users_emb_t*times_emb + users_emb_a*activities_emb
        x     = torch.sum(inner_pro, dim=1)
        return x
