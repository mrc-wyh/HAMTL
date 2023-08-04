import pandas as pd
import numpy as np
# import torch
# import random as rd
import scipy.sparse as sp
import dgl
import pickle
import os, sys
import torch
os.chdir(sys.path[0])

class Data(object):
    def __init__(self, args):
        dataset = args.dataset
        
        loc_file = dataset + '/loc.csv'
        if args.dist == 1000:
            geo_edge_file = dataset + '/geo_edge_1000.csv'
        elif args.dist == 500:
            geo_edge_file = dataset + '/geo_edge_500.csv'
        elif args.dist == 1500:
            geo_edge_file = dataset + '/geo_edge_1500.csv'
        else:
            geo_edge_file = dataset + '/geo_edge_2000.csv'
        tran_edge_file = dataset + '/tran_edge.csv'
        user_id_file = dataset + '/user_id.csv'
        train_forward_file = dataset + '/train_forward.pickle'
        train_labels_file = dataset + '/train_labels.pickle'
        train_user_file = dataset + '/train_user.pickle'
        valid_forward_file = dataset + '/valid_forward.pickle'
        valid_labels_file = dataset + '/valid_labels.pickle'
        valid_user_file = dataset + '/valid_user.pickle'
        test_forward_file = dataset + '/test_forward.pickle'
        test_lables_file = dataset + '/test_labels.pickle'
        test_user_file = dataset + '/test_user.pickle'
        hier_trans_file = dataset + '/hier_trans.csv'
        loc_in_cat_file = dataset + '/loc_in_cat.csv'
                
        loc = pd.read_csv(loc_file, names=['loc_ID', 'loc_cat_new_name', 'cat_id', 'loc_catin_id', 'latitude', 'longitude', 'loc_new_ID'], sep=',', header=0)
        user = pd.read_csv(user_id_file, names=['old_id', 'new_id'], sep=',', header=0)
        geo_edge = pd.read_csv(geo_edge_file, names=['src', 'dst'], sep=',', header=0)
        tran_edge = pd.read_csv(tran_edge_file, names=['src', 'dst', 'freq', 'weight'], sep=',', header=0)
        hier_trans = pd.read_csv(hier_trans_file, names=['src', 'dst', 'freq'], sep=',', header=0) 
        loc_in_cat_e = pd.read_csv(loc_in_cat_file, names=['src', 'dst'], sep=',', header=0)       
        self.cat_num = max(loc['cat_id']) + 1
        self.loc_num = max(loc['loc_new_ID']) + 1
        self.user_num = max(user['new_id']) + 1
        loc_cat = loc[['loc_new_ID', 'cat_id', 'loc_catin_id']]
        loc_cat.sort_values(by='loc_new_ID', ascending=True, inplace=True)
        loc_map_cat = list(loc_cat['cat_id'])
        self.loc_cat = loc_map_cat
        if args.cat_learnable == 2:
            self.loc_g, self.tran_edge_weight = self.build_graph(geo_edge, tran_edge)
            self.hier_tree = self.build_hier_tree(loc_in_cat_e)
        else:
            self.loc_g, self.tran_edge_weight = self.build_graph(geo_edge, tran_edge)               
        self.trans_matrix = self.tran_matrix(tran_edge)      
        self.hier_trans =  self.hier_tran_matrix(hier_trans, args.cat_to_loc_piror) if args.cat_to_loc_piror else self.hier_tran_matrix(hier_trans, 1)
        
        train_forward = open(train_forward_file,'rb')
        self.train_forward = pickle.load(train_forward)
        train_labels = open(train_labels_file,'rb')
        self.train_labels = pickle.load(train_labels)
        train_user = open(train_user_file,'rb')
        self.train_user = pickle.load(train_user)
        valid_forward = open(valid_forward_file,'rb')
        self.valid_forward = pickle.load(valid_forward)
        valid_labels = open(valid_labels_file,'rb')
        self.valid_labels = pickle.load(valid_labels)
        valid_user = open(valid_user_file,'rb')
        self.valid_user = pickle.load(valid_user)
        test_forward = open(test_forward_file,'rb')
        self.test_forward = pickle.load(test_forward)
        test_labels = open(test_lables_file,'rb')
        self.test_labels = pickle.load(test_labels)
        test_user = open(test_user_file,'rb')
        self.test_user = pickle.load(test_user)
        print('train traj num:', len(self.train_forward))
        print('valid traj num:', len(self.valid_forward))
        print('test traj num:', len(self.test_forward))
        
    def build_graph(self, geo_edge, tran_edge):
        geo = np.array(geo_edge)
        geo_e = [tuple(geo[i]) for i in range(len(geo))]
        tran = np.array(tran_edge[['src', 'dst']])
        tran_e_w = np.array(tran_edge['weight'])
        tran_e = [tuple(tran[i]) for i in range(len(tran))]

        data_dict = {
            ('loc', 'geo', 'loc'): geo_e,
            ('loc', 'trans', 'loc'): tran_e
            }
        return dgl.heterograph(data_dict), tran_e_w
    
    def build_hier_tree(self, loc_in_cat_e):
        l_c = np.array(loc_in_cat_e)
        l_c = [tuple(l_c[i]) for i in range(len(l_c))]
        
        data_dict = {
            ('loc', 'in', 'cat'): l_c
            }

        return dgl.heterograph(data_dict)
    
    def tran_matrix(self, tran_edge):
        trans = tran_edge[['src', 'dst', 'freq']]
        trans = trans.sort_values(by='src').replace(True)
        trans.index = range(len(trans))
        gtedge_g = trans.groupby(by='src')
        trans_prob = pd.DataFrame()
        for src, e in gtedge_g:
            e.index = range(len(e))
            total_freq = e['freq'].sum()
            e['weight'] = e['freq'] / total_freq
            trans_prob = pd.concat((trans_prob, e))
        trans_prob = trans_prob.sort_values(by=['src', 'dst']).replace(True)
        trans_prob = trans_prob[['src', 'dst', 'weight']]
        trans_prob.index = range(len(trans_prob))
        row = np.array(trans_prob['src'])
        col = np.array(trans_prob['dst'])
        data = np.array(trans_prob['weight'])
        
        return sp.coo_matrix((data, (row, col)), shape=(self.loc_num, self.loc_num), dtype=np.float)
    
    def hier_tran_matrix(self, tran_edge, cat_to_loc_piror):
        if cat_to_loc_piror == 2:
            tran_edge['freq'] = 1
        gtedge_g = tran_edge.groupby(by='src')
        trans_prob = pd.DataFrame()
        for src, e in gtedge_g:
            e.index = range(len(e))
            total_freq = e['freq'].sum()
            e['weight'] = e['freq'] / total_freq
            trans_prob = pd.concat((trans_prob, e))
        trans_prob = trans_prob[['src', 'dst', 'weight']]
        row = np.array(trans_prob['src'])
        col = np.array(trans_prob['dst'])
        data = np.array(trans_prob['weight'])
        indice = torch.tensor(np.array([row, col]))
        value = torch.tensor(data)
        trans = torch.sparse_coo_tensor(indices=indice, values=value, size=[self.cat_num, self.loc_num], dtype=torch.float32).to_dense()
        return trans