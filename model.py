import torch
import torch.nn as nn
from torch.nn import GRU
import dgl
import dgl.function as fn
import math
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os, sys
os.chdir(sys.path[0])

class GeoGCNLayer(nn.Module):
    def __init__(self,
                g,
                args,
                device='cuda'
                ):
        super(GeoGCNLayer, self).__init__()
        self.g = g
        self.device = device
        self.act = nn.LeakyReLU(0.2)
        self.args = args
        self.attn_fuse = SemanticAttention(args.loc_dim, args.loc_dim*4)

    def forward(self, loc_feat):
        funcs = {}#message and reduce functions dict
        self.g.ndata['f'] = loc_feat
        for srctype, etype, dsttype in self.g.canonical_etypes:
            if etype == 'geo':
                funcs[etype] = (fn.copy_u('f', 'm'), fn.mean('m', 'geo'))
            else:
                funcs[etype] = (fn.u_mul_e('f', 'w', 'm'), fn.sum('m', 'trans'))
                    
        self.g.multi_update_all(funcs, 'sum')
        geo = self.g.ndata['geo'].unsqueeze(1)
        trans = self.g.ndata['trans'].unsqueeze(1)
        z = torch.cat([geo, trans], 1)
        loc_feat = self.attn_fuse(z)
        return loc_feat

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        #vectore-level
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
    #vectore-level
    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1) 
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) 
        return (beta * z).sum(1)

class HierTree(nn.Module):
    def __init__(self,
                g,
                args,
                device='cuda'
                ):
        super(HierTree, self).__init__()
        self.g = g.to(device)
        self.device = device
        self.act = nn.LeakyReLU(0.2)
        self.loc_to_cat = nn.Linear(args.loc_dim, args.cat_dim)
        self.init_weights() 

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.loc_to_cat.bias)
        nn.init.uniform_(self.loc_to_cat.weight, -initrange, initrange)

    def forward(self, loc_emb):
        funcs = {}
        self.g.nodes['loc'].data['f'] = loc_emb
        for srctype, etype, dsttype in self.g.canonical_etypes:
            funcs[etype] = (fn.copy_u('f', 'm'), fn.sum('m', 'in'))
                    
        self.g.multi_update_all(funcs, 'sum')
        cat_emb = self.act(self.loc_to_cat(self.g.nodes['cat'].data['in']))
        return cat_emb
                                                                    
class GeoGCN(nn.Module):
    def __init__(self,
                g,
                tree,
                tran_e_w,
                args,
                device='cuda'
                ):
        super(GeoGCN, self).__init__()
        g = g.int()
        g = dgl.remove_self_loop(g, etype='geo')
        g = dgl.add_self_loop(g, etype='geo')
        self.g = g.to(device)
        self.g.edges['trans'].data['w'] = torch.tensor(tran_e_w).float().to(device)
        self.num_layer = args.GeoGCN_layer_num
        self.device = device
        self.act = nn.LeakyReLU(0.2)
        self.cat_learnable = args.cat_learnable
            
        self.gcn = nn.ModuleList()
        for i in range(self.num_layer):
            self.gcn.append(
            GeoGCNLayer(self.g, args, device)
        )
        if args.cat_learnable == 2:
            self.tree = HierTree(tree, args, device)
            
    def forward(self, loc_emb):
        for i in range(self.num_layer - 1):
            loc_emb = self.gcn[i](loc_emb)
        loc_emb = self.gcn[-1](loc_emb)
        if self.cat_learnable == 2:
            cat_emb = self.tree(loc_emb)
            return loc_emb, cat_emb
        else:
            return loc_emb

class SlotEncoding(nn.Module):
    "Position Encoding module"
    def __init__(self, dim_model, max_len=500, device='cuda'):
        super(SlotEncoding, self).__init__()
        pe = torch.zeros(max_len, dim_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2, dtype=torch.float) *
                             -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)  # Not a parameter but should be in state_dict
    
    def forward(self, pos):
        return torch.index_select(self.pe, 1, pos).squeeze(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x) 

class SeqPred(nn.Module):
    def __init__(self,
                cat_num,
                loc_num,
                trans,
                args,
                dropout=0.1,
                device='cuda'
                ):
        super(SeqPred, self).__init__()

        self.dim = args.user_dim + args.loc_dim + args.cat_dim + args.time_dim
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(int(self.dim), args.enc_drop)
        encoder_layers = TransformerEncoderLayer(int(self.dim), args.enc_nhead, args.enc_ffn_hdim, args.enc_drop)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.enc_layer_num)
        #decoder
        self.decoder_cat = nn.Linear(self.dim, cat_num)
        if args.decoder_mode:
            self.fc = nn.Linear(self.dim, loc_num)
        else:
            self.fc = nn.Linear(cat_num + loc_num, loc_num) if args.cat_to_loc_piror else nn.Linear(cat_num, loc_num)
        self.trans = trans
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)
        self.args = args   
        self.init_weights()
        self.device = device

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder_cat.bias)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.decoder_cat.weight, -initrange, initrange)
        nn.init.uniform_(self.fc.weight, -initrange, initrange)
    
    def forward(self, src, key_pad_mask, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        
        src = src * math.sqrt(self.dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask, key_pad_mask.transpose(0, 1)) #shape * batch *dim
        cat_out = self.decoder_cat(output)
        if self.args.decoder_mode:
            loc_out = self.fc(output)
        else:
            if self.args.cat_to_loc_piror:
                loc_prior = torch.matmul(cat_out, self.trans)            
                loc_out = torch.cat((cat_out, loc_prior), dim=-1)
                loc_out = self.fc(self.dropout(self.act(loc_out)))
            else:
                loc_out = self.fc(self.dropout(self.act(cat_out)))
        return loc_out, cat_out

class DatasetPrePare(Dataset):
    def __init__(self, forward, label, user):
        self.forward = forward
        self.label = label
        self.user = user

    def __len__(self):
        assert len(self.forward) == len(self.label) == len(self.user)
        return len(self.forward)

    def __getitem__(self, index):
        return (self.forward[index], self.label[index], self.user[index])

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.best_epoch_val_loss = 0
        
    def step(self, score, loss, user_model, cat_model, loc_model, gcn_model, enc_model, epoch, result_dir):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = loss
            self.save_checkpoint(user_model, cat_model, loc_model, gcn_model, enc_model, result_dir)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = loss
            self.save_checkpoint(user_model, cat_model, loc_model, gcn_model, enc_model, result_dir)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, user_model, cat_model, loc_model, gcn_model, enc_model, result_dir):
        state_dict = {
            'user_emb_model_state_dict': user_model.state_dict(),
            'cat_emb_model_state_dict': cat_model.state_dict(),
            'loc_emb_model_state_dict': loc_model.state_dict(),                     
            'geogcn_model_state_dict': gcn_model.state_dict(),
            'transformer_encoder_model_state_dict': enc_model.state_dict()
            } 
        best_result = os.path.join(result_dir, 'checkpoint.pt')    
        torch.save(state_dict, best_result) 