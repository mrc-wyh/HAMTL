import datetime, os
import argparse, logging
import numpy as np
from numpy import random
import torch
import torch.nn as nn
import dgl
from HFMTL.data_prepare import Data
from model import SeqPred, SlotEncoding, GeoGCN, DatasetPrePare, EarlyStopping
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from time import time
from pathlib import Path
import re
import glob

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)

def increment_path(path, exist_ok=False, sep='', mkdir=True):
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir_ = path if path.suffix == '' else path.parent  # directory
    if not dir_.exists() and mkdir:
        dir_.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def batch_seq_emb(args, data_b, max, cat_emb, loc_emb, pos_emb, user_emb_model, device):
    traj_n = len(data_b) 
    batch_input_emb = []
    batch_loc_labels = []
    batch_cat_labels = []
    labels = []
    lengh = []
    prior_loc = []
    src_lengths = []
    for i in range(traj_n):        
        if len(data_b[i][0]) <= max:
            traj_forward = np.array(data_b[i][0])
            traj_labels = np.array(data_b[i][1])    
        else:
            traj_forward = np.array(data_b[i][0][-max:])
            traj_labels = np.array(data_b[i][1][-max:])
        loc  = traj_forward[:, 0]
        loc_labels = traj_labels[:, 0]
        timeslot = traj_forward[:, 1]
        day_of_week = traj_forward[:, 2]
        cat_id = traj_forward[:, 3]
        cat = traj_forward[:, 3]
        cat_labels = traj_labels[:, 3]
        loc_s_emb = loc_emb[loc]
        if args.cat_learnable:
            cat_s_emb = cat_emb[cat]
        else:
            cat_s_emb = pos_emb(torch.tensor(cat_id).to(device))
        timeslot_emb = pos_emb(torch.tensor(timeslot).to(device))
        dw_emb = pos_emb(torch.tensor(day_of_week).to(device))
        user_emb = user_emb_model(torch.tensor(data_b[i][2]).to(device)).unsqueeze(0)

        user_emb = user_emb.expand(len(loc_s_emb), -1)
        time_emb = timeslot_emb + dw_emb
        input_emb = torch.cat((user_emb, loc_s_emb, cat_s_emb, time_emb), 1)
        batch_input_emb.append(input_emb)
        batch_loc_labels.append(torch.tensor(loc_labels).to(device))
        batch_cat_labels.append(torch.tensor(cat_labels).to(device))
        src_lengths.append(len(data_b[i][0]))    
        labels.append(loc_labels[-1])   
        prior_loc.append(loc[-1]) 
        lengh.append(len(loc))              
    batch_pad_emb = pad_sequence(batch_input_emb, batch_first=False, padding_value=0)
    batch_loc_labels_pad = pad_sequence(batch_loc_labels, batch_first=False, padding_value=-1)
    batch_cat_labels_pad = pad_sequence(batch_cat_labels, batch_first=False, padding_value=-1)
    
    #增加_key_padding_mask
    key_pad_mask = torch.where(batch_loc_labels_pad==-1, True, False)

    return batch_pad_emb, batch_loc_labels_pad, batch_cat_labels_pad, labels, key_pad_mask, prior_loc, lengh


def cal_acc_mrr(idxx, label, indices):
    acc = np.zeros((4, 1)) #top20 10 5 1
    mrr = 0
    for j, p in enumerate(idxx):
        t = label[j]
        if t in p:
            acc[0] += 1 #@20
            pos = np.argwhere(p == t)
            if pos >= 10:
                continue
            elif pos >= 5 and pos < 10:
                acc[1] += 1 #@10
            elif pos >= 1 and pos < 5:
                acc[1] += 1 #@10
                acc[2] += 1 #@5
            else:
                acc[1] += 1 #@10
                acc[2] += 1 #@5
                acc[3] += 1 #@1

    for i, loc_id in enumerate(label):
        id = np.argwhere(indices[i] == loc_id)
        mrr += 1 / (id + 1)
    acc_20 = acc[0] / len(label)
    acc_10 = acc[1] / len(label)
    acc_5 = acc[2] / len(label)
    acc_1 = acc[3] / len(label)
    mrr = mrr / len(label)
    return acc_1, acc_5, acc_10, acc_20, mrr

def evaluate(args, valid_loader, data, max_len, time_emb_model, user_emb_model, cat_emb_model, 
             loc_emb_model, geogcn_model, transformer_encoder_model, loss_fn, device):
    user_emb_model.eval()
    cat_emb_model.eval()
    loc_emb_model.eval()           
    geogcn_model.eval()
    transformer_encoder_model.eval()
    loc_emb = loc_emb_model(torch.tensor(range(data.loc_num)).to(device))
    if args.cat_learnable == 1: 
        cat_emb = cat_emb_model(torch.tensor(range(data.cat_num)).to(device))
        loc_emb = geogcn_model(loc_emb)
    elif args.cat_learnable == 0:
        cat_emb = torch.zeros([data.cat_num, args.cat_dim],dtype=torch.float).to(device)
        loc_emb = geogcn_model(loc_emb)
    else:
        loc_emb, cat_emb = geogcn_model(loc_emb)
    loss_loc_list = []
    loss_cat_list = []
    loss_total_list = []
    acc_list_1 = [] #Top 1
    acc_list_5 = [] #Top 5
    acc_list_10 = []
    acc_list_20 = []
    mrr_list = []
    for b, data_b in enumerate(valid_loader): 
        batch_emb, loc_labels_emb, cat_labels_emb, label_loc, key_pad_mask, prior_loc, lengh = batch_seq_emb(args, data_b, max_len, cat_emb, loc_emb, time_emb_model, 
                                                                             user_emb_model, device)

        seq_out_loc, seq_out_cat = transformer_encoder_model(batch_emb, key_pad_mask)

        loss_loc = loss_fn(seq_out_loc.transpose(1, 2), loc_labels_emb)
        loss_cat = loss_fn(seq_out_cat.transpose(1, 2), cat_labels_emb)
        loss =  loss_loc + loss_cat
        pred_loc = []
        for i in range(len(lengh)):
            tmp = seq_out_loc[lengh[i]-1, i, :]
            pred_loc.append(tmp)
        pred_loc = torch.stack(pred_loc)               
        _, idxx = pred_loc.data.topk(20, dim=-1)
        idxx = idxx.detach().cpu().numpy()
        indices = torch.argsort(pred_loc, descending=True)
        indices = indices.detach().cpu().numpy()
        acc1, acc5, acc10, acc20, mrr = cal_acc_mrr(idxx, label_loc, indices)
        acc_list_1.append(acc1)
        acc_list_5.append(acc5)
        acc_list_10.append(acc10)
        acc_list_20.append(acc20)
        mrr_list.append(mrr)
        loss_loc_list.append(loss_loc.detach().cpu().numpy())
        loss_cat_list.append(loss_cat.detach().cpu().numpy())
        loss_total_list.append(loss.detach().cpu().numpy())
    epoch_acc1 = np.mean(acc_list_1)
    epoch_acc5 = np.mean(acc_list_5)
    epoch_acc10 = np.mean(acc_list_10)
    epoch_acc20 = np.mean(acc_list_20)
    epoch_mrr = np.mean(mrr_list)
    epoch_loss_loc = np.mean(loss_loc_list)
    epoch_loss_cat = np.mean(loss_cat_list)
    epoch_loss_total = np.mean(loss_total_list)
    return epoch_acc1, epoch_acc5, epoch_acc10, epoch_acc20, epoch_mrr, epoch_loss_loc, epoch_loss_cat, epoch_loss_total

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
    
def main(args):
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = increment_path('run-TKY-SP/' + args.run_directory + '-' + 'lr_pa', sep='-')
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    logging.Formatter.converter = beijing
    log_name=(datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
    logname = args.log_name + log_name + '.log'
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=20,
        filename=os.path.join(result_dir, logname),
        filemode='a'
        ) 
    best_result = os.path.join(result_dir, 'best.txt')  
    f = open(best_result, 'a')
    f.write('-----------dataset:' + args.dataset +'----------------\n')
    f.write('enc_layer:'+' '+str(args.enc_layer_num)+' '+'GeoGCN_layer:'+' '+str(args.GeoGCN_layer_num)+' '+'lr:'+' '+str(args.lr)+' '+'weight_decay:'+' '+str(args.weight_decay)+' '+'enc_ffn_hdim:'+str(args.enc_ffn_hdim)+'\n')
    f.write('enc_drop:'+' '+str(args.enc_drop)+' '+'enc_nhead:'+' '+str(args.enc_nhead)+'\n')
    f.write('early_patience:'+' '+str(args.patience)+' '+'lr_patience:'+' '+str(args.lr_patience)+' '+'cat_dim:'+' '+str(args.cat_dim)+' '+'time_dim:'+' '+str(args.time_dim)+'\n')
    f.write('cat_to_loc_piror:'+' '+str(args.cat_to_loc_piror)+' '+'cat_learnable:'+' '+str(args.cat_learnable)+'\n')
        
    logging.log(23, args)
    logging.log(23,f"---------------------dataset: {args.dataset}---------------------------------------")
    logging.log(23,f"enc_layer: {args.enc_layer_num} GeoGCN_layer: {args.GeoGCN_layer_num} lr: {args.lr} weight_decay: {args.weight_decay} enc_ffn_hdim: {args.enc_ffn_hdim}")
    logging.log(23,f"enc_drop: {args.enc_drop} enc_nhead:{args.enc_nhead}")
    logging.log(23,f"early_patience:{args.patience} lr_patience:{args.lr_patience} cat_dim: {args.cat_dim} time_dim: {args.time_dim}")
    if args.early_stop:
        stopper = EarlyStopping(args.patience)
        
    data = Data(args)
    trans = data.hier_trans
    trans = trans.to(device)
    time_emb_model = SlotEncoding(args.time_dim, device=device)
    user_emb_model = nn.Embedding(data.user_num, args.user_dim).to(device) 
    loc_emb_model = nn.Embedding(data.loc_num, args.loc_dim).to(device)        
    cat_emb_model = nn.Embedding(data.cat_num, args.cat_dim).to(device) 
    if args.cat_learnable == 2:
        geogcn_model = GeoGCN(data.loc_g, data.hier_tree, data.tran_edge_weight, args, device).to(device)
    else:
        geogcn_model = GeoGCN(data.loc_g, data.loc_g, data.tran_edge_weight, args, device).to(device)
    transformer_encoder_model = SeqPred(data.cat_num, data.loc_num, trans, args, device=device).to(device)
    if args.cat_learnable == 1:
        optimizer = torch.optim.Adam(params=list(user_emb_model.parameters()) +
                                    list(cat_emb_model.parameters()) +
                                    list(loc_emb_model.parameters()) +
                                    list(geogcn_model.parameters()) +
                                    list(transformer_encoder_model.parameters()),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params=list(user_emb_model.parameters()) +
                                    list(loc_emb_model.parameters()) +
                                    list(geogcn_model.parameters()) +
                                    list(transformer_encoder_model.parameters()),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', verbose=True, factor=args.lr_factor, patience=args.lr_patience, min_lr=1e-7)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    train_dataset = DatasetPrePare(data.train_forward, data.train_labels, data.train_user)
    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                            shuffle=True, pin_memory=True, num_workers=0, collate_fn=lambda x:x)
    valid_dataset = DatasetPrePare(data.valid_forward, data.valid_labels, data.valid_user)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch,
                            shuffle=False, pin_memory=True, num_workers=0, collate_fn=lambda x:x)

    start_time = time()
    last_time = start_time
    for epoch in range(args.epochs):
        user_emb_model.train()
        if args.cat_learnable == 1: 
            cat_emb_model.train()
        loc_emb_model.train()
        geogcn_model.train()
        transformer_encoder_model.train()
        for b, data_b in enumerate(train_loader):
            torch.cuda.empty_cache()
            loc_emb = loc_emb_model(torch.tensor(range(data.loc_num)).to(device))
            if args.cat_learnable == 1: 
                cat_emb = cat_emb_model(torch.tensor(range(data.cat_num)).to(device))
                loc_emb = geogcn_model(loc_emb)
            elif args.cat_learnable == 0:
                cat_emb = 0
                cat_emb = torch.tensor(cat_emb).to(device)
                # cat_emb = torch.zeros([data.cat_num, args.cat_dim],dtype=torch.float).to(device)
                loc_emb = geogcn_model(loc_emb)
            else:
                loc_emb, cat_emb = geogcn_model(loc_emb)
            batch_emb, loc_labels_emb, cat_labels_emb, _, key_pad_mask,  _, _ = batch_seq_emb(args, data_b, args.traj_max_len, cat_emb, loc_emb, time_emb_model, 
                                                                                user_emb_model, device)
            seq_out_loc, seq_out_cat = transformer_encoder_model(batch_emb, key_pad_mask)
            loss_cat = loss_fn(seq_out_cat.transpose(1, 2), cat_labels_emb)
            loss_loc = loss_fn(seq_out_loc.transpose(1, 2), loc_labels_emb)

            loss = loss_loc + loss_cat
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        val_perf = evaluate(args, valid_loader, data, args.traj_max_len, time_emb_model, user_emb_model, cat_emb_model, 
                            loc_emb_model, geogcn_model, transformer_encoder_model, loss_fn, device)
        monitor_loss = val_perf[-1]
        monitor_score = val_perf[0] + val_perf[3]
        score = np.sum(val_perf[0:4])
        lr_scheduler.step(monitor_score)

        if epoch % args.print_interval == 0:
            duration = time() - last_time
            last_time = time()
            logging.log(23,f"Epoch {epoch} valid:loss_total = {val_perf[-1]:.6f} valid:loss_cat = {val_perf[-2]:.6f} valid:loss_loc = {val_perf[-3]:.6f} Score: {np.sum(val_perf[0:4]):.4f} Acc@20:{val_perf[3]:.4f} Acc@10:{val_perf[2]:.4f} Acc@5:{val_perf[1]:.4f} Acc@1:{val_perf[0]:.4f} MRR:{val_perf[4]:.4f} {duration:.3f} sec")    
        
        if args.early_stop and epoch != 0:
            if stopper.step(score, monitor_loss, user_emb_model, cat_emb_model, loc_emb_model, geogcn_model, transformer_encoder_model, epoch, result_dir):
                break 

    runtime = time() - start_time
    torch.cuda.empty_cache()
    if args.early_stop:
        logging.log(21, f"best epoch: {stopper.best_epoch}, best val acc:{stopper.best_score * 100:.4f}, val_loss:{stopper.best_epoch_val_loss:.6f}, ({runtime:.3f} sec)")
    if args.early_stop:
        state_dict = torch.load(os.path.join(result_dir, 'checkpoint.pt'))
        user_emb_model.load_state_dict(state_dict['user_emb_model_state_dict'])
        cat_emb_model.load_state_dict(state_dict['cat_emb_model_state_dict'])   
        loc_emb_model.load_state_dict(state_dict['loc_emb_model_state_dict'])                        
        geogcn_model.load_state_dict(state_dict['geogcn_model_state_dict'])   
        transformer_encoder_model.load_state_dict(state_dict['transformer_encoder_model_state_dict'])    
    test_dataset = DatasetPrePare(data.test_forward, data.test_labels, data.test_user)
    test_loader = DataLoader(test_dataset, batch_size=args.batch,
                            shuffle=False, pin_memory=True, num_workers=0, collate_fn=lambda x:x)
    test_perf = evaluate(args, test_loader, data, args.traj_max_len, time_emb_model, user_emb_model, cat_emb_model, 
                        loc_emb_model, geogcn_model, transformer_encoder_model, loss_fn, device)
    logging.log(23,f"test: loss_total = {val_perf[-1]:.6f} loss_cat = {val_perf[-2]:.6f} loss_loc = {val_perf[-3]:.6f} Acc@20:{test_perf[3]:.4f} Acc@10:{test_perf[2]:.4f} Acc@5:{test_perf[1]:.4f} Acc@1:{test_perf[0]:.4f} MRR:{test_perf[4]:.4f} {runtime:.3f} sec")    
    f.write('valid: Best epoch: '+str(stopper.best_epoch) +' score: ' +str(float('%0.4f'%(stopper.best_score*100)))+' loss: '+str(stopper.best_epoch_val_loss)+'\n')
    f.write('test Acc@20: '+str(float('%0.4f'%test_perf[3]))+' Acc@10: '+str(float('%0.4f'%test_perf[2]))+' Acc@5: '+str(float('%0.4f'%test_perf[1]))+' Acc@1: '+str(float('%0.4f'%test_perf[0]))+' MRR: '+str(float('%0.4f'%test_perf[4]))+'\n')
    f.write(str(float('%0.2f'%runtime))+'sec'+'\n')
    f.close()       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical')
    parser.add_argument('--dataset', type=str, default='./data/FS_TKY_SP', help='FS_NYC_SP, FS_TKY_SP')
    parser.add_argument("--batch", type=int, default=1200, help="TransformerEncoder batch")
    parser.add_argument("--traj_max_len", type=int, default=40, help="traj max length")
    parser.add_argument("--dist", type=int, default=1000, help="500, 1000, 1500, 2000")
    parser.add_argument("--enc_nhead", type=int, default=2, help="TransformerEncoderLayer Attention head number")
    parser.add_argument("--enc_ffn_hdim", type=int, default=1024, help="TransformerEncoderLayer FFN hidden dim")
    parser.add_argument("--enc_drop", type=float, default=0.2, help="Encoder dropout probability")
    parser.add_argument('--user_dim', type=int, default=128, help='user embedding dim')          
    parser.add_argument('--loc_dim', type=int, default=128, help='location embedding dim')          
    parser.add_argument('--cat_dim', type=int, default=16, help='category embedding dim')          
    parser.add_argument('--time_dim', type=int, default=16, help='time embedding dim')          
    parser.add_argument('--enc_layer_num', type=int, default=2, help='Number of TransformerEncoder layers.')     
    parser.add_argument('--GeoGCN_layer_num', type=int, default=2, help='Number of GCN layers.')     
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='learning rate scale factor.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=20, help='Patience in early stopping')
    parser.add_argument('--lr_patience', type=int, default=10, help='Patience for learning rate adjustment')
    parser.add_argument('--loss_w', type=float, default=1, help='loss weight')
    parser.add_argument('--seed', type=int, default=42, help="seed for our system")
    parser.add_argument('--print_interval', type=int, default=1, help="the interval of printing in training")
    parser.add_argument('--week', type=int, default=1, help="week mode")
    parser.add_argument('--day', type=int, default=1, help="day mode")
    parser.add_argument('--log_name', type=str, default='log', help="Name for logging")
    parser.add_argument('--run_directory', type=str, default='debug', help="run directory")
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--cat_learnable", type=int, default=2, help="0:Absolute Position Encoding Method; 1:Learnable Method; 2:Generative Method")
    parser.add_argument('--cat_to_loc_piror', type=int, default=1, help="0: no; 1: extracted from training data; 2:equal probability")
    parser.add_argument('--decoder_mode', type=int, default=1, help="0: hierarchical; 1: parallel")
    args = parser.parse_args()
       
    main(args)