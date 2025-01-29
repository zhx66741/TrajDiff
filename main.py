import os
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from TrajDiff.model.ddpm import DDPM
from TrajDiff.model.sam import FP_SAMEncoder,MLP
from TrajDiff.utils.DL import DL
from TrajDiff.utils.tools import hitting_ratio


def pre_train(args):
    print('pre_training')
    model = FP_SAMEncoder(args,args.init_dim,args.emb_dim,args.nhead,args.hidden_dim,args.nlayer,0.1).to(args.device)
    ddpm = DDPM(model, 1000, args.device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = 10000
    best_epoch = 0
    bad_counter = 0
    bad_patience = 5

    for i_ep in range(20):
        model.train()
        loss_ep = []
        epoch_time = time.time()
        train_bar = tqdm(dl.generate_ddpm_input())  # [bs,seq_len,emb_dim]
        for i_batch, batch in enumerate(train_bar):
            traj_p, traj_xy, traj_len = batch
            optimizer.zero_grad()
            loss_ = ddpm(traj_p, traj_xy)
            loss_ep.append(loss_.item())
            loss_.backward()
            optimizer.step()
            train_bar.set_description("Train Epoch: [{}/{}] Loss: {:.4f}".format(i_ep, 20, loss_.item()))

        scheduler.step()
        loss_ep_avg = np.mean(loss_ep)
        print("[Training] ep={}, avg_loss={:.7f}, @={:.7f}".format(i_ep, loss_ep_avg, time.time() - epoch_time))

        if best_loss > loss_ep_avg:
            best_epoch = i_ep
            best_loss = loss_ep_avg
            bad_counter = 0
            torch.save(model.state_dict(), args.checkpoint)
            print('model_saved')
        else:
            bad_counter += 1
        if bad_counter == bad_patience or (i_ep + 1) == 20:
            print("[Training] END! best_epoch={}, best_loss_train={:.7f}".format(best_epoch, best_loss))
            break
    return



def fine_tuning(args):
    model = FP_SAMEncoder(args,args.init_dim,args.emb_dim,args.nhead,args.hidden_dim,args.nlayer,0.1).to(args.device)
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    mlp = MLP(args.emb_dim,512).to(args.device)

    optimizer = torch.optim.Adam([{'params': model.parameters(),'lr': 1e-4},{'params': mlp.parameters(),'lr': 1e-4} ] )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_hr_train = 0
    best_epoch = 0
    bad_counter = 0
    bad_patience = 10

    for i_ep in range(30):
        model.train()
        loss_ep = []
        epoch_time = time.time()
        train_bar = tqdm(dl.trajsimi_dataset_generator_pairs_batchi())     # [bs,seq_len,emb_dim]
        for i_batch, batch in enumerate(train_bar) :
            traj_p,traj_xy,traj_len,sub_simi = batch
            sub_simi = sub_simi.to(args.device)
            optimizer.zero_grad()
            model_rtn = model(traj_p.to(torch.float32),traj_xy)
            model_rtn = mlp(model_rtn)

            pred_l1_simi = torch.cdist(model_rtn, model_rtn, 1)
            pred_l1_simi_ = pred_l1_simi
            pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1]
            truth_l1_simi = torch.tensor(sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal=1) == 1],dtype=torch.float32).to(args.device)
            train_loss = model.MSELoss(pred_l1_simi, truth_l1_simi) + args.gamma1 * model.listmle_loss(sub_simi,pred_l1_simi_) + args.gamma2 * model.ListNetLoss(sub_simi, pred_l1_simi_)

            loss_ep.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            train_bar.set_description("Train Epoch: [{}/{}] Loss: {:.4f}".format(i_ep, 30, train_loss.item()))

        scheduler.step()
        loss_ep_avg = np.mean(loss_ep)
        print("[Training] ep={}, avg_loss={:.7f}, @={:.7f}".format(i_ep, loss_ep_avg, time.time() - epoch_time))

        eval_result = Test_Eval(model,mlp,'eval')
        test_result = Test_Eval(model,mlp,'test')
        print('eval:hr1:{},hr5:{},hr10:{},hr20:{},hr50:{},h5r20:{}'.format(*eval_result))
        print('test:hr1:{},hr5:{},hr10:{},hr20:{},hr50:{},h5r20:{}'.format(*test_result))

        if eval_result[0] > best_hr_train :
            best_epoch = i_ep
            best_hr_train = eval_result[0]
            bad_counter = 0
            torch.save({"model": model.state_dict(),"mlp": mlp.state_dict()},args.checkpoint_simi)
            print('model_saved')
        else:
            bad_counter += 1

        if bad_counter == bad_patience or (i_ep + 1) == 30:
            print("[Training] END! best_epoch={}, best_loss_train={:.7f}".format(best_epoch, best_hr_train))
            break

    checkpoint = torch.load(args.checkpoint_simi)
    model.load_state_dict(checkpoint['model'])
    mlp.load_state_dict(checkpoint['mlp'])

    result_test = Test_Eval(model,mlp,'test')
    result_eval = Test_Eval(model,mlp,'eval')
    print('best_test:hr1:{},hr5:{},hr10:{},hr20:{},hr50:{},h5r20:{}'.format(result_test[0], result_test[1],
                                                                            result_test[2],
                                                                            result_test[3], result_test[4],
                                                                            result_test[5]))
    print('best_eval:hr1:{},hr5:{},hr10:{},hr20:{},hr50:{},h5r20:{}'.format(result_eval[0], result_eval[1],
                                                                            result_eval[2],
                                                                            result_eval[3], result_eval[4],
                                                                            result_eval[5]))
    return



@torch.no_grad()
def Test_Eval(model,mlp,mode):
    model.eval()
    mlp.eval()
    if mode == 'eval':
        datasets_simi, max_distance = dl.dic_datasets['evals_simi'], dl.dic_datasets['max_distance']
        datasets = dl.dic_datasets['evals_traj']

    elif mode == 'test':
        datasets_simi, max_distance = dl.dic_datasets['tests_simi'], dl.dic_datasets['max_distance']
        datasets = dl.dic_datasets['tests_traj']

    datasets_simi = torch.tensor(datasets_simi, device=args.device, dtype=torch.float)
    datasets_simi = (datasets_simi + datasets_simi.T) / max_distance
    traj_outs = []  # 用于保存模型输出

    for i_batch, batch in enumerate(dl.trajsimi_dataset_generator_single_batchi(datasets)):  # [bs,seq_len,emb_dim]
        traj_p,traj_xy,traj_len = batch
        model_rtn = model(traj_p.to(torch.float32),traj_xy)
        model_rtn = mlp(model_rtn)
        traj_outs.append(model_rtn.cpu())

    traj_outs = torch.cat(traj_outs)
    pred_l1_simi = torch.cdist(traj_outs, traj_outs, 1)
    pred_l1_simi = torch.tensor(pred_l1_simi)
    truth_l1_simi = datasets_simi.to(device=args.device)

    hr1 = hitting_ratio(pred_l1_simi, truth_l1_simi, 1, 1)
    hr5 = hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5)
    hr10 = hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 10)
    hr20 = hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 20)
    hr50 = hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 50)
    hrBinA = hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 5)

    return hr1, hr5, hr10, hr20, hr50, hrBinA

def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def parse_args():
    root_path = os.getcwd()
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--seed', type=int, default=42, help="")
    parser.add_argument('--gpu', type=int, default=3, help="")

    # dataset
    parser.add_argument('--city', type=str, default='tdriver', choices =['porto', 'geolife','tdriver'],help="")
    parser.add_argument('--measure', type=str, default='hausdorff', choices=['discret_frechet', 'hausdorff', 'sspd'], help="")
    parser.add_argument('--load_checkpoint', type=int, default=1, help="")

    # model parameters
    parser.add_argument('--bs', type=int, default=128, help="")
    parser.add_argument('--init_dim', type=int, default=2, help="")
    parser.add_argument('--nhead', type=int, default=8, help="")
    parser.add_argument('--nlayer', type=int, default=1, help="")
    parser.add_argument('--hidden_dim', type=int, default=2048, help="")
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--gamma1', type=float, default=0.001)
    parser.add_argument('--gamma2', type=float, default=1)

    args = parser.parse_args()

    if args.city == 'porto':
        args.cellspace_path = root_path + '/dataset/porto/porto_cellspace100.pkl'
        args.fine_tuning_data_path = root_path + '/dataset/porto/porto_1w.pkl'
        args.fine_tuning_data_simi = root_path + '/dataset/porto/traj_simi_dict_{}.pkl'.format(args.measure)

    elif args.city == 'geolife':
        args.cellspace_path = root_path + '/dataset/geolife/geolife_cellspace100.pkl'
        args.fine_tuning_data_path = root_path + '/dataset/geolife/geolife_1w.pkl'
        args.fine_tuning_data_simi = root_path + '/dataset/geolife/traj_simi_dict_{}.pkl'.format(args.measure)

    elif args.city == 'tdriver':
        args.cellspace_path = root_path + '/dataset/tdriver/tdriver_cellspace100.pkl'
        args.fine_tuning_data_path = root_path + '/dataset/tdriver/tdriver_1w.pkl'
        args.fine_tuning_data_simi = root_path + '/dataset/tdriver/traj_simi_dict_{}.pkl'.format(args.measure)

    args.pretraining_data_path = root_path + '/dataset/porto/porto_20w.pkl'
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    args.checkpoint = root_path + '/exp/snapshots/FP_SAM.pt'
    args.checkpoint_simi = root_path + '/exp/snapshots/FP_SAM_simi_{}.pt'.format(args.measure)

    return args


if __name__  == "__main__":
    args = parse_args()
    set_seed(args.seed)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    dl = DL(args)
    # pre_train(args)
    fine_tuning(args)
