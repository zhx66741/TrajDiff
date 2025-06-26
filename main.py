import os
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


from utils.DL import DL
from utils.tools import hitting_ratio
from model.AggTransformer import AggAttnEncoder
from TrajDiff_upload.model.karras_diffusion import DDBM


def pre_train(args):
    print('pre_training')
    model = AggAttnEncoder(args, args.emb_dim, args.nhead, args.nlayer, args.hidden_dim).to(args.device)
    BDBM = DDBM(device=args.device, args=args)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = 10000
    best_epoch = 0
    bad_counter = 0
    bad_patience = 5

    for i_ep in range(20):
        model.train()
        loss_ep = []
        epoch_time = time.time()
        train_bar = tqdm(dl.generate_bdbm_input())  # [bs,seq_len,emb_dim]
        for i_batch, batch in enumerate(train_bar):
            x, y = batch
            optimizer.zero_grad()
            loss_ = BDBM.training_bridge_losses(model, x, y)
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
    model = AggAttnEncoder(args, args.emb_dim, args.nhead, args.nlayer, args.hidden_dim).to(args.device)
    print(model)
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_hr_train = 0
    best_epoch = 0
    bad_counter = 0
    bad_patience = 10

    for i_ep in range(args.epoch):
        model.train()
        loss_ep = []
        epoch_time = time.time()
        train_bar = tqdm(dl.trajsimi_dataset_generator_pairs_batchi())     # [bs,seq_len,emb_dim]
        for i_batch, batch in enumerate(train_bar):
            traj_p, traj_xy, traj_mask, sub_simi = batch

            optimizer.zero_grad()
            model_rtn = model(traj_p, traj_xy, None, traj_mask)
            pred_l1_simi = torch.cdist(model_rtn, model_rtn, 1)
            pred_l1_simi_ = pred_l1_simi
            pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1]
            truth_l1_simi = torch.tensor(sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal=1) == 1],dtype=torch.float32).to(args.device)
            train_loss = 0.001 * model.MSELoss(pred_l1_simi, truth_l1_simi) + args.gamma2 * model.rd_listnet_loss(pred_l1_simi_, sub_simi) + args.gamma1 * model.ListNetLoss(sub_simi, pred_l1_simi_)

            loss_ep.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            train_bar.set_description("Train Epoch: [{}/{}] Loss: {:.4f}".format(i_ep, 30, train_loss.item()))

        scheduler.step()

        loss_ep_avg = np.mean(loss_ep)
        print("[Training] ep={}, avg_loss={:.7f}, @={:.7f}".format(i_ep, loss_ep_avg, time.time() - epoch_time))

        eval_result = Test_Eval(model, 'eval')
        print('eval:hr1:{},hr5:{},hr10:{},hr20:{},hr50:{},h5r20:{}'.format(*eval_result))
        if eval_result[0] > best_hr_train:
            best_epoch = i_ep
            best_hr_train = eval_result[0]
            bad_counter = 0
            # torch.save(model.state_dict(), args.checkpoint_simi)
            # print('model_saved')
            test_result = Test_Eval(model, 'test')
            print('TEST------>: hr1:{},hr5:{},hr10:{},hr20:{},hr50:{},h5r20:{}'.format(test_result[0], test_result[1],
                                                                                    test_result[2], test_result[3],
                                                                                    test_result[4], test_result[5]))

        else:
            bad_counter += 1

        if bad_counter == bad_patience or (i_ep + 1) == args.epoch:
            print("[Training] END! best_epoch={}, best_loss_train={:.7f}".format(best_epoch, best_hr_train))
            break


    return



@torch.no_grad()
def Test_Eval(model, mode):
    model.eval()
    if mode == 'eval':
        datasets_simi, max_distance = dl.dic_datasets['evals_simi'], dl.dic_datasets['max_distance']
        datasets = dl.dic_datasets['evals_traj']

    elif mode == 'test':
        datasets_simi, max_distance = dl.dic_datasets['tests_simi'], dl.dic_datasets['max_distance']
        datasets = dl.dic_datasets['tests_traj']

    datasets_simi = torch.tensor(datasets_simi, device=args.device, dtype=torch.float)
    datasets_simi = (datasets_simi + datasets_simi.T) / max_distance
    traj_outs = []  

    for i_batch, batch in enumerate(dl.trajsimi_dataset_generator_single_batchi(datasets)):  # [bs,seq_len,emb_dim]
        traj_p, traj_xy, traj_mask = batch
        model_rtn = model(traj_p, traj_xy, None, traj_mask)
        traj_outs.append(model_rtn.cpu())

    traj_outs = torch.cat(traj_outs)
    pred_l1_simi = torch.cdist(traj_outs, traj_outs, 1)
    pred_l1_simi = torch.tensor(pred_l1_simi)
    truth_l1_simi = datasets_simi.to(args.device)

    hr1 = hitting_ratio(pred_l1_simi, truth_l1_simi, 1,1)
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

    parser.add_argument('--seed', type=int, default=108)
    parser.add_argument('--gpu', type=int, default=4)

    # dataset
    parser.add_argument('--city', type=str, default='porto', choices = ['porto', 'geolife','tdriver'],help="")
    parser.add_argument('--measure', type=str, default='sspd', choices = ['hausdorff','sspd','discret_frechet'])
    parser.add_argument('--load_checkpoint', type=int, default=0)

    # model parameters
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--init_dim', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=16)
    parser.add_argument('--nlayer', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--gamma1', type=float, default=0.1)
    parser.add_argument('--gamma2', type=float, default=0.001)

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
    args.checkpoint = root_path + '/exp/snapshots/DBDB.pt'
    args.checkpoint_simi = root_path + '/exp/snapshots/DBDM_simi_{}.pt'.format(args.measure)

    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    dl = DL(args)
    pre_train(args)
    fine_tuning(args)
