import pickle as pk
import torch
import argparse
import warnings

warnings.filterwarnings("ignore")

from utils.cellspace import CellSpace
from config import DATASET


def load_pretain_dataset():
    # 读取预训练数据
    if args.dataset == "porto" or args.dataset == "geolife":
        pretain_data_path = DATASET["pretrain_data_1"]
    elif args.dataset == "tdrive":
        pretain_data_path = DATASET["pretrain_data_2"]
    pretain_data = pk.load(open(pretain_data_path, 'rb'))[:20000]
    return pretain_data


def create_cellspace():
    area_range, cell_size = DATASET[args.dataset]["area_range"], DATASET[args.dataset]["cell_size"]
    cell_space = CellSpace(cell_size, cell_size, area_range["min_lon"], area_range["min_lat"], area_range["max_lon"],
                           area_range["max_lat"])
    return cell_space


def load_fine_tuning_dataset():
    train_start, eval_start, test_start = 0, 2000, 3000
    traj_data_path = DATASET[args.dataset]["traj_data"]
    dis_matrix_path = DATASET[args.dataset]["dis_matrix"][args.target_measure]
    traj_data = pk.load(open(traj_data_path, 'rb'))
    dis_matrix = torch.tensor(pk.load(open(dis_matrix_path, 'rb')))

    dis_matrix = dis_matrix + dis_matrix.T
    train_dis_matrix = torch.tensor(dis_matrix[train_start:eval_start, train_start:eval_start]).float()
    eval_dis_matrix = torch.tensor(dis_matrix[eval_start:test_start, eval_start:test_start]).float()
    test_dis_matrix = torch.tensor(dis_matrix[test_start:, test_start:]).float()
    train_data = (traj_data[train_start:eval_start], train_dis_matrix)
    eval_data = (traj_data[eval_start:test_start], eval_dis_matrix)
    test_data = (traj_data[test_start:], test_dis_matrix)

    return train_data, eval_data, test_data


def data_prepare():
    cellspace = create_cellspace()
    if args.mode == "pretrain":
        dataset = load_pretain_dataset()
    elif args.mode == "finetune":
        dataset = load_fine_tuning_dataset()
    else:
        ValueError(f"Invalid mode: {args.mode}. Please choose either 'pretrain' or 'finetune'.")
    return dataset, cellspace


def go(args):
    dataset, cellspace = data_prepare()

    # 创建DL
    from trajdiff.DL import TransformerDataLoader
    if args.mode == "pretrain":
        pretrain_dataloader = TransformerDataLoader(dataset, batch_size=args.pretrain_bs, mode=args.mode,
                                                    alpha=args.alpha, sampling_num=args.sampling_num,
                                                    cell_space=cellspace)
    else:
        train_dataloader = TransformerDataLoader(dataset[0][0], dataset[0][1], batch_size=args.batch_size, mode="train",
                                                 alpha=args.alpha, sampling_num=args.sampling_num, cell_space=cellspace)
        eval_dataloader = TransformerDataLoader(dataset[1][0], dataset[1][1], batch_size=args.batch_size, mode="eval",
                                                alpha=args.alpha, sampling_num=args.sampling_num, cell_space=cellspace)
        test_dataloader = TransformerDataLoader(dataset[2][0], dataset[2][1], batch_size=args.batch_size, mode="test",
                                                alpha=args.alpha, sampling_num=args.sampling_num, cell_space=cellspace)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from trajdiff.loss import Loss
    loss = Loss(args.emb_sim_metric, args.gamma1, args.gamma2, args.gamma3)

    from trajdiff.karras_diffusion import DDBM
    ddbm = DDBM(device=device, args=args)
    from trajdiff.SAM import SAMEncoder
    model = SAMEncoder(2, args.hidden_dim, args.n_heads, args.num_layers, 4 * args.hidden_dim, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----------- print the model and device info -----------
    print("-" * 40 + "DEVICE INFO" + "-" * 40)
    print(f"[Device Info] Using {device} for training")
    print("-" * 40 + "MODEL PRRINTER" + "-" * 40)
    print(f"[Model Initialization] Using {device} for training")
    print(f"[Model Initialization] Model: {model}")
    print(f"[Model Initialization] Optimizer: Adam, Learning ratio {args.lr}")
    # Calculate the total number of traina ble parameters.
    print("-" * 100)
    # -------------------------------------------------------

    from trajdiff.trainer import Trainer
    trainer = Trainer(model, ddbm, optimizer, loss, device, args)

    if args.mode == "pretrain":
        trainer.pre_train(pretrain_dataloader)
    elif args.mode == "finetune":
        trainer.run(train_dataloader, eval_dataloader, test_dataloader)
    return None

import numpy as np
import random

def setup_seed(seed=42):
    """
    设置随机种子以确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 让 cudnn 的计算可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("-" * 100)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training params
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", type=str, default="finetune", choices=["pretrain", "finetune"])
    parser.add_argument("--pretrain_bs", type=int, default=128)
    parser.add_argument("--pretrain_epoch", type=int, default=5)
    parser.add_argument("--finetune_epoch", type=int, default=100)

    parser.add_argument("--dataset", type=str, default="porto", choices=["porto", "geolife", "tdrive"])
    parser.add_argument("--target_measure", type=str, choices=["haus", "DFD", "sspd"], default="sspd")
    parser.add_argument("--load_checkpoint", type=int, choices=[0, 1], default=0)

    # model params
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sampling_num", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--gamma1", type=float, default=0.01)
    parser.add_argument("--gamma2", type=float, default=1)
    parser.add_argument("--gamma3", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(args)
    if args.target_measure == "sspd":
        setattr(args, "pe", "linear")
        setattr(args, "emb_sim_metric", "euc")
    else:
        setattr(args, "pe", "rnn")
        setattr(args, "emb_sim_metric", "chebyshev")

    setup_seed(args.seed)
    go(args)
