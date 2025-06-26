import os
import sys
import math
import time
import torch
import random
import logging
import pandas as pd
import pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from TrajDiff_dif.utils.tools import merc2cell2,generate_spatial_features

# Load traj dataset for trajsimi learning
def read_trajsimi_traj_dataset(file_path):
    print('[Load trajsimi traj dataset] START.')
    _time = time.time()

    df_trajs = pd.read_pickle(file_path)
    assert df_trajs.shape[0] == 10000
    l = 10000

    train_idx = (int(l*0), int(l*0.7))
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))
    trains = df_trajs.iloc[train_idx[0] : train_idx[1]]
    evals = df_trajs.iloc[eval_idx[0] : eval_idx[1]]
    tests = df_trajs.iloc[test_idx[0] : test_idx[1]]

    print("trajsimi traj dataset sizes. traj: #total={} (trains/evals/tests={}/{}/{})".format(l, trains.shape[0], evals.shape[0], tests.shape[0]))
    return trains, evals, tests

# Load simi dataset for trajsimi learning
def read_trajsimi_simi_dataset(file_path):
    print('[Load trajsimi simi dataset] START.')
    _time = time.time()
    if not os.path.exists(file_path):
        print('trajsimi simi dataset does not exist')
        exit(200)

    with open(file_path, 'rb') as fh:
        trains_simi, evals_simi, tests_simi, max_distance = pickle.load(fh)
        print("[trajsimi simi dataset loaded] @={}, trains/evals/tests={}/{}/{}" \
                .format(time.time() - _time, len(trains_simi), len(evals_simi), len(tests_simi)))
        return trains_simi, evals_simi, tests_simi, max_distance

class TrajDataset(Dataset):
    def __init__(self, data):
        # data: DataFrame
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DL:
    def __init__(self,args):
        self.args = args
        self.cellspace = pickle.load(open(args.cellspace_path, 'rb'))
        self.dic_datasets = self.load_trajsimi_dataset()

    def generate_bdbm_input(self):
        ssl = self.dic_datasets['ssl']
        ssl_data = TrajDataset(ssl)
        dl = DataLoader(ssl_data, batch_size=2 * self.args.bs, shuffle=True, collate_fn=self.ssl_collate, num_workers=0,drop_last=True)

        return dl

    def ssl_collate(self, trajs):

        trajs_cell, trajs_p, trajs_xy = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
        trajs_p = [generate_spatial_features(t, self.cellspace) for t in trajs_p]
        trajs_mask = self.creat_padding_mask(trajs_p)
        trajs_p = pad_sequence(trajs_p, batch_first=True).to(self.args.device)
        trajs_xy = pad_sequence(trajs_xy, batch_first=True).to(self.args.device)

        x = (trajs_p[:self.args.bs], trajs_xy[:self.args.bs], trajs_mask[:self.args.bs])
        y = (trajs_p[self.args.bs:], trajs_xy[self.args.bs:], trajs_mask[self.args.bs:])
        return x, y


    def trajsimi_dataset_generator_pairs_batchi(self):
        datasets_simi, max_distance = self.dic_datasets['trains_simi'], self.dic_datasets['max_distance']
        datasets = self.dic_datasets['trains_traj']

        len_datasets = len(datasets)
        datasets_simi = torch.tensor(datasets_simi, device=self.args.device)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance

        count_i = 0
        counts = 3000
        while count_i < counts:
            dataset_idxs_sample = random.sample(range(len_datasets), k=self.args.bs)
            sub_simi = datasets_simi[dataset_idxs_sample][:, dataset_idxs_sample]

            # 1.获取轨迹 2.过滤轨迹 3.归一化wgs坐标 4.产生mask 5.填充
            trajs = [datasets[d_idx] for d_idx in dataset_idxs_sample]
            trajs_cell, trajs_p, trajs_xy = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            trajs_p = [generate_spatial_features(t, self.cellspace) for t in trajs_p]
            trajs_mask = self.creat_padding_mask(trajs_p)
            trajs_p = pad_sequence(trajs_p, batch_first=True).to(self.args.device)
            trajs_xy = pad_sequence(trajs_xy,batch_first=True).to(self.args.device)

            yield trajs_p, trajs_xy, trajs_mask.to(self.args.device), sub_simi.to(self.args.device)
            count_i += 1

    def trajsimi_dataset_generator_single_batchi(self, datasets):
        cur_index = 0
        len_datasets = len(datasets)

        while cur_index < len_datasets:
            end_index = cur_index + self.args.bs if cur_index + self.args.bs < len_datasets else len_datasets

            # 1.获取轨迹 2.过滤轨迹 3.归一化wgs坐标 4.产生mask 5.填充
            trajs = [datasets[d_idx] for d_idx in range(cur_index, end_index)]
            trajs_cell, trajs_p, trajs_xy = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            trajs_p = [generate_spatial_features(t, self.cellspace) for t in trajs_p]
            trajs_mask = self.creat_padding_mask(trajs_p)
            trajs_p = pad_sequence(trajs_p, batch_first=True).to(self.args.device)
            trajs_xy = pad_sequence(trajs_xy, batch_first=True).to(self.args.device)

            yield trajs_p, trajs_xy, trajs_mask.to(self.args.device)
            cur_index = end_index

    def load_trajsimi_dataset(self):

        ssl = pickle.load(open(self.args.pretraining_data_path, 'rb'))
        ssl = ssl.merc_seq.values

        trains_traj, evals_traj, tests_traj = read_trajsimi_traj_dataset(self.args.fine_tuning_data_path)
        trains_traj, evals_traj, tests_traj = trains_traj.merc_seq.values, evals_traj.merc_seq.values, tests_traj.merc_seq.values
        trains_simi, evals_simi, tests_simi, max_distance = read_trajsimi_simi_dataset(self.args.fine_tuning_data_simi)

        return {"ssl":ssl, 'trains_traj': trains_traj, 'evals_traj': evals_traj, 'tests_traj': tests_traj, \
                'trains_simi': trains_simi, 'evals_simi': evals_simi, 'tests_simi': tests_simi, \
                'max_distance': max_distance}

    @staticmethod
    def creat_padding_mask(trajs):
        """Create a mask for a batch of trajectories.
        - False indicates that the position is a padding part that exceeds the original trajectory length
        - while True indicates that the position is the valid part of the trajectory.
        """
        lengths = torch.tensor([len(traj) for traj in trajs])
        max_len = max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        return ~mask