from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from TKDE_3_copy_3.utils.tools import merc2cell2


def print_stats(trajs):
    lons = []
    lats = []
    for traj in trajs:
        for p in traj:
            lon, lat = p[0], p[1]
            lons.append(lon)
            lats.append(lat)
    lons = np.array(lons)
    lats = np.array(lats)
    mean_lon, mean_lat, std_lon, std_lat = np.mean(lons), np.mean(lats), np.std(lons), np.std(lats)
    x = {"mean_lon": mean_lon, "mean_lat": mean_lat, "std_lon": std_lon, "std_lat": std_lat}
    return x

class TransformerDataset(Dataset):
    def __init__(self, trajs):
        super(TransformerDataset, self).__init__()
        self.trajs = trajs
        self.trajs_wgs = trajs.wgs_seq.tolist()
        self.trajs_merc = trajs.merc_seq.tolist()
        self.IDs = [i for i in range(len(self.trajs))]
        print(f"[Data Preparation] Totally {len(self.trajs_wgs)} samples prepared in Transformer Dataset.")

    def __getitem__(self, idx):
        return torch.tensor(self.trajs_merc[idx]), idx

    def get_items(self, indices):
        merc = [self.__getitem__(i)[0] for i in indices]
        return merc

    def __len__(self):
        return len(self.IDs)


class TransformerDataLoader():
    def __init__(self, trajs, dis_matrix=None, batch_size=20, mode="train", sampling_num=20, alpha=16, cell_space=None):
        self.dataset = TransformerDataset(trajs)
        self.data_range = print_stats(self.dataset.trajs_merc)
        self.mean_lon = self.data_range["mean_lon"]
        self.mean_lat = self.data_range["mean_lat"]
        self.std_lon = self.data_range["std_lon"]
        self.std_lat = self.data_range["std_lat"]

        self.mode = mode
        self.dis_matrix = dis_matrix
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.alpha = alpha
        self.cell_space = cell_space

        print(f"[Data Preparation] Creating Transformer DataLoader for {mode} ...")
        print(f"[Data Preparation] Transformer DataLoader: {len(trajs)} samples")
        self.dataloader = self.create_dataloader()
        print("-" * 100)

    def get_dataloader(self):
        return self.dataloader

    def get_dis_matrix(self):
        return self.dis_matrix


    @staticmethod
    def creat_padding_mask(trajs):
        """Create a mask for a batch of trajectories.
        - False indicates that the position is a padding part that exceeds the original trajectory length
        - while True indicates that the position is the valid part of the trajectory.
        """
        lengths = torch.tensor([len(traj) for traj in trajs])
        max_len = max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        return ~mask, lengths

    def collate_fn_pretrain(self, data):
        trajs, _ = zip(*data)

        trajs_xy, trajs_merc = zip(*[merc2cell2(t, self.cell_space) for t in trajs])
        trajs_merc = self.trajs_normalize(trajs_merc)
        trajs_mask, _ = self.creat_padding_mask(trajs_merc)
        trajs_merc = pad_sequence(trajs_merc, batch_first=True)
        trajs_xy = pad_sequence(trajs_xy, batch_first=True)

        x = (trajs_merc[:self.batch_size], trajs_xy[:self.batch_size], trajs_mask[:self.batch_size])
        y = (trajs_merc[self.batch_size:], trajs_xy[self.batch_size:], trajs_mask[self.batch_size:])

        return x, y

    def collate_fn_train(self, data):
        _, indices = zip(*data)

        target_idxs = []
        for _ in indices:
            target_indices = np.random.choice(range(len(self.dis_matrix)), size=self.sampling_num, replace=False)
            target_idxs.extend(target_indices)

        # target
        target_trajs_merc = self.dataset.get_items(target_idxs)
        target_trajs_xy, target_trajs_merc = zip(*[merc2cell2(t, self.cell_space) for t in target_trajs_merc])
        target_trajs_merc = self.trajs_normalize(target_trajs_merc)
        target_trajs_mask, target_len = self.creat_padding_mask(target_trajs_merc)
        target_trajs_merc = pad_sequence(target_trajs_merc, batch_first=True, padding_value=0)
        target_trajs_xy = pad_sequence(target_trajs_xy, batch_first=True, padding_value=0)

        sub_simi = self.dis_matrix[target_idxs][:, target_idxs]
        sub_simi = torch.exp(- self.alpha * sub_simi)
        target = (target_trajs_merc, target_trajs_xy, target_trajs_mask, sub_simi)

        return target




    def collate_fn_test_eval(self, data):
        _, indices = zip(*data)
        trajs_merc = self.dataset.get_items(indices)
        trajs_xy, trajs_merc = zip(*[merc2cell2(t, self.cell_space) for t in trajs_merc])
        trajs_merc = self.trajs_normalize(trajs_merc)
        trajs_mask, trajs_len = self.creat_padding_mask(trajs_merc)
        trajs_merc = pad_sequence(trajs_merc, batch_first=True)
        trajs_xy = pad_sequence(trajs_xy, batch_first=True)

        return trajs_merc, trajs_xy, trajs_mask, torch.tensor(indices), trajs_len


    def trajs_normalize(self, trajs):
        mean = torch.tensor([self.mean_lon, self.mean_lat], dtype=torch.float32)
        std = torch.tensor([self.std_lon, self.std_lat], dtype=torch.float32)
        normalized_trajs = [(traj - mean) / std for traj in trajs]
        return normalized_trajs


    def create_dataloader(self) -> DataLoader:
        """
        given trajectory dataset and batch_size, return the corresponding DataLoader 
        """
        if self.mode == "pretrain":
            dataloader = DataLoader(dataset=self.dataset, batch_size=2 * self.batch_size, shuffle=True, num_workers=32,
                                    pin_memory=True, collate_fn=self.collate_fn_pretrain, drop_last=True)
            print(
                f"[Data Preparation] TransformerDataLoader: batch size {self.batch_size}, {len(dataloader.dataset)} samples")
            return dataloader

        elif self.mode == "train":
            dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=32,
                                    pin_memory=True, collate_fn=self.collate_fn_train)
            print(
                f"[Data Preparation] TransformerDataLoader: batch size {self.batch_size}, {len(dataloader.dataset)} samples")
            return dataloader

        elif self.mode == "test" or self.mode == "eval":
            dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=32,
                                    pin_memory=True, collate_fn=self.collate_fn_test_eval)
            print(
                f"[Data Preparation] TransformerDataLoader: batch size {self.batch_size}, {len(dataloader.dataset)} samples")
            return dataloader

















