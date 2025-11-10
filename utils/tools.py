import torch
import numpy as np



def merc2cell2(src, cs):
    tgt = [(cs.get_xyidx_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i - 1][0]]
    tgt_xy, tgt_p = zip(*tgt)
    return torch.tensor(tgt_xy,dtype=torch.float32), torch.stack(tgt_p, dim=0)

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


def mean_pooling(x, padding_masks):
    """
    input: batch_size, seq_len, hidden_dim
    output: batch_size, hidden_dim
    """
    x = x*padding_masks.unsqueeze(-1)
    x = torch.sum(x, dim=1)/torch.sum(padding_masks, dim=1).unsqueeze(-1)  #  mean pooling excluding the padding part.
    return x