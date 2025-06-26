import math
import torch
import numpy as np
from TrajDiff.utils.cellspace import CellSpace
# ref: TrjSR
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def hitting_ratio(preds: torch.Tensor, truths: torch.Tensor, pred_topk: int, truth_topk: int):
    # hitting ratio and recall metrics. see NeuTraj paper
    # the overlap percentage of the topk predicted results and the topk ground truth
    # overlap(overlap(preds@pred_topk, truths@truth_topk), truths@truth_topk) / truth_topk

    # preds = [batch_size, class_num], tensor, element indicates the probability
    # truths = [batch_size, class_num], tensor, element indicates the probability
    assert preds.shape == truths.shape and pred_topk < preds.shape[1] and truth_topk < preds.shape[1]

    _, preds_k_idx = torch.topk(preds, pred_topk + 1, dim=1, largest=False)
    _, truths_k_idx = torch.topk(truths, truth_topk + 1, dim=1, largest=False)

    preds_k_idx = preds_k_idx.cpu()
    truths_k_idx = truths_k_idx.cpu()

    tp = sum([np.intersect1d(preds_k_idx[i], truths_k_idx[i]).size for i in range(preds_k_idx.shape[0])])

    return (tp - preds.shape[0]) / (truth_topk * preds.shape[0])


def merc2cell2(src, cs: CellSpace):
    # convert and remove consecutive duplicates
    tgt, tgt_p, tgt_xy = [], [], []
    last_cell_id = None
    for p in src:
        cell_id = cs.get_cellid_by_point(*p)
        if cell_id != last_cell_id:
            tgt.append(cell_id)
            tgt_p.append(p)
            tgt_xy.append(cs.get_xyidx_by_point(*p))
            last_cell_id = cell_id
    return torch.tensor(tgt).float(), torch.tensor(tgt_p).float(), torch.tensor(tgt_xy).float()

def generate_spatial_features(src,cs: CellSpace):  # [x,y]
    x = (src[:, 0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[:, 1] - cs.y_min) / (cs.y_max - cs.y_min)
    return torch.stack((x, y), dim=1)


