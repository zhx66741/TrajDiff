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
    tgt = [ (cs.get_cellid_by_point(*p), p, cs.get_xyidx_by_point(*p)) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]
    tgt, tgt_p, tgt_xy = zip(*tgt)
    return tgt, tgt_p, tgt_xy

def generate_spatial_features(src,cs: CellSpace):  # [x,y,ri,li]
    tgt = []
    for i in range(len(src)):
        x = (src[i][0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (src[i][1] - cs.y_min) / (cs.y_max - cs.y_min)
        tgt.append([x, y])
    return tgt


###########################################################################
# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='hausdorff'):
    print("traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()

    data_1w = pickle.load(open(fine_tuning_data_path, 'rb'))
    data_1w.reset_index()
    l = data_1w.shape[0]

    trains,evals,tests = data_1w[:7000],data_1w[7000:8000],data_1w[8000:10000]
    trains, evals, tests = _normalization([trains, evals, tests])

    print("traj dataset sizes. traj: trains/evals/tests={}/{}/{}".format(trains.shape[0], evals.shape[0], tests.shape[0]))

    # 2.
    fn = _get_simi_fn(fn_name)
    tests_simi = _simi_matrix(fn, tests)
    evals_simi = _simi_matrix(fn, evals)
    trains_simi = _simi_matrix(fn, trains)  # [ [simi, simi, ... ], ... ]

    max_distance = max(max(map(max, trains_simi)), max(map(max, evals_simi)), max(map(max, tests_simi)))
    _output_file = '{}/traj_simi_dict_{}.pkl'.format(root_path, fn_name)
    with open(_output_file, 'wb') as fh:
        tup = trains_simi, evals_simi, tests_simi, max_distance
        pickle.dump(tup, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print("traj_simi_computation ends. @={:.3f}".format(time.time() - _time))
    return tup

def _normalization(lst_df):
    # lst_df: [df, df, df]
    xs = []
    ys = []
    for df in lst_df:
        for _, v in df.merc_seq.items():
            arr = np.array(v)
            xs.append(arr[:,0])
            ys.append(arr[:,1])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    mean = np.array([xs.mean(), ys.mean()])
    std = np.array([xs.std(), ys.std()])

    for i in range(len(lst_df)):
        lst_df[i].merc_seq = lst_df[i].merc_seq.apply(lambda lst: ( (np.array(lst)-mean)/std ).tolist())
    return lst_df


def _get_simi_fn(fn_name):
    fn =  {'lcss': tdist.lcss, 'edr': tdist.edr, 'frechet': tdist.frechet,
            'discret_frechet': tdist.discret_frechet,'sspd':tdist.sspd,'hausdorff': tdist.hausdorff, 'edwp': edwp}.get(fn_name, None)
    if fn_name == 'lcss' or fn_name == 'edr':
        fn = partial(fn, eps = 0.25)
    return fn


def _simi_matrix(fn, df):
    _time = time.time()

    l = df.shape[0]
    batch_size = 50
    assert l % batch_size == 0

    # parallel init
    tasks = []
    for i in range(math.ceil(l / batch_size)):
        if i < math.ceil(l / batch_size) - 1:
            tasks.append((fn, df, list(range(batch_size * i, batch_size * (i + 1)))))
        else:
            tasks.append((fn, df, list(range(batch_size * i, l))))

    num_cores = int(mp.cpu_count()) // 2
    assert num_cores > 0
    print("pool.size={}".format(num_cores))
    pool = mp.Pool(num_cores)
    lst_simi = pool.starmap(_simi_comp_operator, tasks)
    pool.close()

    # extend lst_simi to matrix simi and pad 0s
    lst_simi = sum(lst_simi, [])
    for i, row_simi in enumerate(lst_simi):
        lst_simi[i] = [0] * (i + 1) + row_simi
    assert sum(map(len, lst_simi)) == l ** 2
    print('simi_matrix computation done., @={}, #={}'.format(time.time() - _time, len(lst_simi)))

    return lst_simi


# async operator
def _simi_comp_operator(fn, df_trajs, sub_idx):
    simi = []
    l = df_trajs.shape[0]
    for _i in sub_idx:
        t_i = np.array(df_trajs.iloc[_i].merc_seq)
        simi_row = []
        for _j in range(_i + 1, l):
            t_j = np.array(df_trajs.iloc[_j].merc_seq)
            simi_row.append( float(fn(t_i, t_j)) )
        simi.append(simi_row)
    print('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
                    .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi
