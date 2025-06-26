import os
import math
import time
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from ast import literal_eval
import traj_dist.distance as tdist
from TrajDiff_dif.utils.edwp import edwp
from TrajDiff_dif.utils.cellspace import CellSpace
from TrajDiff_dif.utils.tools import lonlat2meters


def inrange(lon, lat):
    if lon <= min_lon or lon >= max_lon or lat <= min_lat or lat >= max_lat:
        return False
    return True


def clean_and_output_data():
    _time = time.time()
    dfraw = pd.read_csv(raw_data_path)
    dfraw = dfraw.rename(columns={"POLYLINE": "wgs_seq"})

    dfraw = dfraw[dfraw.MISSING_DATA == False]

    # length requirement
    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= min_traj_len) & (dfraw.trajlen <= max_traj_len)]
    print('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))
    dfraw = dfraw[dfraw.inrange == True]
    print('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))

    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

    print('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq']].reset_index(drop=True)  # 1372725

    dfraw.to_pickle(clean_data_path)
    print('Preprocess end. @={:.0f}'.format(time.time() - _time))
    return


def init_cellspace():
    # 1. create cellspase
    # 2. initialize cell embeddings (create graph, train, and dump to file)

    x_min, y_min = lonlat2meters(min_lon, min_lat)
    x_max, y_max = lonlat2meters(max_lon, max_lat)
    x_min -= cellspace_buffer
    y_min -= cellspace_buffer
    x_max += cellspace_buffer
    y_max += cellspace_buffer

    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    with open(cellspace_path, 'wb') as fh:
        pickle.dump(cs, fh)
    return


def filtering_data():
    clean_data = pickle.load(open(clean_data_path, 'rb'))
    pretraining_data = clean_data.iloc[:200000]
    pretraining_data.to_pickle(pretraining_data_path)

    idx = int(clean_data.shape[0] * 0.7)
    fine_tuning_data = clean_data.iloc[idx:idx + 10000]
    fine_tuning_data.to_pickle(fine_tuning_data_path)


###########################################################################
# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='hausdorff'):
    print("traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()

    data_1w = pickle.load(open(fine_tuning_data_path, 'rb'))
    data_1w.reset_index()
    l = data_1w.shape[0]

    trains, evals, tests = data_1w[:7000], data_1w[7000:8000], data_1w[8000:10000]
    trains, evals, tests = _normalization([trains, evals, tests])

    print(
        "traj dataset sizes. traj: trains/evals/tests={}/{}/{}".format(trains.shape[0], evals.shape[0], tests.shape[0]))

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
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    mean = np.array([xs.mean(), ys.mean()])
    std = np.array([xs.std(), ys.std()])

    for i in range(len(lst_df)):
        lst_df[i].merc_seq = lst_df[i].merc_seq.apply(lambda lst: ((np.array(lst) - mean) / std).tolist())
    return lst_df


def _get_simi_fn(fn_name):
    fn = {'discret_frechet': tdist.discret_frechet, 'sspd': tdist.sspd, 'hausdorff': tdist.hausdorff}.get(fn_name, None)
    if fn_name == 'lcss' or fn_name == 'edr':
        fn = partial(fn, eps=0.25)
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
            simi_row.append(float(fn(t_i, t_j)))
        simi.append(simi_row)
    print('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
          .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi


if __name__ == "__main__":
    min_lon = -8.7005
    min_lat = 41.1001
    max_lon = -8.5192
    max_lat = 41.2086
    cellspace_buffer = 500
    cell_size = 100

    min_traj_len = 20
    max_traj_len = 200

    root_path = os.getcwd()
    raw_data_path = root_path + '/train(1).csv'
    clean_data_path = root_path + '/clean_porto.pkl'
    cellspace_path = root_path + '/porto_cellspace100.pkl'

    pretraining_data_path = root_path + "/porto_20w.pkl"
    fine_tuning_data_path = root_path + "/porto_1w.pkl"

    # init_cellspace()
    # clean_and_output_data()
    """
    Preprocessed-rm length. #traj=1499510
    Preprocessed-rm range. #traj=1372725
    Preprocessed-output. #traj=1372725
    Preprocess end. @=729
    """
    # filtering_data()
    traj_simi_computation('hausdorff')  # ['hausdorff','sspd','discret_frechet']
    #  @=654.243