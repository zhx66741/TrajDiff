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


def lonlat2meters(lon, lat):
    R = 6378137.0  # WGS84 椭球体半径
    # 限制纬度范围，避免无穷大
    lat = max(min(lat, 85.05112878), -85.05112878)

    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def inrange(lon, lat):
    if lon <= min_lon or lon >= max_lon or lat <= min_lat or lat >= max_lat:
        return False
    return True


def clean_and_output_data():
    _time = time.time()
    dfraw = pd.read_csv(raw_data_path)
    print("==========> Data loading.....")

    dfraw = dfraw.rename(columns={"POLYLINE": "wgs_seq"})
    dfraw = dfraw[dfraw.MISSING_DATA == False]

    # length requirement
    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['traj_len'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.traj_len >= min_traj_len) & (dfraw.traj_len <= max_traj_len)]
    print('==========> Preprocessed length require. #traj={}'.format(dfraw.shape[0]))

    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))
    dfraw = dfraw[dfraw.inrange == True]
    print('==========> Preprocessed range require. #traj={}'.format(dfraw.shape[0]))

    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

    print('==========> Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    dfraw = dfraw[['traj_len', 'wgs_seq', 'merc_seq']].reset_index(drop=True)  # 1372725

    dfraw.to_pickle(clean_data_path)
    print('==========> Preprocess end. @={:.0f}'.format(time.time() - _time))
    return


###########################################################################

def split_data():
    clean_data = pickle.load(open(clean_data_path, 'rb'))

    idx = int(clean_data.shape[0] * 0.7)
    fine_tuning_data = clean_data.iloc[idx:idx + 10000]
    fine_tuning_data.to_pickle(porto_1w)
    print("==========> data have been split")
    return

###########################################################################
# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='haus'):
    print("==========> traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()

    data_1w = pickle.load(open(porto_1w, 'rb'))
    data_1w.reset_index()
    l = data_1w.shape[0]

    # 2.
    fn = _get_simi_fn(fn_name)
    data_simi = _simi_matrix(fn, data_1w)  # [ [simi, simi, ... ], ... ]

    _output_file = '{}/traj_simi_dict_{}_{}.pkl'.format(root_path, fn_name, str(min_traj_len)+str(max_traj_len))
    with open(_output_file, 'wb') as fh:
        tup = data_simi
        pickle.dump(tup, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print("==========> traj_simi_computation ends. @={:.3f}".format(time.time() - _time))
    return




def _get_simi_fn(fn_name):
    fn = {'DFD': tdist.discret_frechet, 'sspd': tdist.sspd, 'haus': tdist.hausdorff}.get(fn_name, None)
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
        t_i = np.array(df_trajs.iloc[_i].wgs_seq)
        simi_row = []
        for _j in range(_i + 1, l):
            t_j = np.array(df_trajs.iloc[_j].wgs_seq)
            simi_row.append(float(fn(t_i, t_j)))
        simi.append(simi_row)
    print('simi_comp_operator ends. sub_idx=[{}:{}], pid={}'.format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi


if __name__ == "__main__":
    from TKDE.config import DATASET
    min_lon, min_lat, max_lon, max_lat = DATASET["porto"]["range"].values()
    min_traj_len, max_traj_len = DATASET['porto']["length"]

    root_path = os.getcwd()
    raw_data_path = "/data/data_666/zhx111/dataset/porto/train(1).csv"
    clean_data_path = root_path + '/clean_porto_{}.pkl'.format(str(min_traj_len)+str(max_traj_len))
    porto_1w = root_path + "/porto_1w_{}.pkl".format(str(min_traj_len)+str(max_traj_len))

    # 1.
    # clean_and_output_data()
    # 2.
    # split_data()
    # 3.
    traj_simi_computation('DFD')  # ['haus','sspd','DFD']
