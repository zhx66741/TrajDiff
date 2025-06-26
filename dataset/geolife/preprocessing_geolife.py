import os
import math
import time
import skmob
import pickle
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from ast import literal_eval
import traj_dist.distance as tdist
from TrajDiff.utils.edwp import edwp



from TrajDiff_dif.utils.tools import lonlat2meters, merc2cell2
from TrajDiff_dif.utils.cellspace import CellSpace


def inrange(lon, lat):
    if lon <= min_lon or lon >= max_lon or lat <= min_lat or lat >= max_lat:
        return False
    return True


def init_cellspace():
    # 1. create cellspase
    # 2. initialize cell embeddings (create graph, train, and dump to file)
    x_min, y_min = lonlat2meters(min_lon, min_lat)
    x_max, y_max = lonlat2meters(max_lon, max_lat)
    x_min -= 500
    y_min -= 500
    x_max += 500
    y_max += 500

    cell_size = int(100)
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    with open(cellspace_path, 'wb') as fh:
        pickle.dump(cs, fh, protocol = pickle.HIGHEST_PROTOCOL)
    return

############################################################################

def get_all_trajs_path(data_path):
    traj_paths = []
    for i in range(0,182):
        user_data_path = data_path + '/' + str(i).zfill(3)
        if os.path.exists(user_data_path):
            traj_data_path = user_data_path + '/Trajectory'
            if os.path.exists(traj_data_path):
                trajs_name = os.listdir(traj_data_path)
                traj_paths.extend([traj_data_path + '/' + traj_name for traj_name in trajs_name])
    return traj_paths


def read_traj(traj_path):
    df = pd.read_csv(traj_path, header=None, sep=',', skiprows=6, names=['lat', 'lon', 'zero', 'alt', 'days', 'date', 'time'])
    # df["timestamp"] = df["date"] + ' ' + df["time"]
    lats = df["lat"].to_list()
    lons = df["lon"].to_list()
    # times = df["timestamp"].to_list()
    trajs = []
    for lat, lon in zip(lats, lons):
        record = [lon, lat]
        trajs.append(record)
    return trajs


def batch_read_traj(traj_paths):
    all_trajs = []
    for i, traj_path in enumerate(traj_paths):
        traj = read_traj(traj_path)
        all_trajs.append(traj)
        if i % 100 == 0:
            print('read {} trajs'.format(i))
    print(f'{len(all_trajs )}done!')
    return all_trajs

def filter_data(src, cs: CellSpace):
    merc_seq_ = [list(lonlat2meters(p[0], p[1])) for p in src]
    tgt = [[cs.get_cellid_by_point(*merc), wgs, merc] for wgs, merc in zip(src, merc_seq_)]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i - 1][0]]
    tgt, wgs_seq, merc_seq = zip(*tgt)
    return list(wgs_seq), list(merc_seq)


def clean_and_output_data(data):
    _time = time.time()
    cellspace = pickle.load(open(cellspace_path,'rb'))
    dfraw = pd.DataFrame({'wgs_seq': [traj for traj in data]})

    # 1.range filter
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))
    dfraw = dfraw[dfraw.inrange == True]

    print('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))     #

    # 2.
    dfraw['wgs_seq'],dfraw['merc_seq'] = zip(* dfraw.wgs_seq.apply(lambda traj:filter_data(traj,cellspace)))

    # 3.len filter
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= min_traj_len) & (dfraw.trajlen <= max_traj_len)]
    print('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))        #

    # 4.output
    dfraw = dfraw[['wgs_seq', 'merc_seq']].reset_index(drop=True)

    dfraw.to_pickle(clean_data_path)
    print('Preprocess end. @={:.0f}'.format(time.time() - _time))
    return

def filtering_data():
    clean_data = pickle.load(open(clean_data_path, 'rb'))
    idx = random.sample(range(clean_data.shape[0]), 10000)
    data_1w = clean_data.iloc[idx]
    data_1w.to_pickle(fine_tuning_data_path)
    return


######################################################################################

###########################################################################
# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='hausdorff'):
    print("traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()

    data_1w = pickle.load(open(fine_tuning_data_path, 'rb'))
    data_1w.reset_index()
    assert data_1w.shape[0] == 10000
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
    fn =  {'discret_frechet': tdist.discret_frechet,'sspd':tdist.sspd,'hausdorff': tdist.hausdorff}.get(fn_name, None)
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


if __name__ == "__main__":
    min_lon = 116.25  # √
    max_lon = 116.5  # √
    min_lat = 39.8  # √
    max_lat = 40.1  # √
    cellspace_buffer = 500
    cell_size = 100

    min_traj_len = 20
    max_traj_len = 300

    root_path = os.getcwd()
    raw_data_path = root_path + "/Geolife Trajectories 1.3/Data/"
    cellspace_path = root_path + '/geolife_cellspace100.pkl'
    clean_data_path = root_path + '/clean_geolife.pkl'
    fine_tuning_data_path = root_path + "/geolife_1w.pkl"

    # 1. init_cellspace
    # init_cellspace()

    # 2
    # traj_paths = get_all_trajs_path(raw_data_path)      # 18670
    # trajs = batch_read_traj(traj_paths)                 #
    # print(len(trajs))
    # clean_and_output_data(trajs)
    """
    18670done!
    18670
    Preprocessed-rm range. #traj=14988
    Preprocessed-rm length. #traj=10940
    Preprocess end. @=31
    """
    # 3.
    # filtering_data()
    traj_simi_computation('sspd')            # ['hausdorff','sspd','discret_frechet']

