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
from TrajDiff_dif.utils.edwp import edwp

from skmob.preprocessing import detection

from TrajDiff.utils.tools import lonlat2meters,merc2cell2
from TrajDiff.utils.cellspace import CellSpace


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


###################################################################################
# 2.
def full_traj2trip(df, minutes_for_a_stop=5.0,spatial_radius_km=0.2):
    trips = []
    tdf = skmob.TrajDataFrame(df, latitude='latitude',longitude="longitude",datetime='time', user_id='taxi_id')
    stdf = detection.stay_locations(tdf, stop_radius_factor=0.5, minutes_for_a_stop=minutes_for_a_stop, spatial_radius_km=spatial_radius_km, leaving_time=True)
    if len(stdf) ==0 : # zero means this traj does not have stay points, only one trip
        total_time = tdf['datetime'].max() - tdf['datetime'].min()
        trips.append({"traj":tdf[['lng', 'lat']].values, "time":total_time.total_seconds()/60})
    else:
    # step 2. extract the start time and end time of each trip
        trip_start_time = tdf.datetime.min()
        trip_ranges = []
        for idx, row in stdf.iterrows():
            trip_end = row['datetime'] # stay start
            trip_start = row['leaving_datetime'] # stay end
            trip_ranges.append((trip_start_time, trip_end))
            trip_start_time = trip_start
        trip_ranges.append((trip_start_time, tdf.datetime.max()))
        trips = []
        for trip_range in trip_ranges:
            trip = tdf[(tdf.datetime >= trip_range[0]) & (tdf.datetime < trip_range[1])]
            if len(trip) > 0:
                trip = trip.sort_values(by='datetime')
                total_time = trip['datetime'].max() - trip['datetime'].min()
                trips.append({"traj":trip[['lng', 'lat']].values, "time":total_time.total_seconds()/60})
    return trips

def trip_extraction(minutes_for_a_stop=5.0, spatial_radius_km=0.1):
    all_trips = []
    for i in range(1, 10358):       # 10358
        taxi_df = None
        try:
            path = root_path + "/release/taxi_log_2008_by_id/{}.txt".format(i)
            taxi_df = pd.read_csv(path, header=None)
        except:
            print(f'error in {i}, empty file')
            continue
        taxi_df.columns = ['taxi_id', 'time', 'longitude', 'latitude']
        trips = full_traj2trip(taxi_df, minutes_for_a_stop=minutes_for_a_stop, spatial_radius_km=spatial_radius_km)
        all_trips.extend(trips)
        if i % 100 == 0:
            print(f'processed {i} files')

    tlen = [len(trip['traj']) for trip in all_trips]
    print(f'average trip length is {sum(tlen) / len(tlen)}')
    print(f'total trips {len(all_trips)}')
    return all_trips


def clean_and_output_data(data):
    _time = time.time()
    cellspace = pickle.load(open(cellspace_path,'rb'))
    dfraw = pd.DataFrame({'wgs_seq': [traj for traj in data]})

    # 1.range filter
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))
    dfraw = dfraw[dfraw.inrange == True]

    print('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))     #

    # 2.
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

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


if __name__ == '__main__':
    min_lon = 116.25  # √
    max_lon = 116.5  # √
    min_lat = 39.8  # √
    max_lat = 40.1  # √
    cellspace_buffer = 500
    cell_size = 100

    min_traj_len = 20
    max_traj_len = 200

    root_path = os.getcwd()
    cellspace_path = root_path + '/tdriver_cellspace100.pkl'
    clean_data_path = root_path + '/clean_tdriver.pkl'
    fine_tuning_data_path = root_path + "/tdriver_1w.pkl"
    # 1. init_cellspace
    # init_cellspace()

    # 2. extract the trips
    # extraction rules -> minutes_for_a_stop is 5 mins, spatial_radius is 100m
    # all_trips = trip_extraction(minutes_for_a_stop=5.0, spatial_radius_km=0.1)
    # trips = [trip["traj"] for trip in all_trips]
    # clean_and_output_data(trips)
    """
    Preprocessed-rm range. #traj=669350
    Preprocessed-rm length. #traj=33397
    Preprocess end. @=19
    """
    #
    # 3
    # filtering_data()
    traj_simi_computation('discret_frechet')         # ['hausdorff','sspd','discret_frechet']
