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
import traj_dist.distance as tdist
from skmob.preprocessing import detection


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
            path = raw_data_path + "/release/taxi_log_2008_by_id/{}.txt".format(i)
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
    dfraw = pd.DataFrame({'wgs_seq': [traj for traj in data]})

    # 1.range filter
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))
    dfraw = dfraw[dfraw.inrange == True]

    print('==========> Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))     #

    # 2.
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

    # 3.len filter
    dfraw['traj_len'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.traj_len >= min_traj_len) & (dfraw.traj_len <= max_traj_len)]
    print('==========> Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))        #

    # 4.output
    dfraw = dfraw[['wgs_seq', 'merc_seq', "traj_len"]].reset_index(drop=True)

    dfraw.to_pickle(clean_data_path)
    print('==========> Preprocess end. @={:.0f}'.format(time.time() - _time))
    return None


def sample_trajs():
    clean_data = pickle.load(open(clean_data_path, 'rb'))
    data_1w = clean_data.tail(10000)
    data_1w.to_pickle(tdrive_1w)
    return


###########################################################################
# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='hausdorff'):
    print("traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()

    data_1w = pickle.load(open(tdrive_1w, 'rb'))
    data_1w.reset_index()
    assert data_1w.shape[0] == 10000
    l = data_1w.shape[0]

    # 2.
    fn = _get_simi_fn(fn_name)
    simi = _simi_matrix(fn, data_1w)

    _output_file = '{}/traj_simi_dict_{}_{}.pkl'.format(root_path, fn_name,str(min_traj_len)+str(max_traj_len))
    with open(_output_file, 'wb') as fh:
        tup = simi
        pickle.dump(tup, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print("traj_simi_computation ends. @={:.3f}".format(time.time() - _time))
    return tup



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
            simi_row.append( float(fn(t_i, t_j)) )
        simi.append(simi_row)
    print('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
                    .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi


if __name__ == '__main__':
    from TKDE_final.config import DATASET
    min_lon, min_lat, max_lon, max_lat = DATASET["tdrive"]["area_range"].values()
    min_traj_len, max_traj_len = DATASET['tdrive']["length"]
    cell_size = DATASET['tdrive']["cell_size"]

    root_path = os.getcwd()
    # 替换为自己的原始数据集地址
    raw_data_path = "/data/data_666/zhx111/dataset/tdrive"
    clean_data_path = root_path + '/clean_tdrive_{}.pkl'.format(str(min_traj_len)+str(max_traj_len))
    tdrive_1w = root_path + "/tdrive_1w_{}.pkl".format(str(min_traj_len)+str(max_traj_len))

    # 1. extract the trips
    # extraction rules -> minutes_for_a_stop is 5 mins, spatial_radius is 100m
    # all_trips = trip_extraction(minutes_for_a_stop=5.0, spatial_radius_km=0.1)
    # trips = [trip["traj"] for trip in all_trips]
    # clean_and_output_data(trips)

    #
    # 3
    # sample_trajs()
    traj_simi_computation('sspd')         # ['haus','sspd','DFD']
