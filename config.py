import os
base_path = os.getcwd()

DATASET = \
    {
        "pretrain_data_1": f"{base_path}/dataset/porto/clean_porto_20200.pkl",
        "pretrain_data_2": f"{base_path}/dataset/tdrive/clean_tdrive_20200.pkl",

        "porto": {
            "traj_data": f"{base_path}/dataset/porto/porto_1w_20200.pkl",
            "length": [20, 200],
            "dis_matrix": {
                "haus": f"{base_path}/dataset/porto/traj_simi_dict_haus_20200.pkl",
                "DFD": f"{base_path}/dataset/porto/traj_simi_dict_DFD_20200.pkl",
                "sspd": f"{base_path}/dataset/porto/traj_simi_dict_sspd_20200.pkl"},
            "area_range": {"min_lon": -8.7005, "min_lat": 41.1001, "max_lon": -8.5192, "max_lat": 41.2086},
            "cell_size": 100
        },

        "geolife": {
            "traj_data": f"{base_path}/dataset/geolife/geolife_1w_20300.pkl",
            "dis_matrix": {
                "sspd": f"{base_path}/dataset/geolife/traj_simi_dict_sspd_20300.pkl",
                "haus": f"{base_path}/dataset/geolife/traj_simi_dict_haus_20300.pkl",
                "DFD": f"{base_path}/dataset/geolife/traj_simi_dict_DFD_20300.pkl",},

            "area_range": {"min_lon": 116.25, "min_lat": 39.8, "max_lon": 116.5, "max_lat": 40.1},
            "cell_size": 100,
            "length": [20, 300],

        },
        "tdrive": {
            "traj_data": f"{base_path}/dataset/tdrive/tdrive_1w_20200.pkl",
            "dis_matrix": {
                "haus": f"{base_path}/dataset/tdrive/traj_simi_dict_haus_20200.pkl",
                "DFD": f"{base_path}/dataset/tdrive/traj_simi_dict_DFD_20200.pkl",
                "sspd": f"{base_path}/dataset/tdrive/traj_simi_dict_sspd_20200.pkl",},
            "area_range": {"min_lon": 116.25, "min_lat": 39.8, "max_lon": 116.5, "max_lat": 40.1},
            "cell_size": 100,
            "length": [20, 200],
        }
    }
