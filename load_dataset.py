from Max.dataset_lib import DataSet
import pandas as pd
import os

_PATH = "Max/bestest_hydronic_heat_pump_900_Random"
_U_LABEL = ["oveHeaPumY_u"]
_TVP_LABEL = ["time",
              "weaSta_reaWeaTDryBul_y",
              "weaSta_reaWeaRelHum_y",
              "weaSta_reaWeaHDirNor_y",
              "weaSta_reaWeaSolAlt_y",
              "InternalGainsCon[1]",
              "InternalGainsLat[1]",
              "InternalGainsRad[1]"]

_TVP_LABEL_EXTENDED = ["weaSta_reaWeaTDryBul_y",
                       "weaSta_reaWeaRelHum_y",
                       "weaSta_reaWeaHDirNor_y",
                       "weaSta_reaWeaSolAlt_y",
                       "Q_occ"]
_Y_LABEL = ["reaTZon_y"]
_FILE_LIST = ["train_1.csv", "train_2.csv", "train_3.csv", "train_4.csv", "train_5.csv"]


def get_dataset():
    final_dataset = None
    for file in _FILE_LIST:
        data_frame = pd.read_csv(os.path.join(_PATH, file))
        y = data_frame[_Y_LABEL].values
        u = data_frame[_U_LABEL + _TVP_LABEL].values

        local_dataset = DataSet(u=u, y=y)
        final_dataset = local_dataset if final_dataset is None else local_dataset + final_dataset
    return final_dataset