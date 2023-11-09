import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import h5py
from params import *
 
class H5Dataset:

    def __init__(self, filename, train_on = "all", re_normed = False):
        print("initializing h5 dataset...")
        with h5py.File(filename, 'r') as hf:
            if re_normed:
                x_data = hf["x"][:]
                y_e_data = hf["y_e"][:]
                y_zen_data = hf["y_zen"][:]
                y_azi_data = hf["y_azi"][:]
            else:
                x_data = hf["x"][:]
                y_e_data = hf["y_e"][:]
                y_zen_data = hf["y_zen"][:]
                y_azi_data = hf["y_azi"][:]
                y_x_data = hf["y_x"][:]
                y_y_data = hf["y_y"][:]
                y_z_data = hf["y_z"][:]
                y_inj_e_data = hf["y_inj_e"][:]
                y_inj_zen_data = hf["y_inj_zen"][:]
                y_inj_azi_data = hf["y_inj_azi"][:]
                y_inj_x_data = hf["y_inj_x"][:]
                y_inj_y_data = hf["y_inj_y"][:]
                y_inj_z_data = hf["y_inj_z"][:]
                y_num_hits = hf["ma_hit"][:]


        self.x_data = x_data.astype(np.float32)
        self.y_e_data = np.log10(1.0 + y_e_data.astype(np.float32))
        self.y_zen_data = np.cos(y_zen_data.astype(np.float32))
        self.y_azi_data = np.cos(y_azi_data.astype(np.float32))
        self.re_normed = re_normed
        if not re_normed:
            self.y_x_data = y_x_data.astype(np.float32)
            self.y_y_data = y_y_data.astype(np.float32)
            self.y_z_data = y_z_data.astype(np.float32)
            self.y_inj_e_data = np.log10(1.0 + y_inj_e_data.astype(np.float32))
            self.y_inj_zen_data = np.cos(y_inj_zen_data.astype(np.float32))
            self.y_inj_azi_data = np.cos(y_inj_azi_data.astype(np.float32))
            self.y_inj_x_data = y_inj_x_data.astype(np.float32)
            self.y_inj_y_data = y_inj_y_data.astype(np.float32)
            self.y_inj_z_data = y_inj_z_data.astype(np.float32)
            self.y_num_hits = y_num_hits.astype(np.float32)



        self.train_on = train_on

        # self.output_scaler = StandardScaler() # scaler for output
        # self.y_scaler = None
        print("finished initializing dataset")

    def X(self):
        # return np.expand_dims(self.data['x'], axis=-1)
        return self.x_data.transpose([0, 1, 4, 2, 3]).reshape(-1, SIZE_T, SIZE_H, SIZE_X, SIZE_Y)

    def Y(self):
        # try inplementing selection on the run
        if self.re_normed:
            output =  np.hstack([self.y_e_data.reshape(-1, 1), \
                    self.y_zen_data.reshape(-1, 1), \
                    self.y_azi_data.reshape(-1, 1)])
        else:
            output =  np.hstack([self.y_e_data.reshape(-1, 1), \
                        self.y_zen_data.reshape(-1, 1), \
                        self.y_azi_data.reshape(-1, 1),\
                        self.y_x_data.reshape(-1, 1), \
                        self.y_y_data.reshape(-1, 1), \
                        self.y_z_data.reshape(-1, 1), \
                        self.y_inj_e_data.reshape(-1, 1), \
                        self.y_inj_zen_data.reshape(-1, 1), \
                        self.y_inj_azi_data.reshape(-1, 1),\
                        self.y_inj_x_data.reshape(-1, 1), \
                        self.y_inj_y_data.reshape(-1, 1), \
                        self.y_inj_z_data.reshape(-1, 1), \
                        self.y_num_hits.reshape(-1, 1)])
        return output
