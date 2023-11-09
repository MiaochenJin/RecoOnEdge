import os 
import sys 
from datetime import datetime

LOCAL = False

BIG_MODEL = False

IN_NORM = False 

NORM_METHOD = "Gaussian_int_cutoff"

MEDIUM = 'Water'
if MEDIUM == 'Water':
    EXPTNAME = 'water_final'
elif MEDIUM == 'Ice':
    EXPTNAME = 'ice_final'

DETECTOR = "IC"

DATA_SIZE = 2000 # needs modification

TIME_SLICE = 30

TRAIN_ON = "zen_only"

BATCH_SIZE = 32
INIT_LR = 1*1e-4  # initial learning rate
EPOCH = 100

EVALUATE_EVERY = 500

NORM_METHOD = 'Gaussian_int_cutoff'

FULL_DATA = True

if FULL_DATA:
    DATA_SIZE = 300000 # for muon1 + muon2 data this is 206550 after my own cuts

# hyperparameter for LSTM, for TimeRNNModel
LSTM_DIM = 128
LSTM_DROPOUT=0.2
HIDDEN_2D = 128
if TRAIN_ON == "all":
    OUT_DIM = 2
elif TRAIN_ON == "zen_only":
    OUT_DIM = 1
elif TRAIN_ON == "azi_only":
    OUT_DIM = 1
elif TRAIN_ON == "E_only":
    OUT_DIM = 1
elif TRAIN_ON == "direction":
    OUT_DIM = 3
elif TRAIN_ON == "zen_dir":
    OUT_DIM = 2
else:
    print("invalid TRAIN_ON parameter provided")
    exit(0)

SIZE_T = TIME_SLICE
SIZE_H = 60
SIZE_X = 12
SIZE_Y = 12
if not IN_NORM:
    if not FULL_DATA:
        datafilename = "/n/holyscratch01/arguelles_delgado_lab/Everyone/miaochenjin/tpu_data/{}Hex_7_0-{}_ts={}.h5".format(MEDIUM, DATA_SIZE, TIME_SLICE)
    else:
        datafilename = "/n/holyscratch01/arguelles_delgado_lab/Everyone/miaochenjin/tpu_data/{}Hex_7_0-0_ts={}.h5".format(MEDIUM, TIME_SLICE)
if IN_NORM:
    if not FULL_DATA:
        datafilename = "/n/holyscratch01/arguelles_delgado_lab/Everyone/miaochenjin/tpu_data/{}_{}Hex_7_0-{}_ts={}.h5".format(NORM_METHOD, MEDIUM, DATA_SIZE, TIME_SLICE)
    else:
        datafilename = "/n/holyscratch01/arguelles_delgado_lab/Everyone/miaochenjin/tpu_data/{}_{}Hex_7_0-0_ts={}.h5".format(NORM_METHOD, MEDIUM, TIME_SLICE)

savename = "expts/{}/{}_loss.csv".format(EXPTNAME, TRAIN_ON)
checkpoint_path = "expts/{}/train_best_{}.ckpt".format(EXPTNAME, TRAIN_ON)
checkpoint_path = "expts/{}/train_best_{}.ckpt".format(EXPTNAME, TRAIN_ON)


if LOCAL:
    EXPTNAME = "0921_norm_water_2000_try"
    datafilename = "../local_data/Norm_WaterHex_7_0-2000_ts=30.h5"
    TRAIN_ON = "zen_only"
    if TRAIN_ON == "zen_only":
        checkpoint_path = "expts/{}/train_best_zen.ckpt".format(EXPTNAME)
    elif TRAIN_ON == "azi_only":
        checkpoint_path = "expts/{}/train_best_azi.ckpt".format(EXPTNAME)
    OUT_DIM = 1
    DATA_SIZE = 2000
    TIME_SLICE = 30
    SIZE_T = 30
    SIZE_H = 60
    SIZE_X = 12
    SIZE_Y = 12
    BATCH_SIZE = 64
    INIT_LR = 1e-4
    EPOCH = 30
    EVALUATE_EVERY = 50
    savename = "expts/{}/{}_loss.csv".format(EXPTNAME, TRAIN_ON)
