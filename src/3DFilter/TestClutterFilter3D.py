""" 
Test the 3D clutter filtering model.
In this module, it is possible to test the network on a subset of the views and
clutter classes.

"""
import os
import argparse
import json
import numpy as np
import pandas as pd

from utils import *
from Model_ClutterFilter3D import clutter_filter_3D
from DataGen import DataGen
from Error_analysis import compute_mae
    
def data_generation(in_ids_te, out_ids_te, config):
    DtaGenTe_prm = {
        'dim': config["network_prm"]["input_dim"],
        'in_dir': in_ids_te,
        'out_dir': out_ids_te,
        'id_list': np.arange(len(in_ids_te)),
        'batch_size': config["learning_prm"]["batch_size"],
        'tr_phase': False} 
    return DataGen(**DtaGenTe_prm)

def main(config):
    in_ids_te, out_ids_te, te_subject, val_subject = id_preparation(config)
    te_gen = data_generation(in_ids_te, out_ids_te, config)
    model = clutter_filter_3D(**config)
    weight_dir = create_weight_dir(val_subject, te_subject, config)
    model.load_weights(
        os.path.join(weight_dir, config["weight_name"] + ".hdf5"))
    results_te = model.predict_generator(te_gen, verbose=2)
    df_errors = compute_mae(in_ids_te, results_te)
    df_errors.to_csv(
        os.path.join(weight_dir, config["weight_name"] + ".csv"))
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path of the config file", default="config.json")
    args = parser.parse_args()
    assert os.path.isfile(args.config)
    with open(args.config, "r") as read_file:
        config = json.load(read_file)
    main(config)