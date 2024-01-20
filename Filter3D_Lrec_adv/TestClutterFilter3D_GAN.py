""" 
Module for testing the 3D clutter filtering model.
"""
import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(code_dir))
from Filter3D_L2_loss.utils import *
from Filter3D_L2_loss.Error_analysis import *
from Filter3D_L2_loss.Model_ClutterFilter3D import Unet3D
from DataGen import DataGen
from Discriminator import discriminator_3D
from Train_GAN import train_gan

def data_generation(in_ids_te, out_ids_te, config):
    DtaGenTe_prm = {
        'dim': config["generator_prm"]["input_dim"],
        'in_dir': in_ids_te,
        'out_dir': out_ids_te,
        'id_list': np.arange(len(in_ids_te)),
        'batch_size': config["learning_prm"]["batch_size"],
        'tr_phase': False} 
    return DataGen(**DtaGenTe_prm)

def generator_3D(**prm):
    prm_gen = prm
    prm_gen["network_prm"] = prm["generator_prm"]
    main_in = Input(prm_gen["network_prm"]["input_dim"])
    filter_out = Unet3D(x_in=main_in, name="CF", **prm_gen)
    model = Model(inputs=main_in, outputs=filter_out, name='generator_3D')
    model.summary()
    return model

def main(config):
    in_ids_te, out_ids_te, te_subject, val_subject = id_preparation(config)
    te_gen = data_generation(in_ids_te, out_ids_te, config)
    weight_dir = create_weight_dir(val_subject, te_subject, config) 
    generator = generator_3D(**config)
    generator.load_weights(os.path.join(weight_dir, config["weight_name"] + ".hdf5"))
    results_te = generator.predict_generator(te_gen, verbose=2)
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