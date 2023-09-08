import os
import argparse
import json
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from utils import *
from Model_ClutterFilter3D import clutter_filter_3D
from DataGen import DataGen

def id_preparation(config):
    tr_subjects, val_subject, te_subject = generate_tr_val_te_subject_ids(
        subject_list=config["subject_list"], val_subject_id=config["CV"]["val_subject_id"])
    in_ids_tr, out_ids_tr = generate_data_ids(config["paths"]["data_path"], tr_subjects)
    in_ids_val, out_ids_val = generate_data_ids(config["paths"]["data_path"], val_subject)
    return in_ids_tr, in_ids_val, out_ids_tr, out_ids_val, val_subject, te_subject

def data_generation(in_ids_tr, in_ids_val, out_ids_tr, out_ids_val, config):
    DtaGenTr_prm = {
        'dim': config["network_prm"]["input_dim"],
        'in_dir': in_ids_tr,
        'out_dir': out_ids_tr,
        'id_list': np.arange(len(in_ids_tr)),
        'batch_size': config["learning_prm"]["batch_size"],
        'tr_phase': True}
    DtaGenVal_prm = {
        'dim': config["network_prm"]["input_dim"],
        'in_dir': in_ids_val,
        'out_dir': out_ids_val,
        'id_list': np.arange(len(in_ids_val)),
        'batch_size': config["learning_prm"]["batch_size"],
        'tr_phase': True}
    tr_gen = DataGen(**DtaGenTr_prm)
    val_gen = DataGen(**DtaGenVal_prm)
    return tr_gen, val_gen

def model_chkpnt(val_subject, te_subject, weight_dir, config):
    weight_name = (
            f'CF3D_ValTeSbj_{val_subject}_{te_subject}_nLvl{config["network_prm"]["n_levels"]}'
            f'_InSkp{config["network_prm"]["in_skip"]}_Att{config["network_prm"]["attention"]}'
            f'_Act{config["network_prm"]["act"]}_nInitFlt{config["network_prm"]["n_init_filters"]}_lr{config["learning_prm"]["lr"]}')
    filepath = (weight_dir + '/'+  weight_name +
                '_epc' + "{epoch:03d}" + '_trloss' + "{loss:.5f}" +
                '_valloss' + "{val_loss:.5f}" + ".hdf5")
    model_checkpoint = ModelCheckpoint(filepath=filepath,
                                       monitor="val_loss",
                                       verbose=0,
                                       save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=4, min_lr=1e-7)
    return model_checkpoint, reduce_lr

def main(config):
    in_ids_tr, in_ids_val, out_ids_tr, out_ids_val, val_subject, te_subject = id_preparation(config)
    weight_dir = create_weight_dir(val_subject, te_subject, config)
    tr_gen, val_gen = data_generation(in_ids_tr, in_ids_val, out_ids_tr, out_ids_val, config)
    model = clutter_filter_3D(**config)
    model_checkpoint, reduce_lr = model_chkpnt(val_subject, te_subject, weight_dir, config)
    model.fit(tr_gen,
              validation_data=val_gen,
              epochs=config["learning_prm"]["n_epochs"],
              verbose=1,
              callbacks=[model_checkpoint, reduce_lr])
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path of the config file", default="config.json")
    args = parser.parse_args()
    assert os.path.isfile(args.config)
    with open(args.config, "r") as read_file:
        config = json.load(read_file)
    main(config)


