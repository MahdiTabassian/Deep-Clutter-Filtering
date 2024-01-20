""" 
Module for training the 3D clutter filtering model with adversarial loss.
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
from Filter3D_Lrec.utils import *
from Filter3D_Lrec.Model_ClutterFilter3D import Unet3D
from DataGen import DataGen
from Discriminator import discriminator_3D
from Train_GAN import train_gan

def data_generation(in_ids_tr, out_ids_tr, config):
    DtaGenTr_prm = {
        'dim': config["generator_prm"]["input_dim"],
        'in_dir': in_ids_tr,
        'out_dir': out_ids_tr,
        'id_list': np.arange(len(in_ids_tr)),
        'batch_size': config["learning_prm"]["batch_size"],
        'tr_phase': True}
    return DataGen(**DtaGenTr_prm)

def generator_3D(**prm):
    prm_gen = prm
    prm_gen["network_prm"] = prm["generator_prm"]
    main_in = Input(prm_gen["network_prm"]["input_dim"])
    filter_out = Unet3D(x_in=main_in, name="CF", **prm_gen)
    model = Model(inputs=main_in, outputs=filter_out, name='generator_3D')
    model.summary()
    return model

def gan_3D(gen_model, disc_model, w_g, w_d, lr):
    disc_model.trainable = False
    disc_output = disc_model(gen_model.output)
    opt = Adam(lr=lr, beta_1=0.5)
    model = Model(gen_model.input, [gen_model.output, disc_output])
    model.compile(loss=['mse', 'binary_crossentropy'],
                  loss_weights=[w_g, w_d],
                  optimizer=opt)
    return model

def main(config):
    in_ids_tr, in_ids_val, out_ids_tr, out_ids_val, val_subject, te_subject = id_preparation(config)
    weight_dir = create_weight_dir(val_subject, te_subject, config)
    weight_name = (
      f'CF3D_GAN_ValTeSbj_{val_subject}_{te_subject}'
      f'_InSkp{config["generator_prm"]["in_skip"]}'
      f'_Att{config["generator_prm"]["attention"]}_lr{config["learning_prm"]["lr"]}'
      f'_wg{config["generator_prm"]["w_g"]}'
      f'_MaskedIn{config["discriminator_prm"]["masked_in"]}')
    tr_gen = data_generation(in_ids_tr, out_ids_tr, config)
    generator = generator_3D(**config)
    discriminator = discriminator_3D(config["learning_prm"]["lr"],
                                     **config["discriminator_prm"])
    gan_model = gan_3D(generator, discriminator, 
                       config["generator_prm"]["w_g"], config["discriminator_prm"]["w_d"],
                       config["learning_prm"]["lr"])
    train_gan(g_model=generator, d_model=discriminator, gan_model=gan_model,
              n_epochs=config["learning_prm"]["n_epochs"],
              tr_batches=tr_gen, w_dir=weight_dir, w_name=weight_name,
              masked_in=config["discriminator_prm"]['masked_in'])
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path of the config file", default="config.json")
    args = parser.parse_args()
    assert os.path.isfile(args.config)
    with open(args.config, "r") as read_file:
        config = json.load(read_file)
    main(config)