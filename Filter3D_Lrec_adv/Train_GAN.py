""" 
Functions for training the 3D GAN model.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model

def _apply_the_mask(in_g, in_d, gen_out, th=0.1):
    mask = np.abs(in_g-in_d)
    mask[mask < th] = 0
    mask[mask >= th] = 1
    in_d_masked = np.multiply(in_d, mask)
    gen_out_masked = np.multiply(gen_out, mask)
    return gen_out_masked, in_d_masked

def train_gan(g_model, d_model, gan_model, n_epochs, tr_batches, w_dir, w_name, masked_in, **prm):
    for i in range(n_epochs):
        print(f"epoch:{i}")       
        rnd_ids = np.random.permutation(tr_batches.__len__())
        for j in range(tr_batches.__len__()):
            #  Train Discriminator
            in_g, in_d = tr_batches.__getitem__(rnd_ids[j])[0], tr_batches.__getitem__(rnd_ids[j])[1]
            gen_out = g_model.predict(in_g) 
            if masked_in:
                gen_out, in_d = _apply_the_mask(in_g, in_d, gen_out)
            d_loss_r, d_acc_r = d_model.train_on_batch(in_d, np.ones((1, 1)))
            d_loss_f, d_acc_f = d_model.train_on_batch(gen_out, np.zeros((1, 1)))
            d_loss = 0.5 * np.add(d_loss_r, d_loss_f)
            #  Train Generator
            g_loss = gan_model.train_on_batch(tr_batches.__getitem__(rnd_ids[j])[0],
                                              [tr_batches.__getitem__(rnd_ids[j])[1], np.ones((1, 1))])
        # Save weights after each epoch
        filename = (w_dir + '/' + f"{w_name}_epc{i}_g_loss{np.round(g_loss, 3)}_d_loss{np.round(d_loss, 3)}" + ".hdf5")
        g_model.save_weights(filename)
    return None