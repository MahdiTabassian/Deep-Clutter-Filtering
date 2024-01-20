"""
Module for generating batches of 2D images.  
"""
import os
import numpy as np
import random
import scipy.io as sio
import tensorflow as tf

def _shift_first_frame(vol_in, vol_out, tr_phase):
    n_frames = vol_in.shape[-1]
    n, p = 1, 0.5 # p is the probability of shifting the first frame.
    if tr_phase:
        if np.random.binomial(n,p):
            first_frm = np.random.permutation(np.arange(n_frames))[0]
            vol_in = np.concatenate((vol_in[:,:,first_frm:], vol_in[:,:,:first_frm]), axis=-1)
            vol_out = np.concatenate((vol_out[:,:,first_frm:], vol_out[:,:,:first_frm]), axis=-1)        	
    return vol_in, vol_out

def _image_vol_normalization(vol):
    vol = vol/255
    vol[vol < 0] = 0
    vol[vol > 1] = 1
    return vol

def _image_vol_augmentation(vol_in, vol_out, tr_phase):
    vol_in, vol_out = _shift_first_frame(vol_in, vol_out, tr_phase)
    vol_in_norm = _image_vol_normalization(vol_in)
    vol_out_norm = _image_vol_normalization(vol_out)
    vol_shape = [sh for sh in vol_in.shape]
    vol_shape.append(1)
    vol_in, vol_out = np.empty(vol_shape), np.empty(vol_shape)
    vol_in[:,:,:,0], vol_out[:,:,:,0] = vol_in_norm, vol_out_norm
    return [vol_in, vol_out]

class DataGen(tf.keras.utils.Sequence):
    """ 
    Generating batches of input cluttered volumes and their corresponding
    clutter-free output volumes 
    """
    def __init__(
        self,
        dim:list,
        in_dir:str,
        out_dir:str,
        id_list:list,
        batch_size:int,
        tr_phase=True,
        te_subsample=False,
        te_frames=0,
	    *args,
        **kwargs):
        'Initialization'
        self.dim = dim
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.id_list = id_list
        self.batch_size = batch_size
        self.tr_phase = tr_phase
		
    def __len__(self):
        return int(np.floor(len(self.id_list) / self.batch_size))
    
    def __getitem__(self, idx):
        batch = self.id_list[idx*self.batch_size:(idx+1)*self.batch_size]
        x_aug, y_aug = [], []
        for i, ID in enumerate(batch):
            vol_in = sio.loadmat(self.in_dir[ID])['data_artf']
            vol_out = sio.loadmat(self.out_dir[ID])['data_org']           
            # Call the data augmentation function 
            aug_vols = _image_vol_augmentation(vol_in, vol_out, self.tr_phase)
            x_aug.append(aug_vols[0])
            y_aug.append(aug_vols[1])
        if self.tr_phase:
            return np.asarray(x_aug), np.asarray(y_aug)
        else:
            return np.asarray(x_aug)