"""
A module for generating batches of 2D images.  
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

def _reshape_vol(vol):
    new_vol = np.empty([vol.shape[-1], vol.shape[0], vol.shape[1]])
    for i in range(vol.shape[-1]):
	    new_vol[i,:,:] = vol[:,:,i]
    return new_vol

def _image_vol_augmentation(vol_in, vol_out, tr_phase):
    """ 
    Augmenting and normalizing the input and output image volumes. 
    """
    vol_in, vol_out = _shift_first_frame(vol_in, vol_out, tr_phase)
    vol_in_norm = _image_vol_normalization(vol_in)
    vol_out_norm = _image_vol_normalization(vol_out)
    return [_reshape_vol(vol_in_norm), _reshape_vol(vol_out_norm)]

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
        return len(self.id_list)
    
    def __getitem__(self, idx):
        'Generate one or more batches of data'
        
        # Initialization
        in_out_shape = [self.batch_size, self.dim[0], self.dim[1], self.dim[2]]
        x_aug, y_aug = np.empty(in_out_shape), np.empty(in_out_shape) 
        vol_id = self.id_list[idx:(idx + 1)]
        
        # Generate the data
        for i, ID in enumerate(vol_id):	
            # Store sample
            vol_in = sio.loadmat(self.in_dir[ID])['data_artf']
            vol_out = sio.loadmat(self.out_dir[ID])['data_org']           
            # Call the data augmentation function 
            aug_vols = _image_vol_augmentation(vol_in, vol_out, self.tr_phase)  
            x_aug[:,:,:,0], y_aug[:,:,:,0] = aug_vols[0], aug_vols[1]
        
        if self.tr_phase:
          return x_aug, y_aug
        else:
          return x_aug