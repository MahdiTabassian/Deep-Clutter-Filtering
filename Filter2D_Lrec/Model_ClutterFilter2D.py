"""
Functions of the 2D clutter filtering algorithm.
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Activation, BatchNormalization, Add,
                                     Dropout, Concatenate, UpSampling2D, multiply, Input, Lambda) 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def tensor_expansion(tensor, rep, axs):
    expanded_tensor = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=axs),
                             arguments={'repnum': rep})(tensor)
    return expanded_tensor

def attention_gate_block_2D(x, g, n_inter_filters=None, name=None, **prm):
    """ 
    Attention gate block.
    """
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g) 
    if n_inter_filters is None:
       n_inter_filters = shape_x[-1] // 2
         
    theta_x = Conv2D(n_inter_filters, 3, strides=(2, 2), padding='same', name=f"{name}_theta_x")(x)
    phi_g = Conv2D(n_inter_filters, 1, strides=1, padding='valid', name=f"{name}_phi_g")(g) 
    concat_xg = Add()([phi_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, 1, padding='same', name=f"{name}_psi")(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_sigmoid = UpSampling2D(size=(2, 2), name=f"{name}_upsampled_sig")(sigmoid_xg)
    upsample_sigmoid_rep = tensor_expansion(upsample_sigmoid, rep=shape_x[-1], axs=-1)
    y = multiply([upsample_sigmoid_rep, x], name=f"{name}_weighted_x")
    return y

def conv_block(x, filters_list, act='linear', kernel_size=3, stride=1, pad='same', drp=0.05, name=None):
    """ 
    Blocks of 2D conv filters.
    """
    for i in range(len(filters_list)):	
        x = Conv2D(filters_list[i], kernel_size, padding=pad, strides=stride, name=f"{name}_blk{i+1}")(x)
        x = BatchNormalization(name=f"{name}_bn{i+1}")(x)
        x = Activation(act, name=f"{name}_act{i+1}")(x)
        x = Dropout(drp)(x)
    return x

def encoding_block(x_in, name, **config):
    """
    Encoding block of the 2D Unet.
    """
    encoding_dct = {}
    for i in range(config["network_prm"]["n_levels"]):        
        if i == 0:
            x = x_in  
            n_filters = config["network_prm"]["n_init_filters"]
        else:
            n_filters = (2**i)*config["network_prm"]["n_init_filters"]
            x = MaxPooling2D(pool_size=config["network_prm"]["pool_size"], name=f"{name}_encd_pool{i}")(x)
        x = conv_block(x, filters_list=[n_filters, 2*n_filters], act=config["network_prm"]["act"],
	                   kernel_size=config["network_prm"]["kernel_size"],
                       stride=config["network_prm"]["conv_stride"], 
                       pad=config["network_prm"]["padding"],
                       drp=config["learning_prm"]['drp'], name=f"{name}_encd_conv_lvl{i}")
        encoding_dct[f"{name}_out_lvl{i}"] = x
    return encoding_dct

def decoding_block(encoding_dct, name, **config):
    """
    Decoding block of the 2D Unet.
    """
    decoding_dct = {}
    n_levels = config["network_prm"]["n_levels"]
    for i in range(n_levels-1):
        if i == 0:
            x = encoding_dct[f"{name}_out_lvl{n_levels-i-1}"]
	    # upsampling via Conv(Upsampling)
        x_shape = K.int_shape(x)
        x_up = Conv2D(x_shape[-1], 2, activation=config["network_prm"]["act"], padding='same', strides=1,
                      name=f"{name}_decd_upsmpl{i}")(UpSampling2D(size=(2,2))(x))
        x_up_shape = K.int_shape(x_up)
        # concatenation
        if config["network_prm"]['attention']:
            if i == 0:
                g = encoding_dct[f"{name}_out_lvl{n_levels-1}"]
            else:
                g = decoding_dct[f"{name}_out_lvl{i-1}"]
            x_encd = attention_gate_block_2D(x=encoding_dct[f"{name}_out_lvl{n_levels-i-2}"], 
                                             g=g, name=f"{name}_att_blk{i}")
        else:
            x_encd = encoding_dct[f"{name}_out_lvl{n_levels-i-2}"]
        x_concat = Concatenate(axis=-1, name=f"{name}_decd_concat{i}")([x_encd, x_up])
        n_filters = x_up_shape[-1]//2
        x = conv_block(x_concat, filters_list=[n_filters, n_filters], act=config["network_prm"]["act"],
		               kernel_size=config["network_prm"]["kernel_size"],
                       stride=config["network_prm"]["conv_stride"],
		               pad=config["network_prm"]["padding"],
                       drp=config["learning_prm"]['drp'], name=f"{name}_decd_conv_lvl{i}")
        decoding_dct[f"{name}_out_lvl{i}"] = x
    x = conv_block(x, filters_list=[1], act=config["network_prm"]["act"], kernel_size=1, stride=1, 
		           pad='same', drp=1e-4, name=f"{name}_final_decd_conv")
    decoding_dct[f"{name}_final_conv"] = x
    return decoding_dct

def Unet2D(x_in, name, **config):
    """
    Spatial clutter filtering based on 2D Unet.
    """
    encoding_dct = encoding_block(x_in, name, **config)
    decoding_dct = decoding_block(encoding_dct, name, **config)
    # Add the Main input here
    if config["network_prm"]["in_skip"]:
        out_Unet = Add()([x_in, decoding_dct[f"{name}_final_conv"]])
    else:
        out_Unet = decoding_dct[f"{name}_final_conv"]
    return out_Unet

def clutter_filter_2D(**config):
    """
    The main function for designing the clutter filtering algorithm.
    """
    main_in = Input(config["network_prm"]["input_dim"])
    filter_out = Unet2D(x_in=main_in, name="CF", **config)
    model = Model(inputs=main_in, outputs=filter_out, name=config['model_name'])
    opt = Adam(learning_rate=config["learning_prm"]["lr"])
    model.compile(optimizer=opt, loss=config["learning_prm"]["loss"], metrics=config["learning_prm"]["metrics"])
    model.summary()
    return model