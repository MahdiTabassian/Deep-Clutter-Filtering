""" 
Functions for training discriminator model (ResNet) of the 3D clutter filtering network.
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv3D, MaxPooling3D, Dense, Activation, BatchNormalization, Add,
                                     Dropout, Concatenate, multiply, Input, Flatten, Lambda) 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def max_fnc(x):
    return K.max(x, axis=1, keepdims=False)

def ResNetModule(x, n_filters, n_blks, krnl_size=3, stride=1, name=None, **prm):
    
    shortcut = Conv3D(2*n_filters, 1, strides=stride, name=f"{name}_conv_shortcut")(x)
    shortcut = BatchNormalization(name=f"{name}_bn_shortcut")(shortcut)
    for i in range(n_blks):
        x = Conv3D(n_filters, krnl_size, padding='same', name=f"{name}_conv{i+1}")(x)
        x = BatchNormalization(name=f"{name}_bn{i+1}")(x)
        x = Activation('relu', name=f"{name}_relu{i+1}")(x)
        x = Dropout(0.05, name=f"{name}_drp{i+1}")(x)
    
    x = Conv3D(2*n_filters, 1, name=f"{name}_conv_last")(x)
    x = BatchNormalization(name=f"{name}_bn_last")(x)
    x = Add(name=f"{name}_add")([shortcut, x])
    x = Activation('relu', name=f"{name}_out")(x)
    return x

def ConvToFC(inp, kernel_size=3, name=None, **prm):
    inp_shape = K.int_shape(inp)
    x = Conv3D(filters=prm['ConvToFC'][0], kernel_size=(1,1,inp_shape[3]), 
               padding='valid', activation='relu', strides=1, name=f'{name}_1')(inp)
    x = Conv3D(filters=prm['ConvToFC'][1], kernel_size=(inp_shape[1],inp_shape[2],1),
               padding='valid', activation='relu', strides=1, name=f'{name}_2')(x)
    if len(prm['ConvToFC']) > 2:
        for i in range(2, len(prm['ConvToFC'])):
            x = Conv3D(filters=prm['ConvToFC'][i], kernel_size=1, padding='valid',
                       activation='relu', strides=1, name=f'{name}_{i+1}')(x)
    print(x.shape)
    x = Flatten()(x)
    return x

def ResNet(inp, n_krn, name=None, **prm):    
    x = Conv3D(n_krn, kernel_size=prm['kernel_size'], padding='same',
               data_format="channels_last", strides=1, name=f"{name}_conv0")(inp)
    x = BatchNormalization(name=f"{name}_bn0")(x)
    x = Activation('relu', name=f"{name}_relu0")(x)
    for i in range(len(prm['lvl_blks_config'])):        
        x = MaxPooling3D(pool_size=prm['pool_size'], strides=prm['strides'], name=f"{name}_pool{i+1}")(x)
        x = ResNetModule(x, n_filters=(2**i)*n_krn, krnl_size=prm['kernel_size'], stride=1, 
                         n_blks=prm['lvl_blks_config'][i], name=f"{name}_module{i+1}", **prm)       
    x = ConvToFC(x, name='CnvToFC', **prm)
    return x

def DenseLayer(x, name=None, **prm):
  for i in range(len(prm['dense_layer_spec'])):
    x = Dense(prm['dense_layer_spec'][i], activation='relu', name=f"{name}_DenseLayer_{i}")(x)
  return Dense(1, activation='sigmoid', name=f"{name}_DenseLayer_out")(x) 

def discriminator_3D(lr, **prm):
    inp_3D = Input(prm['input_dim'])
    conv_out = ResNet(inp_3D, n_krn=prm['n_init_filters'], name=prm['model_name'], **prm) 
    print(conv_out.shape)
    dense_out = DenseLayer(x=conv_out, name=prm['model_name'], **prm)
    conv_model = Model(inputs=inp_3D, outputs=dense_out, name=prm['model_name'])
    conv_model.summary()
    opt = Adam(learning_rate=lr)    
    conv_model.compile(optimizer=opt, loss=prm['loss'], metrics=prm['metrics'])     
    return conv_model