import tensorflow as tf
import keras
from skimage.transform import resize
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten,\
    BatchNormalization, Activation, GlobalAveragePooling2D, \
    Dropout, ReLU, Concatenate, Input, Add

from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from datetime import datetime
import cv2
import os.path
from keras.utils.vis_utils import plot_model
from scipy.spatial import distance
import scipy.io as sio

class NetworkModel:
  def custom_resnet(self, input_shape, output_shape):
    """
    input shape: [150,150,3]
    output shape: 9
    """
    input = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))

    '''block 1'''
    b1_cnv2d_1 = Conv2D(16,kernel_size=(3,3),strides=(2,2),padding='same',use_bias=False,
                        name='b1_cnv2d_1',kernel_initializer='normal')(input)
    b1_relu_1 = ReLU(name='b1_relu_1')(b1_cnv2d_1)
    b1_bn_1 = BatchNormalization(epsilon=1e-3,momentum=0.999,name='b1_bn_1')(b1_relu_1)

    b1_cnv2d_2 = Conv2D(32,kernel_size=(1,1),strides=(2,2),padding='same',use_bias=False,
                        name='b1_cnv2d_2',kernel_initializer='normal')(b1_bn_1)
    b1_relu_2 = ReLU(name='b1_relu_2')(b1_cnv2d_2)
    b1_out = BatchNormalization(epsilon=1e-3,momentum=0.999,name='b1_out')(b1_relu_2)


    '''block 2'''
    b2_cnv2d_1 = Conv2D(32,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=False,
                        name='b2_cnv2d_1',kernel_initializer='normal')(b1_out)
    b2_relu_1 = ReLU(name='b2_relu_1')(b2_cnv2d_1)
    b2_bn_1 = BatchNormalization(epsilon=1e-3,momentum=0.999,name='b2_bn_1')(b2_relu_1)

    b2_add = Add()([b1_out,b2_bn_1])

    b2_cnv2d_2 = Conv2D(64,kernel_size=(3,3),strides=(2,2),padding='same',use_bias=False,
                        name='b2_cnv2d_2',kernel_initializer='normal')(b2_add)
    b2_relu_2 = ReLU(name='b2_relu_2')(b2_cnv2d_2)
    b2_out =  BatchNormalization(epsilon=1e-3,momentum=0.999,name='b2_bn_2')(b2_relu_2)                 

    '''block 3'''
    b3_cnv2d_1 = Conv2D(64,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=False,
                        name='b3_cnv2d_1',kernel_initializer='normal')(b2_out)
    b3_relu_1 = ReLU(name='b3_relu_1')(b3_cnv2d_1)
    b3_bn_1 = BatchNormalization(epsilon=1e-3,momentum=0.999,name='b3_bn_1')(b3_relu_1)

    b3_add = Add()([b2_out,b3_bn_1])

    b3_cnv2d_2 = Conv2D(128,kernel_size=(3,3),strides=(2,2),padding='same',use_bias=False,
                        name='b3_cnv2d_2',kernel_initializer='normal')(b3_add)
    b3_relu_2 = ReLU(name='b3_relu_2')(b3_cnv2d_2)
    b3_out = BatchNormalization(epsilon=1e-3,momentum=0.999,name='b3_out')(b3_relu_2)   

    '''block 4'''
    b4_avg_p = GlobalAveragePooling2D()(b3_out)
    output = Dense(output_shape,name='model_output',activation='softmax',
                   kernel_initializer='he_uniform')(b4_avg_p)

    model = Model(input,output)
    model_json = model.to_json()

    with open("create_custom_residuel_CNN","w") as json_file:
      json_file.write(model_json)
    
    model.summary()
    return model

                  
