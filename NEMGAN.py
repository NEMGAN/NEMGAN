from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import time
import random
#from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import activations, initializers, regularizers, constraints, Lambda
from keras.engine import InputSpec
import tensorflow as tf
from keras.utils import to_categorical
from pdb import set_trace as trace

import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class NEMGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.Dec = None   # decoder
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        self.AE = None  # Autoencoder model
        
    def decoder(self, z_dim = 50):
        if self.Dec:
            return self.Dec  
        dropout = 0.4
        depth = 32
        img_input = Input(shape=(self.img_rows, self.img_cols, self.channel))
        x = Conv2D(depth*1, 5, strides=2, padding='same')(img_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(depth*2, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(depth*4, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)

        x = Conv2D(depth*8, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)
        
        x = Conv2D(depth*16, 5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout)(x)
        
        x = Flatten()(x)
        z = Dense(z_dim)(x)
        z = LeakyReLU(alpha=0.2)(z)
		
        x1 =  AMSoftmax(10, 30, 0.4)(z)
        self.Dec = Model(img_input, [z, x1], name='Decoder')

        return self.Dec

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D

        depth = 64
        dropout = 0.4

        img_input = Input(shape=(self.img_rows, self.img_cols, self.channel))
        x =  Conv2D(depth*1, 5, strides=2, padding='same')(img_input)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  Conv2D(depth*2, 5, strides=2, padding='same')(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  Conv2D(depth*4, 5, strides=2, padding='same')(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  Conv2D(depth*8, 5, strides=1, padding='same')(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dropout(dropout)(x)

        x =  Flatten()(x)
        x =  Dense(1)(x)
        x =  Activation('sigmoid')(x)
        self.D = Model(img_input, x, name='Discriminator')

        return self.D

    def generator(self, z_dim = 50):
        if self.G:
            return self.G

        dropout = 0.3
        depth = 32+32+32+32
        dim = 7

        g_input = Input(shape=[z_dim])
        x =  Dense(1024)(g_input)
        x =  BatchNormalization(momentum=0.9)(x)

        x =  LeakyReLU(alpha=0.2)(x)
        x =  Dense(dim*dim*depth)(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)
        x =  Reshape((dim, dim, depth))(x)
        x =  Dropout(dropout)(x)

        x =  UpSampling2D()(x)
        x =  Conv2DTranspose(int(depth/2), 5, padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)

        x =  UpSampling2D()(x)
        x =  Conv2DTranspose(int(depth/4), 5, padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)

        x =  Conv2DTranspose(int(depth/8),5 , padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)
        
        x =  Conv2DTranspose(int(depth/16), 5, padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)
        
        x =  Conv2DTranspose(int(depth/32),5, padding='same')(x)
        x =  BatchNormalization(momentum=0.9)(x)
        x =  LeakyReLU(alpha=0.2)(x)

        x =  Conv2DTranspose(1, 5, padding='same')(x)
        x =  Activation('sigmoid')(x)
        self.G = Model(g_input, x, name='Generator')

        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(lr=0.0002, decay=6e-8)
        self.discriminator().trainable=True
        self.DM = self.discriminator()
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
        return self.DM

    def adversarial_model(self, z_dim = 50):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=0.0001, decay=3e-8)
        gan_input = Input(shape=[z_dim])
        H = self.generator()(gan_input)
        self.discriminator().trainable=False
        V = self.discriminator()(H)
        self.AM = Model(gan_input, V)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
        return self.AM
    
    def autoencoder_model(self, z_dim = 50):
        if self.AE:
            return self.AE
        optimizer = Adam(lr=0.00001, decay=6e-8)
        ae_input = Input(shape=[z_dim])
        H = self.generator()(ae_input)
        [Vz, Vx] = self.decoder()(H)
        self.AE = Model(ae_input, [Vz, Vx])
        self.AE.compile(loss=['mae', amsoftmax_loss], loss_weights=[100.0, 1.0], optimizer=optimizer, metrics=['acc'])
        return self.AE
    
class AMSoftmax(Layer):
    def __init__(self, units, s, m,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMSoftmax, self).__init__(**kwargs)
        self.units = units
        self.s = s
        self.m = m
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs, **kwargs):
        inputs = tf.nn.l2_normalize(inputs, dim=-1)
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=(0, 1))   

        dis_cosin = K.dot(inputs, self.kernel)
        psi = dis_cosin - self.m

        e_costheta = K.exp(self.s * dis_cosin)
        e_psi = K.exp(self.s * psi)
        sum_x = K.sum(e_costheta, axis=-1, keepdims=True)

        temp = e_psi - e_costheta
        temp = temp + sum_x

        output = e_psi / temp
        return output


def amsoftmax_loss(y_true, y_pred):
    d1 = K.sum(y_true * y_pred, axis=-1)
    d1 = K.log(K.clip(d1, K.epsilon(), None))
    loss = -K.mean(d1, axis=-1)
    return loss

class MNIST_NEMGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.z_dim = 50
        
        self.x_train = input_data.read_data_sets(os.path.expanduser("~/mnist"), one_hot=True).train.images
        self.ylabel = input_data.read_data_sets(os.path.expanduser("~/mnist"), one_hot=True).train.labels
        self.x_test = input_data.read_data_sets(os.path.expanduser("~/mnist"), one_hot=True).test.images
        self.x_test=self.x_test.reshape( (self.x_test.shape[0],28,28,1 ) )
        self.ylabel_test = input_data.read_data_sets(os.path.expanduser("~/mnist"), one_hot=True).test.labels
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)
        print(self.ylabel.shape)
        print(self.x_train.shape)
        self.NEMGAN = NEMGAN()
        self.discriminator =  self.NEMGAN.discriminator_model()
        self.adversarial = self.NEMGAN.adversarial_model()
        self.generator = self.NEMGAN.generator()
        self.decoder = self.NEMGAN.decoder()
        self.autoencoder = self.NEMGAN.autoencoder_model()

    def data_gen_test(self,batch_size=256):
        
        ind_classes=[[],[],[],[],[],[],[],[],[],[]]
        
        for i in range(0,self.x_test.shape[0]):
            ind_classes[np.argmax(self.ylabel_test[i])].append(i)
            
        classes_data_x=[self.x_test[inds] for inds in ind_classes]
        classes_data_y=[self.ylabel_test[inds] for inds in ind_classes]
        
        wanted_list=[0,1,2,3,4,5,6,7,8,9]
        
        wanted_data_x=[]
        wanted_data_y=[]
        for i in wanted_list:
            wanted_data_x.append(classes_data_x[i])
            wanted_data_y.append(classes_data_y[i])
        
        x_test=np.concatenate(wanted_data_x)
        y_test=np.concatenate(wanted_data_y)              
        return x_test,y_test


    def data_gen_onefour(self,batch_size=256):
        ind_classes=[[],[],[],[],[],[],[],[],[],[]]        
        for i in range(0,self.x_train.shape[0]):
            ind_classes[np.argmax(self.ylabel[i])].append(i)            
        classes_data=[self.x_train[inds] for inds in ind_classes]                
        dataset=np.concatenate(classes_data)        
        ind_list=list(range(0,dataset.shape[0]))
        random.shuffle(ind_list)      
        while(1):
            x_batch=np.zeros((batch_size,self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3]))
           
            for sample in range(0,batch_size):
                x_batch=dataset[ind_list[:batch_size]]
                ind_list=ind_list[batch_size:]
                if(len(ind_list)<batch_size):
                    ind_list=list(range(0,dataset.shape[0]))
                    random.shuffle(ind_list)
       
            yield x_batch,None           

    def noise_gen_1d(self,batch_size=256,d_size=50000):
       y_dataset=np.zeros((d_size,self.z_dim))
       print('Total random samples: ',y_dataset.shape)      
       d_batch=d_size/10       
       for i in range(0,9):
           y_dataset[int(d_batch*i):int(d_batch*(i+1)),i*5]=1     
       y_dataset_noisy=y_dataset.copy()      
       y_dataset_noisy=y_dataset_noisy + np.random.uniform(-1*0.3,0.3, (y_dataset_noisy.shape) )      
       ind_list=list(range(0,y_dataset.shape[0]))
       random.shuffle(ind_list)              
       while(1):
           y_batch=y_dataset[ind_list[:batch_size]]
           y_batch_noisy=y_dataset_noisy[ind_list[:batch_size]]
           ind_list=ind_list[batch_size:]
           if(len(ind_list)<batch_size):
               ind_list=list(range(0,y_dataset.shape[0]))
               random.shuffle(ind_list)       
           yield y_batch_noisy,y_batch
           
    def noise_gen(self,batch_size=256,d_size=50000):
       y_dataset=np.zeros((d_size,self.z_dim))
       print('Total random samples: ',y_dataset.shape)     
       while(1):
           y_batch = np.zeros((batch_size, self.z_dim))
           for i in range(0,batch_size):
                y_batch[i,5*np.random.randint(10)] = 1
           y_batch_noisy = y_batch + np.random.uniform(-1*0.3,0.3, (y_batch.shape) )                     
           yield y_batch_noisy,y_batch
    
    def plot_images_save(self,noisy_one_hots=None,path=None):   
        images = self.generator.predict(noisy_one_hots)    
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(10, 10, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [28, 28])
            plt.imshow(image, cmap='gray')
            plt.axis('off')   
        plt.tight_layout()
        plt.savefig(path)
       
    def noise_gen_1d_plot(self,batch_size=100,class_i=0):
       ii=class_i
       y_out=np.zeros((batch_size,50))     
       y_out[:][:,5*ii]=1          
       y_out_un=y_out.copy()
       y_out=y_out + np.random.uniform(-1*0.3,0.3, (batch_size,50) )       
       yield y_out,y_out_un    

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        xtest,ytest=self.data_gen_test()
        fig_path = './figure/'
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        model_path = './model/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        pre_saved_ep = 0
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
            labels = np.zeros((batch_size, self.z_dim))
            for ni in range(0,batch_size):
                labels[ni,5*np.random.randint(10)] = 1
            noisy_one_hots = labels + np.random.uniform(-1*0.3,0.3, (labels.shape) )
            images_fake = self.generator.predict(noisy_one_hots)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)
            y = np.ones([batch_size, 1])            
            a_loss = self.adversarial.train_on_batch(noisy_one_hots, y)            
            labels_10d=np.zeros((labels.shape[0],10))
            for idd in range(0,labels_10d.shape[0]):
                labels_10d[idd][np.argmax(labels[idd])//5]=1

            ae_loss = self.autoencoder.train_on_batch(noisy_one_hots, [noisy_one_hots, labels_10d])

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            log_mesg = "%s  [AE loss total: %f, AE loss L1: %f, AE loss bce: %f, AE acc_z: %f, AE acc_l: %f]" % (log_mesg, ae_loss[0], ae_loss[1], ae_loss[2], ae_loss[3], ae_loss[4])
            
            
            if (i+1)%save_interval==0:
                for j in range(0,10):                    
                    nots,ots=next(self.noise_gen_1d_plot(batch_size=100,class_i=j))
                    self.plot_images_save(noisy_one_hots=nots,path=fig_path+'epoch_%d_mode_%d_.jpg' % (i+1+pre_saved_ep,j))                    
                    
                test_decoder=self.autoencoder.layers[-1]    
                pred_z, pred_y=test_decoder.predict(xtest)   
                
                ind_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
                val_dict={0:ind_dict.copy(),1:ind_dict.copy(),2:ind_dict.copy(),3:ind_dict.copy(),4:ind_dict.copy(),5:ind_dict.copy(),6:ind_dict.copy(),7:ind_dict.copy(),8:ind_dict.copy(),9:ind_dict.copy()}
                
                for j in range(0,pred_y.shape[0]):
                    val_dict[np.argmax(ytest[j])][np.argmax(pred_y[j])]+=1
                               
                acc_test=0
                for j in range(0,10):
                    dict_s=val_dict[j]
                    print('\n',j,':\n')
                    tup=sorted(dict_s.items(), key=lambda x:x[1],reverse=True)
                    
                    sum_v=0.0000001
                    for ele in tup:
                        sum_v+=ele[1]
                        print(ele,end=', ')
                    print('Accuracy=',tup[0][1]*100/sum_v)
                    acc_test+=tup[0][1]*100/sum_v
            
                print('\nTotal Accuracy=',acc_test/10)
            
            print(log_mesg)
                  
            if save_interval>0:
               if (i+1)%save_interval==0:
                   print('*********************Saving Weights***********************')
                   self.discriminator.save_weights(os.path.expanduser(model_path+'gan_dircriminator_epoch_%d.h5' % (i+1+pre_saved_ep)))
                   self.generator.save_weights(os.path.expanduser(model_path+'gan_generator_epoch_%d.h5' % (i+1+pre_saved_ep)))
                   self.adversarial.save_weights(os.path.expanduser(model_path+'gan_adversarial_epoch_%d.h5' % (i+1+pre_saved_ep)))
                   self.decoder.save_weights(os.path.expanduser(model_path+'gan_decoder_epoch_%d.h5' % (i+1+pre_saved_ep)))
                   self.autoencoder.save_weights(os.path.expanduser(model_path+'gan_autoencoder_epoch_%d.h5' % (i+1+pre_saved_ep)))


if __name__ == '__main__':
    mnist_nemgan = MNIST_NEMGAN()
    timer = ElapsedTimer()
    mnist_nemgan.train(train_steps=200000, batch_size=64, save_interval=5000)
    timer.elapsed_time()






