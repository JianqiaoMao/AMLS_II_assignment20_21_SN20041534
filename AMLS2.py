# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 02:02:38 2021

@author: Mao Jianqiao
"""
import os 
import numpy as np
import cv2
import time
import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Add, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model

def crop_subImage(img, size, stride, count, save_path):
    width, height = img.shape[1], img.shape[0]
    w_steps = int((width-size)/stride+1)
    h_steps = int((height-size)/stride+1)
    subImg_index = 0
    for w_step in range(w_steps):
        for h_step in range(h_steps):
            subImg = img[(stride*h_step):(stride*h_step+size), 
                           (stride*w_step):(stride*w_step+size),
                           :]
            cv2.imwrite(save_path+'\%d_%d.jpg'%(count,subImg_index),subImg)
            subImg_index+=1
            
def ProcessBar(count,sampleNum,startTime):
    bar='\r %.2f%% %s%s (%d processed, duration: %.2fs)'
    if (count+1)%1==0:
        duration=time.time()-startTime
        F='#'*int((count+1)/(sampleNum*0.025))
        nF='-'*int((sampleNum-(count+1))/(sampleNum*0.025))
        percentage=((count+1)/(sampleNum))*100
        print(bar %(percentage, F, nF, count+1,duration),end='') 

def patch_generator(reading_paths, writing_paths, scale, HR_patch_size=120, HR_stride=120):
     count = 0
     if scale == "HR":
         scale_int = 1
     elif scale == "X2" or scale == "X3" or scale == "X4":
         scale_int = int(scale[1])
     else:
         raise ValueError("scale param. must be 'HR', 'X2' or 'X3' or 'X4'")
     stride = int(HR_stride/(scale_int))
     subImg_size = int(HR_patch_size/(scale_int))
     start = time.time()
     print("\n")
     print("Cropping training images of "+scale+"...")
     for sample in range(len(reading_paths)):
        
        ProcessBar(count, len(reading_paths), start)
        img = cv2.imread(reading_paths[sample])
        crop_subImage(img = img, size = subImg_size, 
                      stride = stride, count = count, 
                      save_path = writing_paths)
        count+=1     
    
# function to process the uint8 image into float32 format
def preprocess_img(image):
    image = tf.image.decode_png(image,channels=3)
    image = tf.cast(image,tf.float32)
    image /= 255.0 
    return image

# function to read images from the given paths
def load_image(path):
    image = tf.io.read_file(path) 
    return preprocess_img(image)

# function to load and split the dataset
def load_training_data(path, factor, scale, batch_size, epoch, val_size=0.1, partial_data=None):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if partial_data is None:
        trainVal_HR_paths = glob.glob(path+"/HR/*")
        trainVal_LR_paths = glob.glob(path+"/"+factor+"/"+scale+"/*")
    else:
        trainVal_HR_paths = glob.glob(path+"/HR/*")[:partial_data]
        trainVal_LR_paths = glob.glob(path+"/"+factor+"/"+scale+"/*")[:partial_data]   
    
    total_samples = len(trainVal_HR_paths)
    val_size = int(0.1*total_samples)
    train_size = total_samples - val_size
    train_steps=tf.math.ceil(train_size/batch_size).numpy()
    
    ## Training Set
    train_HR_paths = trainVal_HR_paths[:train_size]
    train_bicubic2_paths = trainVal_LR_paths[:train_size]
    
    # Map the image paths into actual image data
    HRpath_tr_ds = tf.data.Dataset.from_tensor_slices(train_HR_paths)
    HR_tr_ds = HRpath_tr_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    
    X2path_tr_ds = tf.data.Dataset.from_tensor_slices(train_bicubic2_paths)
    X2_tr_ds = X2path_tr_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    
    # Make minibatches
    train_ds = tf.data.Dataset.zip((X2_tr_ds,HR_tr_ds))
    train_ds = train_ds.cache().shuffle(buffer_size=train_size, seed=2).repeat(epoch+1).batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) 
    
    # Validation Set
    val_HR_paths = trainVal_HR_paths[-val_size:]
    val_bicubic2_paths = trainVal_LR_paths[-val_size:]
    
    # Map the image paths into actual image data
    HRpath_val_ds = tf.data.Dataset.from_tensor_slices(val_HR_paths)
    HR_val_ds = HRpath_val_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    
    X2path_val_ds = tf.data.Dataset.from_tensor_slices(val_bicubic2_paths)
    X2_val_ds = X2path_val_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    
    # Make minibatches
    val_ds = tf.data.Dataset.zip((X2_val_ds,HR_val_ds))
    val_ds = val_ds.cache().shuffle(buffer_size=val_size, seed=17).batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, train_steps

# function to calculate the PSNR
def PSNR_cal(SR,HR):
    HR = tf.convert_to_tensor(HR)
    SR = tf.convert_to_tensor(SR)
    return tf.image.psnr(SR,HR,max_val=1.0)  

# function to calculate the SSIM
def SSIM_cal(SR,HR):
    HR = tf.convert_to_tensor(HR)
    SR = tf.convert_to_tensor(SR)
    return tf.image.ssim(SR,HR,max_val=1.0)   

# the strcture of basic convolutional block
def convolutional_block(X):
    
    X_shortcut = X
    
    X = Conv2D(64, (5,5), padding='same', activation='relu')(X)
    X = Conv2D(128, (3,3), padding='same', activation='relu')(X)
    X = Conv2D(64, (1,1), activation = 'relu', padding='same')(X)
    X = Conv2DTranspose(64, (5,5), activation = 'relu', strides=(2,2), padding='same')(X)
    
    X_shortcut = Conv2DTranspose(64, (1,1), strides=(2,2), padding='same')(X_shortcut)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

# Initialize FD-SRCNN of factor X2
def FD_SRCNN_X2():
    
    X_input = Input(shape=(None, None, 3))
    X = Conv2D(32, (9,9), padding='same', activation='relu')(X_input)
    X_shortcut = X
    
    # Block 1
    X = convolutional_block(X)
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)
    X = MaxPooling2D((2,2))(X)
    
    # Block 2
    X = convolutional_block(X)
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)   
    X = MaxPooling2D((2,2))(X)
    
    # Block 3
    X = convolutional_block(X)
    
    X_shortcut = Conv2DTranspose(64, (5,5), activation = 'relu', strides=(2,2), padding='same')(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)    

    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)   
    
    model = Model(inputs =  X_input, outputs=X)
    print(model.summary())
    
    return model

# Initialize FD-SRCNN of factor X3
def FD_SRCNN_X3():
    
    X_input = Input(shape=(None, None, 3))
    X = Conv2D(32, (9,9), padding='same', activation='relu')(X_input)
    X_shortcut = X
    # Block 1
    X = convolutional_block(X)
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)
    X = MaxPooling2D((3,3))(X)
    
#    # Block 2
    X = convolutional_block(X)
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)   
    X = MaxPooling2D((3,3))(X)
    
    # Block 3
    X = convolutional_block(X)
    
    X_shortcut = Conv2DTranspose(64, (5,5), activation = 'relu', strides=(3,3), padding='same')(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)   
    
    model = Model(inputs =  X_input, outputs=X)
    print(model.summary())
    
    return model

# Initialize FD-SRCNN of factor X4
def FD_SRCNN_X4():
    
    X_input = Input(shape=(None, None, 3))
    X = Conv2D(32, (9,9), padding='same', activation='relu')(X_input)
    
    X_shortcut = X
    X_shortcut1 = X
    # Block 1
    X = convolutional_block(X)
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)
    X = MaxPooling2D((2,2))(X)
    
    # # Block 2
    X = convolutional_block(X)
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)   
    X = MaxPooling2D((2,2))(X)
    
    # Block 3
    X = convolutional_block(X)
    
    X_shortcut1 = Conv2DTranspose(64, (5,5), activation = 'relu', strides=(2,2), padding='same')(X_shortcut1)
    X = Add()([X, X_shortcut1])
    X = Activation('relu')(X)
    
    X_shortcut2 = X
    
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)   
    
    # Block 4
    X = convolutional_block(X)
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)
    X = MaxPooling2D((2,2))(X)
    
    # # Block 5
    X = convolutional_block(X)
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)   
    X = MaxPooling2D((2,2))(X)
    
    # Block 6
    X = convolutional_block(X)
    
    X_shortcut2 = Conv2DTranspose(64, (5,5), activation = 'relu', strides=(2,2), padding='same')(X_shortcut2)
    X = Add()([X, X_shortcut2])
    X = Activation('relu')(X)
    
    X_shortcut = Conv2DTranspose(64, (5,5), activation = 'relu', strides=(4,4), padding='same')(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    X = Conv2D(3, (3,3), padding='same', activation='sigmoid')(X)   
    
    model = Model(inputs =  X_input, outputs = X)
    print(model.summary())
    
    return model


class model():
    
    """
    Works for model training, loading, predicting, evaluating and example demonstrating.
    
    Params.: 
        factor: degradation factor (bicubic/unknown)
        scale: scale of the given factor (X2/X3/X4)
        load_pretrain_model: default True, to indicate if the pretrained model should be used
        pretrain_model_path: default None, the path to load the pretrained models
    """
    
    def __init__(self, factor, scale, load_pretrain_model=True, pretrain_model_path=None):
        self.factor = factor
        self.scale = scale
        
        if load_pretrain_model:
            if pretrain_model_path is None:
                raise ValueError("Pretrained model loading path is required.")
            else:
                pretrain_model_path = pretrain_model_path+"/"+self.factor+"/FDSRCNN_"+self.factor+self.scale+".h5"
        
        if factor != 'bicubic' and factor != 'unknown':
            raise ValueError("factor param. must be 'bicubic' or 'unkown'")
        
        if scale == "X2":
            if load_pretrain_model:
                self.FDSRCNN = load_model(pretrain_model_path,custom_objects={'PSNR_cal': PSNR_cal, 'SSIM_cal':SSIM_cal})
                print("Model loaded.")
            else:
                self.FDSRCNN = FD_SRCNN_X2()
                print("Model initialized.")
        elif scale == 'X3':
            if load_pretrain_model:
                self.FDSRCNN = load_model(pretrain_model_path,custom_objects={'PSNR_cal': PSNR_cal, 'SSIM_cal':SSIM_cal})
                print("Model loaded.")
            else:
                self.FDSRCNN = FD_SRCNN_X3()
                print("Model initialized.")
        elif scale == "X4":
            if load_pretrain_model:
                self.FDSRCNN = load_model(pretrain_model_path,custom_objects={'PSNR_cal': PSNR_cal, 'SSIM_cal':SSIM_cal})
                print("Model loaded.")
            else:
                self.FDSRCNN = FD_SRCNN_X4()
                print("Model initialized.")
        else:
            raise ValueError("scale param. must be 'X2' or 'X3' or 'X4'")
        
        self.scale_int = int(scale[1])
     
    # function to train the initialized model    
    def train(self, train_data, val_data, epoch, train_steps, gpu_num=1, verbose=1):
        
        if gpu_num>1:
            self.FDSRCNN = multi_gpu_model(self.FDSRCNN, gpus=gpu_num) 
        
        self.FDSRCNN.compile(optimizer = 'adam' ,loss = ['mse'],metrics=[PSNR_cal,SSIM_cal])
       
        def scheduler(epoch):
         if epoch % 10 == 0 and epoch != 0:
             lr = K.get_value(self.FDSRCNN.optimizer.lr)
             K.set_value(self.FDSRCNN.optimizer.lr, lr * 0.5)
             print("lr changed to {}".format(lr * 0.5))
         return K.get_value(self.FDSRCNN.optimizer.lr)

        early_stopping=EarlyStopping(monitor='val_PSNR_cal', min_delta=0,
                                   patience=10, verbose=0, mode='max',
                                   baseline=None, restore_best_weights=True)

        reduce_lr = LearningRateScheduler(scheduler)

        self.FDSRCNN.fit(train_data, epochs=epoch, steps_per_epoch=int(train_steps), 
                         callbacks=[reduce_lr,early_stopping], verbose=verbose, 
                         validation_data=val_data)
        
        return self.FDSRCNN
    
    # function to reconstruct the SR image given the trained/loaded model
    def predict(self, LR):
        
        LR_h, LR_w = LR.shape[0], LR.shape[1]
        
        if isinstance(LR[0,0,0],np.uint8):
            pred_input = tf.convert_to_tensor(LR.reshape(-1,LR_h,LR_w,3)/255.,dtype=tf.float32)
        elif isinstance(LR[0,0,0],np.float64):
            pred_input = tf.convert_to_tensor(LR.reshape(-1,LR_h,LR_w,3),dtype=tf.float32)
        else:
            ValueError("LR image should be np.uint8 or np.float64")
        SR_img = (self.FDSRCNN.predict(pred_input)).reshape(LR_h*self.scale_int, LR_w*self.scale_int, 3)
        
        return SR_img
    
    # function to evaluate the performance of the trained/loaded model
    def evaluation(self, evaluation_path, proportion=1):
        
        LR_paths = glob.glob(evaluation_path + "/"+self.factor+"/"+self.scale+"/*")
        HR_paths = glob.glob(evaluation_path + "/HR/*")
        
        if proportion!=1:
            num_to_eval = int(proportion*len(LR_paths))
            LR_paths = LR_paths[:num_to_eval]
            HR_paths = HR_paths[:num_to_eval]          
        
        PSNR_list=[]
        SSIM_list=[]
        print("----------------------------------------------------")
        print("Evaluating on %d samples with "%(len(LR_paths))+self.factor+" factor of scale "+self.scale+"...")
        for sample in range(len(HR_paths)):
            
            LR_img = cv2.imread(LR_paths[sample])
            HR_img = cv2.imread(HR_paths[sample])
            HR_img = (HR_img/255.).astype("float32")
            SR_img = self.predict(LR=LR_img)
            PSNR_list.append(PSNR_cal(SR_img,HR_img))
            SSIM_list.append(SSIM_cal(SR_img,HR_img))
            
        ave_PSNR = sum(PSNR_list)/len(PSNR_list)
        ave_SSIM = sum(SSIM_list)/len(SSIM_list)
        
        self.PSNR_list = PSNR_list
        
        return ave_PSNR, ave_SSIM, PSNR_list, SSIM_list
    
    # function to write example SR images
    def write_example(self, evaluation_path, exp_index=None, write_path=""):

        LR_paths = sorted(glob.glob(evaluation_path + "/"+self.factor+"/"+self.scale+"/*"),reverse=True)
        HR_paths = sorted(glob.glob(evaluation_path + "/HR/*"),reverse=True)
        
        if exp_index == None:
            exp_index = [self.PSNR_list.index(max(self.PSNR_list))]
            print("Example output index is not assigned, the SR image with highest PSNR is selected.(index={})".format(exp_index[0]))
        elif not isinstance(exp_index, list):
            ValueError("Param. exp_index should be a list or None.")

        for i in exp_index:
            LR_exp = cv2.imread(LR_paths[i])
            LR_exp_h, LR_exp_w = LR_exp.shape[0], LR_exp .shape[1]
            LR_exp = LR_exp/255.
            LR_BIC = cv2.resize(LR_exp.astype("float32"), (LR_exp_w*self.scale_int, LR_exp_h*self.scale_int),interpolation=cv2.INTER_CUBIC)
            
            HR_exp = cv2.imread(HR_paths[i])
            HR_exp = (HR_exp/255.).astype("float32")
            
            SR_exp = self.predict(LR=LR_exp)
            
            PSNR_SR = round(PSNR_cal(SR_exp,HR_exp).numpy(),2)
            SSIM_SR = round(SSIM_cal(SR_exp,HR_exp).numpy(),4)
            PSNR_BIC = round(PSNR_cal(LR_BIC,HR_exp).numpy(),2)
            SSIM_BIC = round(SSIM_cal(LR_BIC,HR_exp).numpy(),4)
            
            cv2.imwrite(write_path+"LR_"+self.factor+"_"+self.scale+"_"+str(i)+"_PSNR"+str(PSNR_BIC)+"_SSIM"+str(SSIM_BIC)+".jpg", LR_BIC*255.)
            cv2.imwrite(write_path+"SR_"+self.factor+"_"+self.scale+"_"+str(i)+"_PSNR"+str(PSNR_SR)+"_SSIM"+str(SSIM_SR)+".jpg",SR_exp*255.)           
        
        
        
    

    
        