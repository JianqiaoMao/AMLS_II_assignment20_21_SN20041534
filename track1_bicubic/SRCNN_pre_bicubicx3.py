# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 02:43:54 2021

@author: Mao Jianqiao
"""
#%%
import os
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import multi_gpu_model
import tensorflow.keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
gpus= tf.config.list_physical_devices('GPU') 
print(gpus) 
tf.config.experimental.set_memory_growth(gpus[0], True) 
tf.config.experimental.set_memory_growth(gpus[1], True) 
tf.config.experimental.set_memory_growth(gpus[2], True) 
tf.config.experimental.set_memory_growth(gpus[3], True) 

AUTOTUNE = tf.data.experimental.AUTOTUNE

def CNN():
    model = Sequential()
    
    model.add(Conv2DTranspose(32, (9,9), activation = 'relu', strides=(3,3), padding='same', input_shape=(None,None,3)))
    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu')) 
    model.add(Conv2D(64, (1,1), activation = 'relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation = 'relu', padding='same'))
    model.add(Conv2D(3, (3,3), padding='same', activation='sigmoid'))
    
    print(model.summary())
    
    return model
     
def preprocess_img(image):
    image = tf.image.decode_png(image,channels=3)
    image = tf.cast(image,tf.float32)
    image /= 255.0 
    return image
 
def load_image(path):
    image = tf.io.read_file(path) 
    return preprocess_img(image)

def evaluation(HR_paths, LR_paths, model):
    PSNR_list=[]
    SSIM_list=[]
    print("----------------------------------------------------")
    print("Evaluating on %d samples..."%(len(HR_paths)))
    for sample in range(len(HR_paths)):
        
        LR_img = cv2.imread(LR_paths[sample])
        HR_img = cv2.imread(HR_paths[sample])
        HR_img = (HR_img/255.).astype("float32")
        LR_h, LR_w = LR_img.shape[0], LR_img.shape[1]
        HR_h, HR_w = HR_img.shape[0], HR_img.shape[1]
        pred_input = tf.convert_to_tensor(LR_img.reshape(-1,LR_h,LR_w,3)/255.,dtype=tf.float32)
        SR_img = (model.predict(pred_input)).reshape(HR_h, HR_w, 3)
        PSNR_list.append(PSNR_cal(SR_img,HR_img))
        SSIM_list.append(SSIM_cal(SR_img,HR_img))
        
    ave_PSNR = sum(PSNR_list)/len(PSNR_list)
    ave_SSIM = sum(SSIM_list)/len(SSIM_list)
    
    return ave_PSNR, ave_SSIM, PSNR_list, SSIM_list

def write_example(HR_paths, LR_paths, model, exp_index=None, write_path=r"/home/uceejm3/exp/"):
    if exp_index == None:
        exp_index = test_PSNR_list.index(max(test_PSNR_list))
        print("Example output index is not assigned, the SR image with highest PSNR is selected.(index={})".format(exp_index))
    for i in exp_index:
        LR_exp = cv2.imread(LR_paths[i])
        LR_exp_h, LR_exp_w = LR_exp .shape[0], LR_exp .shape[1]
        HR_exp = cv2.imread(HR_paths[i])
        HR_exp = (HR_exp/255.).astype("float32")
        pred_exp = tf.convert_to_tensor(LR_exp.reshape(-1,LR_exp_h,LR_exp_w,3)/255.,dtype=tf.float32)
        SR_exp = model.predict(pred_exp)
        SR_exp = SR_exp.reshape(SR_exp.shape[1],SR_exp.shape[2],3)
        LR_exp = (LR_exp/255.).astype("float32")
        LR_BIC = cv2.resize(LR_exp, (SR_exp.shape[1],SR_exp.shape[0]),interpolation=cv2.INTER_CUBIC)
        PSNR_SR = round(PSNR_cal(SR_exp,HR_exp).numpy(),2)
        SSIM_SR = round(SSIM_cal(SR_exp,HR_exp).numpy(),4)
        PSNR_BIC = round(PSNR_cal(LR_BIC,HR_exp).numpy(),2)
        SSIM_BIC = round(SSIM_cal(LR_BIC,HR_exp).numpy(),4)
        cv2.imwrite(write_path+"LR_X3_"+str(i)+"_PSNR"+str(PSNR_BIC)+"_SSIM"+str(SSIM_BIC)+".jpg", LR_BIC*255.)
        cv2.imwrite(write_path+"SR_X3_pre_SRCNN_"+str(i)+"_PSNR"+str(PSNR_SR)+"_SSIM"+str(SSIM_SR)+".jpg",SR_exp*255.)  

def PSNR_cal(SR,HR):
    HR = tf.convert_to_tensor(HR)
    SR = tf.convert_to_tensor(SR)
    return tf.image.psnr(SR,HR,max_val=1.0)  

def SSIM_cal(SR,HR):
    HR = tf.convert_to_tensor(HR)
    SR = tf.convert_to_tensor(SR)
    return tf.image.ssim(SR,HR,max_val=1.0)    


#%%
trainVal_HR_paths = glob.glob(r"/home/uceejm3/AMLS2/project/processed_ds/training/HR/*")
trainVal_bicubic3_paths = glob.glob(r"/home/uceejm3/AMLS2/project/processed_ds/training/bicubic/X3/*")
gpu_num = len(gpus)
total_samples = len(trainVal_HR_paths)
val_size = int(0.1*total_samples)
train_size = total_samples - val_size
epochs = 50
batch_size = 8*gpu_num


# Training Set
train_HR_paths = trainVal_HR_paths[:train_size]
train_bicubic3_paths = trainVal_bicubic3_paths[:train_size]

HRpath_tr_ds = tf.data.Dataset.from_tensor_slices(train_HR_paths)
HR_tr_ds = HRpath_tr_ds.map(load_image, num_parallel_calls=AUTOTUNE)

X3path_tr_ds = tf.data.Dataset.from_tensor_slices(train_bicubic3_paths)
X3_tr_ds = X3path_tr_ds.map(load_image, num_parallel_calls=AUTOTUNE)

train_ds = tf.data.Dataset.zip((X3_tr_ds,HR_tr_ds))
train_ds = train_ds.cache().shuffle(buffer_size=train_size, seed=2).repeat(epochs+1).batch(batch_size)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) 

# Validation Set

val_HR_paths = trainVal_HR_paths[-val_size:]
val_bicubic3_paths = trainVal_bicubic3_paths[-val_size:]

HRpath_val_ds = tf.data.Dataset.from_tensor_slices(val_HR_paths)
HR_val_ds = HRpath_val_ds.map(load_image, num_parallel_calls=AUTOTUNE)

X3path_val_ds = tf.data.Dataset.from_tensor_slices(val_bicubic3_paths)
X3_val_ds = X3path_val_ds.map(load_image, num_parallel_calls=AUTOTUNE)

val_ds = tf.data.Dataset.zip((X3_val_ds,HR_val_ds))
val_ds = val_ds.cache().shuffle(buffer_size=val_size, seed=17).batch(batch_size)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE) 

SRCNN = CNN()
SRCNN = multi_gpu_model(SRCNN, gpus=gpu_num)   
SRCNN.compile(optimizer = 'adam' ,loss = ['mse'],metrics=[PSNR_cal,SSIM_cal])
steps=tf.math.ceil(train_size/batch_size).numpy()

def scheduler(epoch):
     if epoch % 10 == 0 and epoch != 0:
         lr = K.get_value(SRCNN.optimizer.lr)
         K.set_value(SRCNN.optimizer.lr, lr * 0.5)
         print("lr changed to {}".format(lr * 0.5))
     return K.get_value(SRCNN.optimizer.lr)

early_stopping=EarlyStopping(monitor='val_PSNR_cal', min_delta=0,
                           patience=10, verbose=0, mode='max',
                           baseline=None, restore_best_weights=True)

reduce_lr = LearningRateScheduler(scheduler)

SRCNN.fit(train_ds, epochs=epochs, steps_per_epoch=int(steps) , callbacks=[reduce_lr,early_stopping], verbose=2, validation_data=val_ds)


#%% Model save and load
SRCNN.save(r"/home/uceejm3/AMLS2/project/saved_models/pre_srcnn_bicubicX3.h5")
print("model has saved")

from tensorflow.keras.models import load_model
SRCNN = load_model(r'/home/uceejm3/AMLS2/project/good_models/bicubic/pre_srcnn_bicubicX3.h5',custom_objects={'PSNR_cal': PSNR_cal, 'SSIM_cal':SSIM_cal})
print("model loaded!")
#%% Evaluate on Training and Validationg Sets

trainVal_HR_eval_paths = sorted(glob.glob(r"/home/uceejm3/AMLS2/project/dataset/training_set/HR/*"), reverse=True)
trainVal_bicubic3_eval_paths = sorted(glob.glob(r"/home/uceejm3/AMLS2/project/dataset/training_set/bicubic/X3/*"), reverse=True)
total_eval_samples = len(trainVal_HR_eval_paths)
val_eval_size = int(0.1*total_eval_samples)
train_eval_size = total_eval_samples - val_eval_size

# Evaluation on training set
train_HR_eval_paths = trainVal_HR_eval_paths[:train_eval_size]
train_bicubic3_paths = trainVal_bicubic3_eval_paths[:train_eval_size]
train_PSNR, train_SSIM, train_PSNR_list, train_SSIM_list = evaluation(train_HR_eval_paths,train_bicubic3_paths,model=SRCNN)
print("Train Ave. PSNR:{:.2f} dB, SSIM:{:.4f}".format(train_PSNR,train_SSIM))

# Evaluation on validation set
val_HR_eval_paths = trainVal_HR_eval_paths[-val_eval_size:]
val_bicubic3_paths = trainVal_bicubic3_eval_paths[-val_eval_size:]
val_PSNR, val_SSIM, val_PSNR_list, val_SSIM_list = evaluation(val_HR_eval_paths,val_bicubic3_paths,model=SRCNN)
print("Validation Ave. PSNR:{:.2f} dB, SSIM:{:.4f}".format(val_PSNR,val_SSIM))

#%% Testing Phase
# Test Set

test_HR_paths = sorted(glob.glob(r"/home/uceejm3/AMLS2/project/dataset/test_set/HR/*"), reverse=True)
test_bicubic3_paths = sorted(glob.glob(r"/home/uceejm3/AMLS2/project/dataset/test_set/bicubic/X3/*"), reverse=True)

test_PSNR,test_SSIM, test_PSNR_list, test_SSIM_list = evaluation(test_HR_paths,test_bicubic3_paths,model=SRCNN)
exp_index=[22,61,79]
write_example(test_HR_paths, test_bicubic3_paths, model=SRCNN, exp_index=exp_index, write_path=r"/home/uceejm3/exp/")

print("Test Ave. PSNR:{:.2f} dB, SSIM:{:.4f}".format(test_PSNR,test_SSIM))
   











