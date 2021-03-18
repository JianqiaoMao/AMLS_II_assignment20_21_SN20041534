# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 01:51:14 2021

@author: Mao Jianqiao
"""
#%% Configuration
import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

## Comment the below line if GPU(s) is(are) available
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

## Uncomment the below lines to assign GPUs

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# import tensorflow as tf
# gpus= tf.config.list_physical_devices('GPU') 
# print(gpus) 
# tf.config.experimental.set_memory_growth(gpus[0], True) 
# tf.config.experimental.set_memory_growth(gpus[1], True) 
# tf.config.experimental.set_memory_growth(gpus[2], True) 
# tf.config.experimental.set_memory_growth(gpus[3], True) 

import AMLS2

"""
The model uses full images for validation and testing, patches of image for training.

So it is recommended to build the dirs to store the original DIV2K dataset under "/Datasets/original_dataset/<factor>/<scale>/"
, where <factor> should be either bicubic or unknown.
For HR images, store it under "/Datasets/original_dataset/HR/"

The dirs for processed patches will be automatically built under the given the root path.
"""

original_img_path = r"./Datasets/original_dataset/"
processed_patch_path = r"./Datasets/processed_ds/"

test_img_path = original_img_path+ "test_set"
train_patch_path = processed_patch_path+"training"

pretrain_model_path = r"./selected_models"
model_save_path = r"./saved_models"

scales = ["X2", "X3", "X4"]
factors = ["bicubic", "unknown"]
exp_index=[22,61,79]
batch_size = 8
epoch = 10

if not os.path.exists(train_patch_path+'/'+"HR"):
    os.makedirs(train_patch_path+'/'+"HR") 
for factor in factors:
    for scale in scales:
        if not os.path.exists(train_patch_path+'/'+factor+"/"+scale):
            os.makedirs(train_patch_path+'/'+factor+"/"+scale) 
#%% Data Processing
"""
If need to train the model by yourself, 
this chunck of code should be run at first to generate patches from the original image.
"""
train_HR_readpaths = sorted(glob.glob(r".\Datasets\original_dataset\training_set\HR\*.png"), reverse=True)
AMLS2.patch_generator(reading_paths =train_HR_readpaths,
                      writing_paths=train_patch_path+'/'+"HR", scale="HR")

for factor in factors:
    for scale in scales:
        train_LR_readpath=sorted(glob.glob(r"./Datasets/original_dataset/training_set/"+factor+"/"+scale+"/*.png"), reverse=True)
        AMLS2.patch_generator(reading_paths =train_LR_readpath,
                              writing_paths=train_patch_path+'/'+factor+"/"+scale, scale=scale)     

#%% Track 1: Bicubic Degradation Factor

"""
For Track 1, the code follows a completed process of loading dataset, 
initializing the model, training and evaluating the model for only demo.
purpose.

While for the Track 2, the code loads the pretrained models, skipping training
process which is very time-consuming.
"""

factor ="bicubic"

for scale in scales:
    
    print("*"*100)
    print("Training the FD-SRCNN model for "+factor+" degradation factor of "+scale+"...")
    
    # Load the training and validation data (preprocessed into patches) as tf.Datasets object
    # For Demo. purpose, only 1,000 out of 149,787 patches are used in this case
    train_ds, val_ds, train_steps = AMLS2.load_training_data(path = train_patch_path, 
                                                             factor = factor, scale = scale, 
                                                             batch_size = batch_size, epoch=epoch, 
                                                             val_size=0.1, partial_data=1000)
    
    # Initialize the FD-SRCNN model for bicubic factor with a certian scale
    FDSRCNN = AMLS2.model(factor = factor, scale = scale,
                          load_pretrain_model=False)
    
    # Train the model
    FDSRCNN = FDSRCNN.train(train_data = train_ds, val_data = train_ds,
                            epoch = epoch, train_steps = train_steps, 
                            gpu_num=1, verbose=1)
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path) 
    FDSRCNN.save(model_save_path+"/FDSRCNN_"+factor+scale+".h5")
    
    ## The code below is avaiable to generate SR image if the LR image has been read.
    # SR = FDSRCNN.predict(LR_image)

    # Evaluate the trained model on test set (original images) in terms of PSNR and SSIM
    # For Demo. purpose, only 10% LR images in the test set are evaluated
    ave_PSNR, ave_SSIM, PSNR_list, SSIM_list = FDSRCNN.evaluation(evaluation_path=test_img_path, proportion=0.1)
    print("Test Ave. PSNR:{:.2f} dB, SSIM:{:.4f}".format(ave_PSNR,ave_SSIM))
    
    # Write examples SR for visualization, and you can specify the image written path
    print("Generate SR image of assgined examples...")
    FDSRCNN.write_example(evaluation_path=test_img_path, exp_index=exp_index, write_path="")

#%% Track 2: Unknown Degradation Factor

"""
Track 2, the code loads the pretrained models, skipping training
process which is very time-consuming.
"""

factor ="unknown"

for scale in scales:
    print("*"*100)
    print("Loading the FD-SRCNN model for "+factor+" degradation factor of "+scale+"...")
    
    # Load the FD-SRCNN model for unknown factor with a certian scale
    FDSRCNN = AMLS2.model(factor = factor, scale = scale, pretrain_model_path = pretrain_model_path)
    
    # Evaluate the trained model on test set (original images) in terms of PSNR and SSIM
    # For Demo. purpose, only 10% LR images in the test set are evaluated
    ave_PSNR, ave_SSIM, PSNR_list, SSIM_list = FDSRCNN.evaluation(evaluation_path=test_img_path, proportion=0.1)
    print("Test Ave. PSNR:{:.2f} dB, SSIM:{:.4f}".format(ave_PSNR,ave_SSIM))
    
    # Write examples SR for visualization, and you can specify the image written path
    print("Generate SR image of assgined examples...")
    FDSRCNN.write_example(evaluation_path=test_img_path, exp_index=exp_index, write_path="")
