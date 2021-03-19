# AMLS_II_assignment20_21_SN20041534

## Overview

Image super-resolution (SR) aims to transform a low-resolution image to a version with higher resolution, better visual quality and more details so-called SR image. Although many real-world applications can benefit from the SR, because of its ill-posed nature, it is still a challenging problem. 

This report (project) mainly focuses on dealing with [NTIRE 2017 challenge](https://data.vision.ee.ethz.ch/cvl/ntire17//), which is an example-based Single Image Super-Resolution (SISR) problem and proposes the the Fast Deep SRCNN (**FD-SRCNN**) based on the [SRCNN](https://link.springer.com/content/pdf/10.1007%2F978-3-319-10593-2_13.pdf) and [FSRCNN](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46475-6_25.pdf). The challenge is based on [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/), which have two types of degradation factors and three different scales for each factor. Training, validating and testing on the independent subsets of DIV2K dataset, respectively, the proposed SR models demonstrate promising performance in terms of PSNR and SSIM for both the bicubic and the unknown degradation factors of 2, 3 and 4.

The experiment results indicate that the proposed FD-SRCNN is able to achieve promising results in terms of PSNR and SSIM metrics with both the standard bicubic and complex unknown degradation factors under different scales specified by the challenge. Furthermore, it is found that the FD-SRCNN outperforms more significantly than the other comparable SR techniques for the more challenging unknown degradation factor. Although the proposed model is very deep, the training process only takes around 10 hours using data-paralleled strategy on GPUs, which is much faster than the compared works.

### Performance Comparisons on DIV2K Validation Set

<div align=center><img src=https://github.com/JianqiaoMao/AMLS_II_assignment20_21_SN20041534/blob/main/demo/performance_table.png width=1000 /></div>

### FD-SRCNN

The proposed FD-SRCNN consists of two upscaling modules containing functional blocks, extra neural network layers and local/global shortcut connections. Given an input low-resolution image, a convolutional layer and a ReLU activation layer perform as the input layer to preprocess the image into 32 feature maps by 9x9 convolutional kernels. Afterwards, the preprocessed feature maps are upsampled progressively with multi-stage upscaling modules.

Each upscaling module has three iterative up-and-down sampling block series, and each block series is a serial connection of three functional blocks: The convolutional block aims to deepen and widen the network, extracting abundant feature maps; The transposed-convolution-based upsampler is designed to adaptively learn appropriate upsampling kernels; The reconstructor makes its effort to estimate a high-quality super-resolution image, where the Sigmoid activation layer is stacked after the 3-channel convolutional layer to approach the realistic high-resolution image. Within an upscaling module, the iterative up-and-down sampling strategy is implemented by alternately using a transposed convolutional layer for upsampling and a maxpooling layer for downsampling. Since there is no downsampling layer after the last block series, the output of the upscaling module is a well-upscaled version of its input image. Furthermore, to fast the model converging speed and improve SR performance, shortcut connections are carefully placed amongst local block series and global modules. 

In this way, the proposed model is able to deal with multiple scale factors, where X2 and X3 scale super-resolution image are output at the end of the first upscaling module and the X4 scale super-resolution image is produced by the followed upscaling module.

<div align=center><img src=https://github.com/JianqiaoMao/AMLS_II_assignment20_21_SN20041534/blob/main/demo/FD-SRCNN.png width=1000 /></div>

### Demonstration

Figure below compares the SR results of the chosen images from DIV2K validation set, which are downsampled by the unknown factor and super-resolved by bicubic interpolation, original SRCNN, IUD-SRCNN, and the proposed FD-SRCNN. It can be observed that the final optimized FD-SRCNN have promising performance under all scales (X2, X3, X4) of the challenging unknown degradation factor. Especially under the higher scale unknown factor where the low-resolution image has been seriously blurred, the proposed FD-SRCNN can predict the almost realistic details and reconstruct a high-quality image.

<div align=center><img src=https://github.com/JianqiaoMao/AMLS_II_assignment20_21_SN20041534/blob/main/demo/SR%20results_word.png width=1000 /></div>

## File Description

1) The file folder **track1_bicubic** and **track2_unknown** contain individual scripts of the models that I have built and tested.

2) The file folder **Selected_models** contains pre-trained FD-SRCNN for every factors and scales.

3) The file folder **demo** contains some png files to demonstrate some examples the FD-SRCNN produces, and others are to present in the **readme** file.

4) The .py file **AMLS2.py** is the packaged module that should be imported to run **main.py**

5) The .py file **main.py** can be excuted to run the project.

## Dependent Environment and Tips

#### Dependent Environment

The whole project is developed in Python3.6. Please note that using other Python versions may lead to unknown errors. Required libraries are shown below, where the recommended package versions are also demonstrated:

  * numpy 1.19.2
  * opencv-python 4.4.0.46
  * tensorflow-gpu 2.3.1 / Alternative: tensorflow (latest version)
  * keras 2.4.3
  
Note that the FD-SRCNN are implemented based on tensorflow-gpu version, while it is uncertain for its compatibility in the basic tensorflow module. Conflict may happen if you install both of the two package within the same virtual environment, since base tensorflow is the default option to be imported. Some other dependent packages may be required, so you may need to install the missing packages as required.

#### Training Details

The **FD-SRCNN** is trained on 4 Nvidia 32gB Tesla V100 GPUs with 2 Intel® Xeon(R) Gold 6248 CPU at 2.50GHz. It is shown that though the model is very deep, the training process only takes around 10 hours for 50 epochs of training and validating on the GPUs using data-paralleled strategy. It is observed from the experiment that more training epochs may further improve the model performance, while this work is limited by objective resources.

#### Tips

The validation and testing use original full images, whereas the cropped patches are used for training. So it is recommended to build the dirs to store the original DIV2K dataset under "/Datasets/original_dataset/<factor>/<scale>/", where <factor> should be either **bicubic** or **unknown**. For HR images, store it under "/Datasets/original_dataset/HR/". If you want to train the model by yourself, the data preprocessing cell has to be run to produce the cropped pathces used in training phase. You are not required to create the dirs for processed patches, they will be automatically created under the given the root path, e.g. "/Datasets/processed_ds/training/...". Note that the data reading works in relative directory, try to excute **main.py** in the directory where it locates.

The **main.py** demonstrates track 1 with a completed process of loading dataset, initializing the model, training and evaluating the model, while only 1,000 out of 149,787 patches are used in the training phase to save time in the demo. For track 2, the code loads the pretrained models, skipping training phase which is very time-consuming.
