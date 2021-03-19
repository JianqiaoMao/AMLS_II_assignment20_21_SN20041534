# AMLS_II_assignment20_21_SN20041534

## Overview

Image super-resolution (SR) aims to transform a low-resolution image to a version with higher resolution, better visual quality and more details so-called SR image. Although many real-world applications can benefit from the SR, because of its ill-posed nature, it is still a challenging problem. 

This report (project) mainly focuses on dealing with [NTIRE 2017 challenge](https://data.vision.ee.ethz.ch/cvl/ntire17//), which is an example-based Single Image Super-Resolution (SISR) problem and proposes the the Fast Deep SRCNN (**FD-SRCNN**) based on the [SRCNN](https://link.springer.com/content/pdf/10.1007%2F978-3-319-10593-2_13.pdf) and [FSRCNN](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46475-6_25.pdf). The challenge is based on [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/), which have two types of degradation factors and three different scales for each factor. Training, validating and testing on the independent subsets of DIV2K dataset, respectively, the proposed SR models demonstrate promising performance in terms of PSNR and SSIM for both the bicubic and the unknown degradation factors of 2, 3 and 4.

The experiment results indicate that the proposed FD-SRCNN is able to achieve promising results in terms of PSNR and SSIM metrics with both the standard bicubic and complex unknown degradation factors under different scales specified by the challenge. Furthermore, it is found that the FD-SRCNN outperforms more significantly than the other comparable SR techniques for the more challenging unknown degradation factor.

### Performance Comparisons on DIV2K Validation Set

<div align=center><img src=https://github.com/JianqiaoMao/AMLS_II_assignment20_21_SN20041534/blob/main/demo/performance_table.png width=1000 /></div>

### FD-SRCNN

The proposed FD-SRCNN consists of two upscaling modules containing functional blocks, extra neural network layers and local/global shortcut connections. Given an input low-resolution image, a convolutional layer and a ReLU activation layer perform as the input layer to preprocess the image into 32 feature maps by 9x9 convolutional kernels. Afterwards, the preprocessed feature maps are upsampled progressively with multi-stage upscaling modules.

Each upscaling module has three iterative up-and-down sampling block series, and each block series is a serial connection of three functional blocks: The convolutional block aims to deepen and widen the network, extracting abundant feature maps; The transposed-convolution-based upsampler is designed to adaptively learn appropriate upsampling kernels; The reconstructor makes its effort to estimate a high-quality super-resolution image, where the Sigmoid activation layer is stacked after the 3-channel convolutional layer to approach the realistic high-resolution image. Within an upscaling module, the iterative up-and-down sampling strategy is implemented by alternately using a transposed convolutional layer for upsampling and a maxpooling layer for downsampling. Since there is no downsampling layer after the last block series, the output of the upscaling module is a well-upscaled version of its input image. Furthermore, to fast the model converging speed and improve SR performance, shortcut connections are carefully placed amongst local block series and global modules. 

In this way, the proposed model is able to deal with multiple scale factors, where X2 and X3 scale super-resolution image are output at the end of the first upscaling module and the X4 scale super-resolution image is produced by the followed upscaling module.

<div align=center><img src=https://github.com/JianqiaoMao/AMLS_II_assignment20_21_SN20041534/blob/main/demo/FD-SRCNN.png width=1000 /></div>

### Demonstration

Figure below compares the SR results of the chosen image from DIV2K using bicubic interpolation, original SRCNN, IUD-SRCNN, and the proposed FD-SRCNN. It can be observed that the final optimized FD-SRCNN have promising performance under all scales (X2, X3, X4) of the challenging unknown degradation factor. Especially under the higher scale unknown factor where the low-resolution image has been seriously blurred, the proposed FD-SRCNN can predict the almost realistic details and reconstruct a high-quality image.

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

  * numpy 1.19.1
  * pandas 1.1.3
  * dlib 19.21.0
  * opencv-python 4.4.0.46
  * seaborn 0.11.0
  * matplotlib 3.3.2
  * scikit-learn 0.23.2
  * keras 2.4.3
  * tensorflow-gpu 2.3.1 / Alternative: tensorflow (latest version)
  
Note that the CNN models are built by tensorflow-gpu version, while it is uncertain for its compatibility in the basic tensorflow module. Conflict may happen if you have both of the two package, since base tensorflow is the default to be imported. Some other dependent libraries may be required to apply **face_recognition** module, if meet errors, please check [here](https://github.com/ageitgey/face_recognition).

#### Tips

The file reading directory (for dataset loading) is tested on Windows10 (x86) using Spyder as IDE, while uncertainty can be expected for running on OS or Linux or other IDEs. If errors encountered, please modify the **pathA**, **pathB**, **pathA_addition**, **pathB_addition** variables in **main.py** at line 8-11. The data reading works on relative directory, try to excute **main.py** in the directory where it locates.

