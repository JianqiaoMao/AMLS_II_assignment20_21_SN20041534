# AMLS_II_assignment20_21_SN20041534

## Overview

Image super-resolution (SR) aims to transform a low-resolution image (ILR) to a version with higher resolution, better visual quality and more details so-called SR image (ISR). Although many real-world applications can benefit from the SR, because of its ill-posed nature, it is still a challenging problem. 

This report (project) mainly focuses on dealing with [NTIRE 2017 challenge](https://data.vision.ee.ethz.ch/cvl/ntire17//), which is an example-based Single Image Super-Resolution (SISR) problem. The challenge is based on [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/), which have two types of degradation factors and three different scales for each factor. Training, validating and testing on the independent subsets of DIV2K dataset, respectively, the proposed SR models demonstrate promising performance in terms of PSNR and SSIM for both the bicubic and the unknown degradation factors of 2, 3 and 4. To tackle this challenge, the Fast Deep SRCNN (FD-SRCNN) based on the [SRCNN](https://link.springer.com/content/pdf/10.1007%2F978-3-319-10593-2_13.pdf) and [FSRCNN](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46475-6_25.pdf) is proposed with both local and global shortcut connections and iterative up-and-down sampling layers. 

The experiment results indicate that the proposed FD-SRCNN is able to achieve promising results in terms of PSNR and SSIM metrics with both the standard bicubic and complex unknown degradation factors under different scales specified by the challenge. Furthermore, it is found that the FD-SRCNN outperforms more significantly than the other comparable SR techniques for the more challenging unknown degradation factor.

#### Performance Comparisons on DIV2K Validation Set

<div align=center><img src=https://github.com/JianqiaoMao/AMLS_II_assignment20_21_SN20041534/blob/main/demo/performance_table.png width=1000 /></div>
