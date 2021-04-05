---
layout:     post
title:      "Understanding color & the in-camera Image Processing Pipeline for Computer Vision Part2"
date:       2021-4-05 16:47:00
author:     "liangyu"
header-img: "img/img-24cdd125021c794a0dfcc76aed9a3dad.jpg"
tags:
    - Computational Photography
    - ISP
---

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Part1 (General)](#part1-general)
  - [Overview of in-camera imaging pipeline](#overview-of-in-camera-imaging-pipeline)
    - [Camera sensor](#camera-sensor)
      - [color filter array(CFA)](#color-filter-arraycfa)
    - [ISO gain and raw-image processing](#iso-gain-and-raw-image-processing)
      - [ISO](#iso)
      - [Black light subtraction](#black-light-subtraction)
      - [Defective pixel mask](#defective-pixel-mask)
      - [Flat-field correction](#flat-field-correction)
    - [RGB Demosaicing](#rgb-demosaicing)
    - [Noise reduction (NR)](#noise-reduction-nr)
    - [White-Balance & Color Space Transform](#white-balance-color-space-transform)
      - [White-Balance](#white-balance)
        - [手动白平衡](#手动白平衡)
        - [自动白平衡](#自动白平衡)
      - [Color Space Transform(CST)](#color-space-transformcst)
    - [Color Manipulation(Photo-finishing)](#color-manipulationphoto-finishing)
    - [Mapping to sRGB output](#mapping-to-srgb-output)
    - [JPEG Compression](#jpeg-compression)
    - [Save to file](#save-to-file)
      - [Exif metedata](#exif-metedata)
    - [ICC and Color Profiles](#icc-and-color-profiles)
    - [注意](#注意)
- [参考资料](#参考资料)

<!-- /code_chunk_output -->

## Part1 (General)

### Overview of in-camera imaging pipeline

**集成信号处理器（Integrated signal processor, ISP）**：

* 专门用来处理sensor image并生成最终图像的**硬件**。**ISP是硬件！**
* 通常会作为一个模块集成到SoC上
* 注意：也可以在设备的CPU或GPU上执行在ISP上的操作

<img src="/img/post-bg/image-20210123145757648.png" alt="image-20210123145757648" style="zoom:80%;" />

* 不同的ISP可能会以不同的顺序或方式应用这些步骤并且往往更加复杂，但一定包含这些步骤

#### Camera sensor

这里只介绍CMOS传感器

相机传感器是能够**衡量光线强弱**的装置，同样长的时间，如果到达传感器某个像素点的光亮度变成两倍，那么该点的数值也会变成两倍。

![image-20210123151835594](/img/post-bg/image-20210123151835594.png)

##### color filter array(CFA)

"Bayer" pattern or color filter array: 让相机传感器区分颜色的装置

<img src="/img/post-bg/image-20210123152634848.png" alt="image-20210123152634848" style="zoom: 50%;" />

通过这样的设计，传感器得到的数值就可以用来衡量不同通道信号的强弱。

不同的相机有着各自的CFA，不同CFA的**滤光色**构成了相机色彩空间（raw RGB）的**原色**，因此相机的raw RGB空间<u>**不是**</u>一个通用的色彩空间，和相机用的CFA对应。

下图是不同相机的色彩空间：

<img src="/img/post-bg/image-20210123153132537.png" alt="image-20210123153132537" style="zoom: 67%;" />

raw-RGB将物理世界的SPD投影到到传感器上。

![image-20210123154734363](/img/post-bg/image-20210123154734363.png)

#### ISO gain and raw-image processing

##### ISO

ISO: 感光度。对于现代相机，ISO并不是像快门时间或者光圈那样具有直白的物理含义，而是通过信号处理想要满足的标准

为了达到与设置相对应的ISO，相机会将接收到的信号进行**增益**，增益倍数越大，对光线越敏感，同时对噪音越敏感。

##### Black light subtraction

* sensor上无光的像素值应该为0
* 但由于sensor噪声，真实情况并非如此
* 随着温度的变化，黑电平也会变化
* 为了进行校准，可以通过黑屏设置一系列无光的像素，得到光学黑的信号，再从整体中减去，实现校准。

##### Defective pixel mask

为了处理sensor中的坏点。这一校准会在工厂中进行，通过拍摄无光的图像，来发现数值异常的点来制作mask，mask处的坏点像素值插值得到

##### Flat-field correction

<img src="/img/post-bg/image-20210124232316737.png" alt="image-20210124232316737" style="zoom:67%;" />

#### RGB Demosaicing

demosaicing其实就是插值，三倍插值

最简单的双线性插值：

<img src="/img/post-bg/image-20210126222011968.png" alt="image-20210126222011968" style="zoom:50%;" />

实际ISP中demosaicing算法要更复杂，还会和其他的处理组合在一起：

* Highlight clipping
* Sharpening
* Noise reduction

#### Noise reduction (NR)

* 所有的传感器都有噪声
* 大多数相机会在模数转换后加入降噪
* 对于高端相机，可以能会根据不同的ISO设置采用不同的降噪策略，当ISO较高时会采取更激进的降噪
* 手机的相机因为传感器较小，往往都会采取激进的降噪策略

**一种简单的降噪方法：**

* 基于ISO的设置调整图像blur的强度
* blur能降噪，但也会损失细节
* Add image detail back for regions that have a high signal

下面这种方法降噪的同时能保留图像的边缘细节：

<img src="/img/post-bg/image-20210126223732271.png" alt="image-20210126223732271" style="zoom: 50%;" />

#### White-Balance & Color Space Transform

* 到这里，我们得到了raw RGB
* 希望把raw RGB这一依赖于设备的色彩空间（由于CFA的存在）转换到一个**设备无关**的色彩空间里
* 在这里采用CIE XYZ为例子，实际上大多数相机会使用一个叫ProPhoto RGB的色彩空间

转换分为两步：

1. apply a white-balance correction to the raw-RGB values
2. map the white-balanced raw-RGB values to CIE XYZ

<img src="/img/post-bg/image-20210126225154259.png" alt="image-20210126225154259" style="zoom:50%;" />

##### White-Balance

* 白平衡就是矫正“白点”，把你认为的“白点”矫正为RGB三个分量相等
* 怎么定义“白点”？分为手动白平衡 and 自动白平衡

###### 手动白平衡

由用户手动设置“白点”。相机一般会提供一些预设的白平衡数据，用户根据拍照的光照环境进行选择。

下图认为图中青色是“白点”，把“白点”矫正为RGB相等：

<img src="/img/post-bg/image-20210126231102498.png" alt="image-20210126231102498" style="zoom:67%;" />

###### 自动白平衡

如果没有手动指定，就会启用**自动白平衡**（Auto White Balance, AWB），这件事就会变得很难，算法必须要能够确定任意照片的场景照明。

* 算法往往假设“白色”就是对场景光源的自然反射
* 如果我们可以定义图像中哪些像素属于“白色”，就可以得到场景光照的RGB表示。注意，这里的“白色”并不一定是白色，也有可能是灰色（白色就是最亮状态的灰色），有时我们会称这样的像素点为"achromatic" or "neutral"

两种基础的AWB算法：

* Gray world
* White patch

**Gray world：**

基本假设：

* This methods assumes that the **average reflectance** of a scene is **achromatic** (i.e. gray)
* This means that image average should have equal energy, i.e. R=G=B

像素的绝对值意义不大，可以将绿通道取1。

<img src="/img/post-bg/image-20210127231929522.png" alt="image-20210127231929522" style="zoom:67%;" />

**White patch：**

* 假设场景中的**高亮点**就是我们想要找的“白点”，也就是以像素最大值的RGB作为白色的数值。

<img src="/img/post-bg/image-20210127232409913.png" alt="image-20210127232409913" style="zoom:67%;" />

**小结：**

* 这两种算法都是非常基础的算法
* 当图像有大面积单色时很容易失败（比如蓝天）
* 对于AWB的算法研究有很多论文研究
* 相机往往会有自己独特的白平衡算法
* 注意，这些算法并不一定为了复原场景光照而设计，而是会出于**审美考虑**，留有一些色差

##### Color Space Transform(CST)

色彩空间转换（color space transform, CST）：将图片从raw RGB空间转换到独立于设备的色彩空间，如CIE XYZ。

* 工厂预标定了不同色温下的CST矩阵
* 根据白平衡计算得到的色温使用不同的CST矩阵
* 若某个色温没有标定，根据内置标定矩阵插值计算得到新的CST矩阵
  * g是根据CCT（Correlated Color Temperature）计算得到的权重

<img src="/img/post-bg/image-20210201231806246.png" alt="image-20210201231806246" style="zoom: 50%;" />

<img src="/img/post-bg/image-20210201232322785.png" alt="image-20210201232322785" style="zoom:50%;" />

#### Color Manipulation(Photo-finishing)

* 各家相机会施展不同的秘法，让照片变得更好看
* 允许用户选择不同的风格
* 这个步骤还有很多叫法：
  * Color manipulation 色彩调控
  * Photo-finishing 冲洗阶段
  * Color rendering or selective color rendering 色彩渲染
  * Yuv processing engine YUV处理引擎

色彩操作可以分为两类：**三维**和**一维**。

三维变换同时处理三个通道，一维曲线则是作用于每个通道。下图是一些变换的例子。**查询表（look up table, LUT）**就相当于函数或者说映射。

<img src="/img/post-bg/image-20210202205523843.png" alt="image-20210202205523843" style="zoom:50%;" />

<img src="/img/post-bg/image-20210202205607700.png" alt="image-20210202205607700" style="zoom:50%;" />

<img src="/img/post-bg/image-20210202210026000.png" alt="image-20210202210026000" style="zoom:50%;" />

#### Mapping to sRGB output

1. photo-finishing CIE XYZ 转换到 linear-sRGB
2. linear-sRGB 转换到 sRGB（Gamma encoding）

<img src="/img/post-bg/image-20210202231153382.png" alt="image-20210202231153382" style="zoom: 67%;" />

#### JPEG Compression

这部分略

#### Save to file

##### Exif metedata

* Exif: Exchangeable image file format, 可交换图像文件格式
* 由 Japan Electronics and Information Technology Industries Association (JEITA)组织创建
* 将meta data与images关联起来，包括：
  * 日期
  * 相机设置（基本）：图像尺寸，光圈，快门速度，焦距，ISO，测光模式
  * 其他信息：白平衡信息，风格，颜色空间（如sRGB、Adobe RGB、RAW），GPS信息
  * 更多...

#### ICC and Color Profiles

* ICC: International Color Consortium 国际色彩协会
* 负责制定颜色管理的ISO标准；推广使用ICC profiles

与raw RGB相关的RGB值称为“**scene referred**”；转换到sRGB空间后称为“**output referred**”

<img src="/img/post-bg/image-20210202213524146.png" alt="image-20210202213524146" style="zoom: 50%;" />

#### 注意

**sRGB和JPEG正慢慢被取代**

* sRGB是为了90年代的显示器设计的，太过于古老了
* JPEG也在逐步被压缩率更高的**HEIC编码**取代
* 苹果设备上已经开始用heic替代jpeg了，同时苹果设备使用Display P3的色彩空间，它是一种数字电影提倡的DCI-P3空间的变体，比sRGB要大25%，也包含了伽马变换
* 越来越多的安卓设备也会开始支持这个色彩空间

**Pipeline comments**

* 再次强调，上面的这些步骤仅仅是一个指南
* 现代相机中的处理流程会更加复杂
* 对于不同品牌/型号的相机，操作的顺序可能会有不同（如在去马赛克之后再白平衡），操作的方法也可能不一样（如把锐化和去马赛克结合）

## 参考资料

1. [理解色彩与相机内图像处理流程——ICCV19 tutorial](https://blog.csdn.net/weixin_42028449/article/details/105340292)
2. [Understanding Color and the In-Camera Image Processing Pipeline for Computer Vision](https://www.eecs.yorku.ca/~mbrown/ICCV2019_Brown.html)

***

本作品采用<a rel="license" href="http://creativecommons.org/licenses/by/3.0/cn/">知识共享署名 3.0 中国大陆许可协议</a>进行许可。
