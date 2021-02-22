---
layout:     post
title:      "Understanding color & the in-camera Image Processing Pipeline for Computer Vision Part1"
date:       2021-2-22 23:30:00
author:     "liangyu"
header-img: "img/img-24cdd125021c794a0dfcc76aed9a3dad.jpg"
tags:
    - Computational Photography
    - ISP
---


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

  - [前言](#前言)
  - [Part1 (General)](#part1-general)
    - [Motivation](#motivation)
    - [Review of color & color spaces](#review-of-color-color-spaces)
      - [什么是Color](#什么是color)
      - [色彩的生理基础和SPD](#色彩的生理基础和spd)
        - [Spectral Power Distribution](#spectral-power-distribution)
      - [格拉斯曼定律](#格拉斯曼定律)
      - [活体实验来量化色彩](#活体实验来量化色彩)
        - [闪烁混合测试](#闪烁混合测试)
        - [色彩匹配实验](#色彩匹配实验)
        - [CIE RGB、CIE XYZ、CIE xyY色彩空间](#cie-rgb-cie-xyz-cie-xyy色彩空间)
        - [如何量化色彩？](#如何量化色彩)
      - [色彩、白平衡和色温](#色彩-白平衡和色温)
        - [色温](#色温)
        - [白平衡](#白平衡)
      - [sRGB与其他色彩空间](#srgb与其他色彩空间)
        - [linear-sRGB](#linear-srgb)
        - [sRGB](#srgb)
        - [其他色彩空间](#其他色彩空间)
          - [CIE LAB](#cie-lab)
          - [Y’UV、Y’IQ、Y’CrCb](#yuv-yiq-ycrcb)
      - [色差](#色差)
- [后记](#后记)
  - [参考资料](#参考资料)

<!-- /code_chunk_output -->

## 前言

Understanding color & the in-camera Image Processing Pipeline for Computer Vision 是Brown在ICCV2019上的一个tutorial，是一份很好的Computational photography的入门材料。本文为Part1，主要是color和color space。

## Part1 (General)

### Motivation

过去我们在研究相机、图像相关问题时，往往会有以下假设：

* Camera = light-measuring device（光测量设备）
* Image = radiant energy measurement（光辐射能量分布的量化）

<img src="/img/post-bg/image-20210115223745438.png" alt="image-20210115223745438" style="zoom: 67%;" />

有很多的计算机视觉的问题都是基于以上的假设进行研究的，包括Shape from shading、Image matching、HDR imaging等等。

**Digital cameras**:

* 相机不仅仅是一个光测量设备
* 现代相机用同样的参数拍同样的东西会有不同的风格
* 现代的相机设计的初衷并不是成为一个光测量设备，而是为了拍出**“好看”**的照片（视频）
* 相机内部会做大量的图像处理工作

<img src="/img/post-bg/image-20210115224337617.png" alt="image-20210115224337617" style="zoom: 67%;" />

这个教程的目标是讨论消费级相机上的一般图像处理步骤。

### Review of color & color spaces

#### 什么是Color

* 色彩是一种**光的视觉效应**。色彩依赖于object发射或者反射的**可见光**，经过**人的视觉系统**处理后，得到可见光对应的视觉效应
* Color不是object的主要物理属性
* **可见光**和**视觉效应**不是一一映射
* 小结：色彩是**可见光**分布在**视觉系统**作用后产出的**感受**

光的本质是电磁波，而可见光就是电磁波谱中很短的一段

<img src="/img/post-bg/image-20210117114528802.png" alt="image-20210117114528802" style="zoom:67%;" />

#### 色彩的生理基础和SPD

在眼睛中用于感受光的细胞有两种，**视杆（rod）细胞**和**视锥（cone）细胞**

* 视杆细胞不区分光线波长，主要在低光时起作用，分布在视网膜中心凹的边缘
* 视锥细胞则对波长敏感，按照敏感波长为三种：长、中、短，分布在视网膜的中间

<img src="/img/post-bg/image-20210117224226558.png" alt="image-20210117224226558" style="zoom: 50%;" />

上图是不同视锥细胞的敏感分布，横轴是波长，纵轴是敏感度。可以看到长和中视锥细胞的敏感曲线很接近，这是因为他们是从同一种细胞突变过来的，在那之前人类都是红绿色盲。

##### Spectral Power Distribution

可见光，其本质就是一段电磁波的分布，用**光谱能量分布（Spectral power distribution，SPD）**来表示，其横轴是波长，纵轴就是不同波长对应光的强度。

<img src="/img/post-bg/image-20210117224700877.png" alt="image-20210117224700877" style="zoom: 67%;" />

由此我们可以看出，对于眼睛接收到的可见光，三种不同的视锥细胞会给出三种信号，而我们看到的色彩就是依靠这三种信号重建出来的。所以，我们看到的色彩空间实际上就是**无穷维的SPD在三维上的投影**，这也解释了为什么我们在表示色彩时总是会取三个参数。

容易想到，这种高维向低维投影时，会出现不同SPD对应同一颜色，这种现象就叫做**条件等色（metamer）**，如下图所示，虽然SPD不同，但看起来却是一样的颜色。

<img src="/img/post-bg/image-20210117224859289.png" alt="image-20210117224859289" style="zoom: 67%;" />

#### 格拉斯曼定律

事实上，在了解生理结构之前，人们就经验性地知道了可以通过三种颜色混合出其他颜色，并且得到了一些结论。

一个重要的理论是**格拉斯曼定律（Grassmann’s law）**。这里提及这个定律是因为它给出了一个很重要的**经验结论**：人眼看到的色彩具有很强的**线性加和性质**。

<img src="/img/post-bg/image-20210117225513588.png" alt="image-20210117225513588" style="zoom: 50%;" />

#### 活体实验来量化色彩

##### 闪烁混合测试

目的是得到人对不同波长光的感受亮度。下面是人眼对不同波长光的敏感度的分布曲线，这条曲线又被称为**光度函数（luminosity function）**

<img src="/img/post-bg/image-20210117230858020.png" alt="image-20210117230858020" style="zoom: 67%;" />

##### 色彩匹配实验

目的是量化所有的可见颜色。

在1920年代，W. David Wright（Wright 1928）和John Guild（Guild 1931）独立进行了这一系列视觉实验，在实验中，让健康的志愿者来担任“标准观察者”，并使用2度视场角的圆形屏幕（固定视场角是因为视锥细胞集中分布于视网膜的中心凹区域，从而对色彩最敏感）。屏幕的一半投上**测试单色光**，另一半投上**可调整的光**。

可调整的光是**三种单色光**的混合，他们**波长固定（700 nm, 546 nm, 435 nm）**强度可调节，并称这三个波长为**原色**。 选择546.1 nm和435.8 nm的原色是因为它们是汞蒸气放电的颜色，容易复现，而选择700 nm是因为眼睛在700 nm处对光线的变化不敏感，波长的误差对感知的影响不大。

对于每个波长下的测试光，观察者可以分别调整三种**原色光**的强度，直到两侧的颜色看起来一样，并记录下三个原色光分别的强度。

在实验中发现，有些测试光，无论怎样调节三原色也无法匹配。在这样的情况下，要求观察者在**测试光**内加入一种可变强度的**原色光**来进行匹配，并将这个原色光的系数看作**负值**。

通过这样的方法，可以将人类感知色彩的范围**完全覆盖** 。

<img src="C:\Users\liangyu\AppData\Roaming\Typora\typora-user-images\image-20210123153828679.png" alt="image-20210123153828679" style="zoom:67%;" />

##### CIE RGB、CIE XYZ、CIE xyY色彩空间

<img src="/img/post-bg/image-20210117231617512.png" alt="image-20210117231617512" style="zoom: 50%;" />

**纵坐标表示为了匹配横坐标所对应的单色光需要的三原色的强度系数**。可以看到大约430 nm到530 nm的单色光需要红色的参与才能达成匹配。

这样的函数被称为**CIE RGB颜色匹配函数**，它的输入是一个**波长**，输出则是一个**代表着强度混合系数的三维向量**

CIE RGB函数包含了负值，为了方便使用，1931年CIE研究并定义了新的权威基准，被称为**CIE XYZ**，这套基准由CIE RGB数据变换得到。

<img src="/img/post-bg/image-20210117231946843.png" alt="image-20210117231946843" style="zoom: 50%;" />

CIE XYZ空间很棒，它不依赖于设备，不同电子设备可以将他们自己对颜色的表达映射到CIE XYZ空间上，从而（至少在理论上）得到了一种设备之间匹配的权威色彩空间。

有时候我们还会用明度（luminance）和色度（chromaticity）来讨论颜色，一个说的是感受到的明暗，一个说的是色调（hue）和饱和度（saturation），由此又发展出**CIE xyY空间**。令x = X/( X + Y + Z)、y = Y/( X + Y + Z)，第三个参数取和原来一样的Y（也是明度函数），这样就得到了CIE xyY空间。

在使用时，还经常将其投影到X + Y + Z = 1平面上，得到的图就叫**CIE xy色度图**

<img src="/img/post-bg/image-20210121225017458.png" alt="image-20210121225017458" style="zoom:67%;" />

##### 如何量化色彩？

假如我们拿到了一个SPD，我们希望得到它对应的XYZ（强度系数）。

SPD表示不同波长的光对应的强度，记作I(λ)；不同波长的光需要的XYZ的强度由CIE XYZ给出，由此我们计算XYZ：

<img src="/img/post-bg/image-20210117233305393.png" alt="image-20210117233305393" style="zoom:67%;" />

#### 色彩、白平衡和色温

我们已经可以用三个值（如CIE XYZ）来表示色彩了，但在现实场景中，我们看到的颜色是由物体的**反射特性**和**场景照明**共同决定的，而在之前的例子中我们都默认是纯白光照或者干脆就是单色光

<img src="/img/post-bg/image-20210118232044241.png" alt="image-20210118232044241" style="zoom: 67%;" />

上图中的苹果在不同的光照环境下有着**不同的SPD**，表现出不同的颜色。

##### 色温

色温的概念来自于黑体辐射理论。可以简单地将该理论理解为：对于一个理想的黑体，其发射出来的**电磁辐射**仅和它的**温度相关**。

<img src="/img/post-bg/image-20210118233024470.png" alt="image-20210118233024470" style="zoom: 67%;" />

##### 白平衡

在不同光照场景下，同一个object的视觉感受不同：

<img src="/img/post-bg/image-20210118233240330.png" alt="image-20210118233240330" style="zoom:67%;" />

为了方便描述，CIE建立了几种合成的SPD来作为真实光源的代表，编号列举如下：

* A 钨丝灯
* B 正午的阳光
* C 白天日光平均值
* D 不同色温下具有代表性的自然日光（5000K，5500K，6500K），一般写作D50，D55，D65
* E 理想的具有恒定SPD的等能量光源，并不代表任何真实光源，但和D55类似
* F 系列：模拟了各种荧光台灯（一共12个）

在色彩空间里，认为**光源对应着白点**，所以**色彩适应**的本质就是将不同光照下场景的**白点**变得相同。

<img src="/img/post-bg/image-20210118233804110.png" alt="image-20210118233804110" style="zoom:67%;" />

这张图里，曲线上的这些颜色其实都是某种场景里的**“白色”**。

<img src="/img/post-bg/image-20210118233905569.png" alt="image-20210118233905569" style="zoom:67%;" />

#### sRGB与其他色彩空间

##### linear-sRGB

尽管CIE XYZ是一个权威的色彩空间，但图像和设备却很少直接使用XYZ：

* XYZ本身并不代表实际的颜色，所以留下了很多空

* 在工业上还是更愿意使用RGB空间，虽然无法表示所有颜色，但容易控制和理解。

而为了让不同厂商的设备能够有个统一标准，于是sRGB诞生了

1996年，微软和惠普定义了一系列RGB三原色：

* R=CIE xyY (0.64, 0.33, 0.2126)
* G=CIE xyY (0.30, 0.60, 0.7153)
* B=CIE xyY (0.15, 0.06, 0.0721)

他们认为这是当时大多数设备能够达到的RGB空间。

白色点被设定为**D65光源**

注意，**指定白点是一件重要的事**，这意味着sRGB是在假设了**观看条件**的前提下建立的（6500K 日光）。每当我们把CIE XYZ映射到一个色彩空间时，都需要**指定白点**

<img src="/img/post-bg/image-20210121230458511.png" alt="image-20210121230458511" style="zoom:67%;" />

<img src="/img/post-bg/image-20210121230658195.png" alt="image-20210121230658195" style="zoom:50%;" />

##### sRGB

<img src="/img/post-bg/image-20210121231225378.png" alt="image-20210121231225378" style="zoom:67%;" />

<img src="/img/post-bg/image-20210121231516378.png" alt="image-20210121231516378" style="zoom: 50%;" />

上式名为Stevens’ power law，是用于描述这一现象的模型。I是输入图像的刺激值，S是经过处理后，符合人感受图像的刺激值。当a取小于1的正数，就是由相机的色彩空间向人的感受转换（编码），当a取大于1时，就是其逆变换（解码）。不同的模型会对系数采用不同的值，在sRGB中，a取2.2。

##### 其他色彩空间

<img src="/img/post-bg/image-20210121231905410.png" alt="image-20210121231905410" style="zoom: 50%;" />

###### CIE LAB

<img src="/img/post-bg/image-20210121232437162.png" alt="image-20210121232437162" style="zoom:50%;" />

###### Y’UV、Y’IQ、Y’CrCb

这些空间将RGB空间分解成 “"brightness-like”和“chrominance”

注意，这些色彩空间里的Y定义在Gamma的sRGB和NTSC色彩空间里。

按理说应该写成Y’以避免误会，但一般会写成Y。把YUV、YIQ、YCrCb中的Y当成CIE XYZ中的Y是计算机视觉中一个常见的错误。

这些空间下可以直接丢弃U和V来得到灰度图，可以用于黑白和彩色电视共用的信号。

#### 色差

* 基于CIE LAB空间定义了色差

<img src="/img/post-bg/image-20210121233358559.png" alt="image-20210121233358559" style="zoom: 50%;" />

***

# 后记

希望读完本文对你有所帮助。

由于本人水平有限，文中难免有疏漏之处，还请大家批评指正。

## 参考资料

1. [理解色彩与相机内图像处理流程——ICCV19 tutorial](https://blog.csdn.net/weixin_42028449/article/details/105340292)
2. [Understanding Color and the In-Camera Image Processing Pipeline for Computer Vision](https://www.eecs.yorku.ca/~mbrown/ICCV2019_Brown.html)


***

本作品采用<a rel="license" href="http://creativecommons.org/licenses/by/3.0/cn/">知识共享署名 3.0 中国大陆许可协议</a>进行许可。
