---
layout:     post
title:      "Jupyter Notebook 使用总结"
date:       2017-2-16 09:30:00
author:     "liangyu"
header-img: "img/1501227778125.jpg"
tags:
    - 折腾
    - 工具
---


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

* [为什么用 Jupyter notebook ?](#为什么用-jupyter-notebook)
* [Jupyter notebook 使用技巧](#jupyter-notebook-使用技巧)
	* [命令模式](#命令模式)
	* [编辑模式](#编辑模式)
* [Markdown 简介](#markdown-简介)
* [Jupyter notebook 演示](#jupyter-notebook-演示)
	* [载入相关库](#载入相关库)
	* [查看数据](#查看数据)
	* [描述数据](#描述数据)
	* [绘图](#绘图)
* [保存](#保存)
* [总结](#总结)

<!-- /code_chunk_output -->


## 为什么用 Jupyter notebook ?

平时用的最多的 Python 的 IDE 是 **Spyder**，在编辑器中编辑代码并在 **IPython** 中运行。但后来在处理一些特定的问题时很不方便，比如运行一些脚本时，希望能够保存每次运行的结果；写代码的时候希望能够记录下一些思路，当然可以用注释，但是这样显得很不整洁；拿到一批数据时，总是希望先可视化，试图发现一些规律，并把结果都保存下来。这时候用 IPython 就不那么方便了。

Ipython更适合写好代码后一次运行，而 **Jupyter notebook** 更像是一个笔记本，可以随时记录，可以直接运行代码直接显示结果，能将思路都完整保留下来，支持 Markdown 语言，很多人用它代替 PTT 做学术汇报。

## Jupyter notebook 使用技巧

单元前面的原色表示不同的状态，蓝色是 **命令模式**，可以对单元进行操作，包括插入新的单元，删除不想要的单元，也可以选择不同的输入状态；绿色是
**编辑模式**，可以编辑代码或者文字，用 ‘Enter’ 和 ‘ESC’ 可以切换不同状态。

通过快捷键操作，提高工作效率，也比较酷。下面是我常用的一些快捷键：

### 命令模式

* m: markdown 输入状态
* y: 代码输入状态
* Shift-Enter : 运行本单元，选中下个单元（如果没有就新建一个单元）
* Ctrl-Enter : 运行本单元
* Alt-Enter : 运行本单元，在其下插入新单元
* a: 在上方插入新单元
* b: 在下方插入新单元
* x: 剪切选中的单元
* c: 复制选中的单元
* Shift-V : 粘贴到上方单元
* v : 粘贴到下方单元
* z : 恢复删除的最后一个单元
* dd : 删除选中的单元
* Shift-M : 合并选中的单元

### 编辑模式

* Shift-Enter : 运行本单元，选中下个单元（如果没有就新建一个单元）
* Ctrl-Enter : 运行本单元
* Alt-Enter : 运行本单元，在其下插入新单元
* Ctrl-Home : 跳到单元开头
* Ctrl-End : 跳到单元开头

其他和普通的编辑器都一样。

## Markdown 简介

Markdown 是这一种标记语言，语法简洁明了，容易理解，几乎感觉不到语法的存在，GitHub 和很多博客平台都支持这种语法。平时我会用 Atom 编辑器，不要用它自带的 Markdown 插件，太不好用了，推荐 **markdown-preview-enhanced**，功能很强大。这里对语法不做介绍，可以看以下的链接：

1. [Markdown，你只需要掌握这几个](https://www.zybuluo.com/AntLog/note/63228)
2. [Markdown 语法说明 (简体中文版)](http://wowubuntu.com/markdown)

## Jupyter notebook 演示

### 载入相关库


```python
import numpy as np
from sklearn import datasets
import pandas as pd
```


```python
# 将iris数据转换为Dataframe格式
iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target
irisdata = np.concatenate((iris_X, iris_y.reshape(-1, 1)), axis=1)
columns_label = ['Sepallength', 'Sepalwidth', 'Petallength', 'Petalwidth', 'Species']
y_label = iris.target_names

irisDf = pd.DataFrame(data=irisdata,
                      columns=columns_label)

for i in xrange(3):
    irisDf['Species'].replace(i, y_label[i], inplace=True)
```

### 查看数据


```python
irisDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepallength</th>
      <th>Sepalwidth</th>
      <th>Petallength</th>
      <th>Petalwidth</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



### 描述数据


```python
irisDf.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepallength</th>
      <th>Sepalwidth</th>
      <th>Petallength</th>
      <th>Petalwidth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.054000</td>
      <td>3.758667</td>
      <td>1.198667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.433594</td>
      <td>1.764420</td>
      <td>0.763161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



### 绘图


```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid", color_codes=True)
%matplotlib inline
```


```python
sns.pairplot(irisDf, hue="Species")
```

![png-01](/img/post-bg/output_10_1.png)


可以看到，Jupyter notebook 对于数据可视化相当方便，能保存每一步的结果，思路很清晰；markdown 语法也比代码注释要美观的多。

## 保存

Jupyter notebook 直接保存为 .ipynb 文件，也可以保存为 .py .html .md .pdf 文件，保存为pdf我没有试过，可以看下下面这个链接：

[MAKING PUBLICATION READY PYTHON NOTEBOOKS](http://blog.juliusschulz.de/blog/ultimate-ipython-notebook)

## 总结

Jupyter notebook 功能很强大，不仅能运行 Python 代码，还能安装不同的 kernal 运行其他代码，其他功能我自己也在摸索中，目前我自己常用的就是这些了。

***

本作品采用 <a rel="license" href="http://creativecommons.org/licenses/by/3.0/cn/">知识共享署名 3.0 中国大陆许可协议</a> 进行许可。
