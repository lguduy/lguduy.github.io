---
layout:     post
title:      "TensorFlow教程：利用TensorFlow建立CNN模型"
date:       2017-6-20 22:30:00
author:     "liangyu"
header-img: "img/Beautiful time.jpg"
tags:
    - Deep Learning
    - TensorFlow
---

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

* [深度学习三要素](#深度学习三要素)
* [搭建网络（模型）](#搭建网络模型)
	* [前期准备](#前期准备)
	* [网络结构](#网络结构)
	* [代码实现](#代码实现)
* [损失函数（策略）](#损失函数策略)
* [优化器（算法）](#优化器算法)
* [准确率](#准确率)
* [总结](#总结)

<!-- /code_chunk_output -->


## 深度学习三要素

李航博士曾在《统计机器学习》中提到：

> 统计学习方法三要素：模型、策略和算法.

我认为深度学习也有这三要素：

1. 模型就是假设空间。在深度学习领域，模型包括网络结构和网络中的参数（权重和偏置等）。通常所说的CNN，RNN其实就是模型，但只是深度学习的一个要素而已，光有模型是没办法学习的。在CNN被提出之时，并没有提出相应的学习策略和学习算法因而并没有取得很好的效果。
2. 策略就是从数学上衡量什么样的模型才是好的，把一个模型的优劣量化。对应的就是损失函数，对于不同的问题要用不同的损失函数，分类问题一般用对数似然损失函数，而回归问题一般用平方损失函数。
3. 算法是指学习模型的具体计算方法，也就是参数寻优的过程。很多人在问机器学习到底是怎么学习的，参数寻优就是通常所说的学习过程。**模型和策略共同定义了一个目标函数**，算法的任务就是找到在训练样本上使目标函数最小的参数，所以这已经是最优化的内容。深度学习最常用的算法就是梯度下降法，目前也有了很多改进的基于梯度下降的算法。


## 搭建网络（模型）

搭建网络也就是搭建一个模型而已，只是学习的一个要素。

### 前期准备

在训练中总是希望能实时看到训练情况，而不只是一个黑箱，**Tensorboard** 就提供了这样的功能。通过Tensorboard
可以实时观察训练验证损失和准确率，能看到输入图像，并将卷积核和激活数据可视化，可以看到模型参数
和激活值的分布和稀疏情况，从而判断当前你的网络是否健康。

下面两个函数是我仿照CIFAR10写的，第一个函数是针对每层网络的激活值，可视化分布和稀疏程度，第二个
函数是针对网络参数（权重和偏置），可视化参数分布。都是用来监测网络训练状态，避免训练无效，比如，在
训练中可能出现ReLU大面积的死亡，网络参数都趋于0等问题。

```python
def _activation_summary(activations):
    """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.

    Parameters:
    -----------
        activation : tensor, activation of layer.

    Returns:
    --------
        no return

    """
    tensor_name = activations.op.name
    # tf.summary
    tf.summary.histogram(tensor_name + '/histogram', activations)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(activations))


def _weight_summary(weights, bias):
    """Helper to create summaries for weights and bias.

        Creates a summary that provides a histogram of weights and bias.

    Parameters:
    -----------
        weights : weights of layers.
        bias : bias of layers.

    Returns:
    --------
        no return
    """
    op_name_weights = weights.op.name
    op_name_bias = bias.op.name
    tf.summary.histogram(op_name_weights + '/histogram', weights)
    tf.summary.histogram(op_name_bias + '/histogram', bias)
```

### 网络结构

INPUT -> [CONV->RELU->POOL->LRN] -> [CONV->RELU->LRN->POOL] -> FC -> FC -> SOFTMAX

整体结构与CIDAR10相同，最简单的线性CNN结构，输入图像[128 * 128 * 3]，第一个卷积层卷积核大小是[3 * 3 * 3]，16个卷积核，采用0填充；池化层是最大池化，然后接了一个lrn（local response normalization, 局部响应归一化），这个层现在一般不用了，因为提升太小了；第二个卷积层卷积核[3 * 3 * 16]，16个卷积核，还是0填充；池化层和lrn都一样；全连接层都是128个神经元，SOFTMAX层出口是11类。

### 代码实现

如果网络很深，最好的方式是将网络 **模块化**，这样搭建深层网络就像搭积木一样，由于这次项目网络较浅，所以就没有实现，下一步的改进就是把网络模块化，这样网络结构会更清晰，便于debug，代码的复用性也更强。

```python
def inference(images, n_classes):
    """Build the model

    Parameters:
    -----------
        images : image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
        n_classes : number of classes, number of model output

    Returns:
    --------
        output tensor with the computed logits, float, [batch_size, n_classes], output of model
    """
    # 变量命名空间
    with tf.variable_scope('Conv1') as scope:
        # 定义第一个卷积层的卷积核
        # 初始化方式 ： truncated_normal_initializer
        # 官方推荐的卷积核初始化方式，CIFAR采用同样的方式
        kernal = tf.get_variable('kernal',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))

        bias = tf.get_variable('bias',
                                shape=[16],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        # 2d卷积，strides=1, padding="SAME"
        conv = tf.nn.conv2d(images, kernal, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, bias)
        # ReLU
        conv1 = tf.nn.relu(pre_activation, name="Conv_ReLU")
        _activation_summary(conv1)
        _weight_summary(kernal, bias)

    with tf.variable_scope('Pooling1_lrn') as scope:
        # Max pooling
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # lrn
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # Conv2 + ReLU [3 * 3 * 16, 16]
    with tf.variable_scope('Conv2') as scope:
        kernal = tf.get_variable('kernal',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        bias = tf.get_variable('bias',
                                shape=[16],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, kernal, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        # conv2 op name: conv2/Conv_ReLU
        conv2 = tf.nn.relu(pre_activation, name='Conv_ReLU')
        _activation_summary(conv2)
        _weight_summary(kernal, bias)

    # Norm2  + Max pool2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling2')

    # FC3
    with tf.variable_scope('local3') as scope:
        # 要求dim是确定值
        dim = int(pool2.get_shape()[1]) * int(pool2.get_shape()[2]) * int(pool2.get_shape()[3])
        reshaped_pool2 = tf.reshape(pool2, shape=[-1, dim])
        # 全连接层是128个神经元
        weights = tf.get_variable(name='weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.04))
        bias = tf.get_variable(name='bias',
                               shape=[128],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        # ReLU
        local3 = tf.nn.relu(tf.matmul(reshaped_pool2, weights) + bias, name=scope.name)
        # Summary
        _activation_summary(local3)
        _weight_summary(weights, bias)

    # FC4
    with tf.variable_scope('local4') as scope:
        weights = tf,get_variable(name='weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.04))
        bias = tf.get_variable(name='bias',
                               shape=[128],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        # ReLU
        local4 = tf.nn.relu(tf.matmul(local3, weights) + bias, name=scope.name)
        # Summary
        _activation_summary(local3)
        _weight_summary(weights, bias)

    # Output
    # 输出层应该是Softmax层，但输出值并没有进行Softmax转换
    # 在计算损失函数中将输出值进行Softmax转换为‘概率’，再计算对数似然损失
    with tf.variable_scope('Softmax_linear') as scope:
        weights = tf,get_variable(name='weights',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.04))
        bias = tf.get_variable(name='bias',
                               shape=[n_classes],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        # output = tf.matmul(local4, weights) + bias
        # 这两种方式精度相同，下面的这种方式可以对op命名
        output = tf.add(tf.matmul(local4, weights), bias, name=scope.name)
        # summary
        _activation_summary(output)
        _weight_summary(weights, bias)

        return output
```

## 损失函数（策略）

空有模型是不行的，还要定义损失函数，由于是多分类问题，且模型最后一层是Softmax变换，采用常用的 **对数似然损失函数**.

```python
def loss(logits, labels):
    """Loss function : computes softmax cross entropy between logits and labels.

    Parameters:
    -----------
        logits: output tensor of inference(), float, [batch_size, n_classes]
        labels: label tensor, tf.int32, one hot, [batch_size, n_classes]
        注意这里的labels是one hot后的.

    Returns:
    --------
        loss tensor of float type
    """
    with tf.variable_scope('Loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit,
                                                                labels=labels,
                                                                name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # summary
        tf.summary.scalar('loss', cross_entropy_mean)

    return cross_entropy_mean
```

要注意的是tf.nn.softmax_cross_entropy_with_logits()函数的使用，对logits和labels的要求，具体的可以查官方API.

## 优化器（算法）

最后一个要素就是算法，定义一个optimizer.

```python
def training(loss, lr):
    """Training op, the op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.
        Define a optimizer.

    Parameters:
    -----------
        loss : loss tensor, from loss()
        lr : initial learning rate

    Returns:
    --------
        train_op : operation for trainning
    """
    with name_scope('Optimizer') as scope:
        # 初始化一个AdamOptimizer类
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # 初始化global_step为0
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        tf.summary.scalar('learning_rate', lr)

    return train_op
```

## 准确率

模型评价指标。一般对于分类问题，loss并不能直观的反映出当前模型的优劣。计算模型的分类准确率直观的评价模型优劣。

```python
def accuracy(logits, labels):
    """Get accuracy of training

    Parameters:
    -----------
        logits: output tensor of model, float - [batch_size, num_classes].
        labels: labels tensor, int32 - [batch_size, num_classes], one hot

    Returns:
    --------
        accuracy of model 模型在输入batch_size上的平均分类准确率
    """
    with name_scope('Accuracy') as scope:
        # one hot to int
        labels = tf.argmax(labels, axis=1)
        top_one = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(top_one, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100
        tf.summary.scalar('accuracy', accuracy)

    return accuracy
```

有几个问题要注意：
1. 输入的labels是经过one hot变换的，要先转化为非one hot形式，具体做法就是利用tf.argmax()函数；
2. 利用tf.nn.in_top_k()函数判断batch_size中每个样本是否被正确分类，返回布尔型；
3. 计算batch_size的平均分类准确率。

## 总结

深度学习三要素（模型 策略 算法）再加上输入数据管线构成了TensorFlow中一个的Graph。

***

本作品采用 <a rel="license" href="http://creativecommons.org/licenses/by/3.0/cn/">知识共享署名 3.0 中国大陆许可协议</a> 进行许可。
