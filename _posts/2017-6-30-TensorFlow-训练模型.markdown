---
layout:     post
title:      "TensorFlow教程：在会话中训练模型"
date:       2017-6-30 15:30:00
author:     "liangyu"
header-img: "img/jpg.jpg"
tags:
    - Deep Learning
    - TensorFlow
---

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

* [前言](#前言)
* [准备工作](#准备工作)
* [构造graph](#构造graph)
* [在Session中运行graph](#在session中运行graph)
* [总结](#总结)

<!-- /code_chunk_output -->

## 前言

在TensorFlow中graph由op构成，前面已经定义了构建一个完整graph需要的op，现在将这些op有序组合起来构成graph，然后启动一个Session并运行graph。

在TensorFlow中，采用计算图的方式构建模型并训练。

> TensorFlow separates definition of computations from their execution
>
> 1. assemble a graph
> 2. use a session to execute operations in the graph.

## 准备工作

前面主要是定义一些文件路径：

```python
def training():
    """Traing model using train iamge and label, validation using val image and label."""
    # project dir
    project_dir = os.getcwd()
    # train data val data path
    train_data_path = os.path.join(project_dir, 'cache', 'train.tfrecords')
    val_data_path = os.path.join(project_dir, 'cache', 'val.tfrecords')

    # logs path
    logs_train_dir = os.path.join(project_dir, 'logs', 'train/')
    logs_val_dir = os.path.join(project_dir, 'logs', 'val/')
```

## 构造graph

将op联系起来构成完整的graph，代码中定义了两个 *placehoder* 主要是为了实现训练和验证同时进行。

注意：用定义的placeholder构造graph。

```python
with tf.Graph().as_default():
    # 利用read_and_decode()函数批量读取数据输入到graph中
    train_img_batch, train_label_batch = read_and_decode(train_data_path,
                                                         batch_size=BATCH_SIZE,
                                                         one_hot=ONE_HOT)
    val_img_batch, val_label_batch = read_and_decode(val_data_path,
                                                     batch_size=N_VAL,
                                                     one_hot=ONE_HOT)
    # placeholder
    image = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3], name='image')
    label_ = tf.placeholder(tf.int32, shape=[None, N_CLASSES], name='label')

    # model
    logits = frame_model.inference(image,                     
                                   N_CLASSES,
                                   visualize)
    # loss function
    loss = frame_model.losses(train_logits, label_)
    # optimizer
    optimizer = frame_model.trainning(train_loss, learning_rate)
    # evaluation
    accuracy = frame_model.evaluation(train_logits, label_)

    # init op
    init_op = tf.global_variables_initializer()

    # merge summary
    summary_op = tf.summary.merge_all()
```

## 在Session中运行graph

启动一个Session并运行graph，其中每100步打印训练信息；每500步验证模型，并运行summary_op，并写入disk，需要注意的是运行summary_op时，也要通过feed_dict供给数据；每2000步保存模型。

```python
# initial a Session()
with tf.Session() as sess:
    # initial tf.train.Saver() class
    saver = tf.train.Saver()
    # run init op
    sess.run(init_op)
    # start input enqueue threads to read data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # initial tf.summary.FileWriter() class
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

    try:
        for step in xrange(MAX_STEP):
            if coord.should_stop():
                    break

            # get train batch data
            train_img, train_label = sess.run([train_img_batch, train_label_batch])

            start_time = time.time()
            # run ops
            _, tra_loss, tra_accuracy = sess.run([optimizer, loss, accuracy],
                                                 feed_dict={image: train_img, label_: train_label})

            duration = time.time() - start_time

            # print info of training
            if step % 100 == 0 or (step + 1) == MAX_STEP:
                sec_per_batch = float(duration)    # training time of a batch
                print('Step {}, train loss = {:.2f}, train accuracy = {:.2f}%, sec_per_batch = {:.2f}s'.format(step,
                                                                                                               tra_loss,
                                                                                                               tra_accuracy,
                                                                                                               sec_per_batch))
            # run summary_op时也要用feed_dict供给数据
            if step % 500 == 0 or (step + 1) == MAX_STEP:
                # run summary op and write train summary to disk
                summary_str = sess.run(summary_op,
                                       feed_dict={image: train_img, label_: train_label})
                train_writer.add_summary(summary_str, step)

                # get val data batch
                val_img, val_label = sess.run([val_img_batch, val_label_batch])

                # run ops
                val_loss, val_acc, summary_str = sess.run([loss, accuracy, summary_op],
                                                          feed_dict={image: val_img, label_:val_label})
                print('*** Step {}, val loss = {:.2f}, val accuracy = {:.2f}% ***'.format(step, val_loss, val_acc))

                # run summary op and write val summary to disk
                val_writer.add_summary(summary_str, step)

            # save model
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    # 捕捉异常
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
```

## 总结

思路很清晰，但是真正上手编还是会出现各种意想不到的问题，在TensorFlow中debug也不方便，感觉结果不正常首先要看代码实现是否正确，确保代码正确后再去考虑是否有其他错误。

***

本作品采用 <a rel="license" href="http://creativecommons.org/licenses/by/3.0/cn/">知识共享署名 3.0 中国大陆许可协议</a> 进行许可。
