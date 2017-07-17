---
layout:     post
title:      "从零搭建 Jekyll + Github Pages 个人博客"
date:       2017-1-21 21:30:00
author:     "liangyu"
header-img: "img/park-289087_1920.jpg"
tags:
    - 折腾
---

# 前言

原来搭建个人博客的时候并没有一步一步记录下来，前几天电脑坏了，重装了系统，导致很多环境要重新搭，这次就把搭建 Jekyll + Githun pages 环境的过程一步一步记录下来，省的以后再重装系统再踩一遍坑.

# 正文

## 操作系统

操作系统：Linux Mint 17.3, 基于 Ubuntu 14.04.

很多开发环境我在 Windows 和 Linux 下都搭建过，还是在 **Linux** 下更方便，很多情况下一条命令行搞定，而且现在 Linux 下的桌面桌面操作系统也已经很成熟了，包括鼎鼎大名的 **Ubuntu**，还有常年在 Linux 发行版中排名第一的 **Linux Mint**，当然国产的 **Deepin** 做的也不错，但是我不太习惯它那浓浓的山寨风（纯属个人看法）。

Jekyll 官方也不推荐在 Windows 下配置，当然也有办法，就是坑比较多，比较麻烦。

## Ruby 相关环境

Jekyll 是基于 Ruby 的，所以要先安装R语言环境。

最简单的是利用系统自带的包管理系统，但是 Ruby 官方不推荐这种方式，通过包管理系统下载的都是很老的版本，存在兼容性方面的问题。

推荐使用 **第三方安装工具** 安装。RVM 是一个命令行工具，可以提供一个便捷的多版本 Ruby 环境的管理和切换。

安装 RVM:

``` bash
$ gpg --keyserver hkp://keys.gnupg.net --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3
$ \curl -sSL https://get.rvm.io | bash -s stable
$ source ~/.bashrc
$ source ~/.bash_profile
```

修改 RVM 的 Ruby 安装源到 Ruby China 的 Ruby 镜像服务器，这样能提高安装速度

``` bash
$ echo "ruby_url=https://cache.ruby-china.org/pub/ruby" > ~/.rvm/user/db
```

列出已知的 Ruby 版本

```bash
$ rvm list known
```

安装一个 Ruby 版本

```bash
$ rvm install 2.2.0 --disable-binary
```

切换 Ruby 版本

```bash
$ rvm use 2.2.0
```

切换版本时如果报错，执行下这个命令

```bash
$ echo '[[ -s "$HOME/.rvm/scripts/rvm" ]] && . "$HOME/.rvm/scripts/rvm"' >>~/.bashrc
$ source ~/.bashrc
```
设置为默认版本，这样一来以后新打开的控制台默认的 Ruby 就是这个版本

```bash
$ rvm use 2.2.0 --default
```

安装 Ruby 下的包管理工具 RubyGems（类似于 Python 下的 pip）

在 [官方网站](https://rubygems.org/pages/download) 下载压缩包，解压后：

``` bash
$ cd rubygems-2.6.12
$ sudo ruby setup.rb
```

至此 Ruby 相关环境就安装好了，在 Linux 下安装还是很简单的，需要注意的一点就是 **不要用自带的包管理器安装 Ruby**，如果系统自带了 Ruby，用 RVM 安装新版本并设置为默认。

## 安装 Jekyll

由于众所周知的原因，下载 Jekyll 很慢，可以切换到国内源：

``` bash
$ gem sources --add https://gems.ruby-china.org/ --remove https://rubygems.org/
$ gem sources -l
https://gems.ruby-china.org
# 确保只有 gems.ruby-china.org
```

准备工作做好了，终于可以安装 Jekyll 了：

``` bash
$ gem install jekyll
```

RubyGems 会把相关依赖环境全部安装上。至此 Jekyll 就安装完成了，使用请参照官方文档，英语比较渣的可以参考 [中文网站](http://jekyll.com.cn/docs/) 。

## Github Pages

Github 为我们提供了一个免费搭建个人博客的机会，详细的可以参考 [官方网站](https://pages.github.com/)，和新建一个仓库流程差不多，默认你已经使用过 Github 和 git，并且有 Github 的账号（毕竟 Github 搬运工）。

新建好仓库后就可以 git clone 和本地相关联，看到好的开源模板也可以拿来用，写好博客后还可以本地调试，实时在浏览器中看到自己的修改：

``` bash
$ jekyll serve --watch
```

注意：要在 Jekyll 3.0 及以上版本实现实时预览功能要安装 jekyll-paginate：
``` bash
$ gem install jekyll-paginate
```

修改好后，就可以推送到远程，前一部分相当于线下调试，这部分就相当于上线。

## 后记

写博客还真是费时间啊，半天时间又过去了，文字功底太渣了。

其实我还挺喜欢前端的，可能是因为接触的不深，但我喜欢这种做了点东西就能立刻看到反馈的感觉。

***

本作品采用 <a rel="license" href="http://creativecommons.org/licenses/by/3.0/cn/">知识共享署名 3.0 中国大陆许可协议</a> 进行许可。
