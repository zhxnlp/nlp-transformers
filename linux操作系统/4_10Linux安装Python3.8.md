# Python 全栈文档

## 第十一章  Linux安装Python

首先我们在安装Tomcat前需要先配置好jdk

#### 1、下载Python官方源码文件

​     在python官网downloads-all releases-3.8.6版本底部 Source release 源代码版本的 [Gzipped source tarball](https://www.python.org/ftp/python/3.8.6/Python-3.8.6.tgz) 文件下载后上传到linux（用xftp）

#### 2、安装Python3.8

（1）安装依赖包

```bash
yum install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc make libffi-devel -y
```

（2）解压安装

​                  安装到目录：/usr/local/python-3.8

```bash
# 解压压缩包
tar -zxvf Python-3.8.1.tgz  

# 进入文件夹
cd Python-3.8.1

# 配置安装位置
./configure prefix=/usr/local/python-3.8#预编译，prefix指定安装目录
检查python3.8能否安装编译
# 安装
make && make install

```

​        （3）查看

```bash
如果最后没提示出错，就代表正确安装了，在/usr/local/目录下就会有python-3.8目录
ls /usr/local/python-3.8
```

​        （4）添加软连接

```bash
#添加python3的软链接 
ln -s /usr/local/python-3.8/bin/python3.8 /usr/bin/python3 （usr/bin就是PATH环境）
#添加 pip3 的软链接 
ln -s /usr/local/python-3.8/bin/pip3.8 /usr/bin/pip3
```

​        （5）好了，我们来测试一下python3

```bash
[root@localhost ~]# python3
Python 3.8.1 (default, Feb  4 2020, 11:28:31) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-39)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 

```

​        
