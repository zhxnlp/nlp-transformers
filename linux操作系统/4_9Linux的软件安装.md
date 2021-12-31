# Python 全栈文档

## 第十章  Linux的软件安装

#### 1、rpm软件安装包

RPM（RedHat Package Manager）安装管理 

​    这个机制最早是由Red Hat开发出来,后来实在很好用,因此很多 distributions（发行版）就使用这个机制来作为软件安装的管理方式 。包括Fedora,CentOS,SuSE等等知名的开发商。 

<font color='red'> rpm -qa|grep python(查看所有安装包里的python)</font>

RPM的优点 

1. RPM内含已经编译过的程序与配置文件等数据,可以让用户免除重 新编译的困扰 

2. RPM在被安装之前,会先检查系统的硬盘容量、操作系统版本等,可 避免文件被错误安装 

3. RPM文件本身提供软件版本信息、相依属性软件名称、软件用途说明、软件所含文件等信息,便于了解软件 

4. RPM管理的方式使用数据库记录 RPM 文件的相关参数,便于升级 、移除、查询与验证 

RPM的缺点：

​	1、rpm在安装的时候不能指定安装路径。安装路径是在制作RPM包的时候已经指定了。

​	2、rpm软件包一般都存在依赖问题没有解决。（rpm安装时依赖的软件没安装就会失败）

安装一个依赖另一个，一堆下来很麻烦，所以现在一般不用rpm安装

rpm安装库：https://mirrors.aliyun.com/centos/7.8.2003/os/x86_64/Packages/

gcc：https://mirrors.aliyun.com/centos/7.8.2003/os/x86_64/Packages/gcc-4.8.5-39.el7.x86_64.rpm

rpm安装

  rpm -ivh package_name 

选项与参数: 

-i :install的意思 

-v :察看更细部的安装信息画面 

-h :以安装信息列显示安装进度

Ø 安装单个rpm包 

   rpm -ivh package_name 

Ø 安装多个rpm包 

   rpm -ivh a.i386.rpm b.i386.rpm *.rpm 

Ø 安装网上某个位置rpm包 

   rpm -ivh http://website.name/path/pkgname.rpm

 **rpm查询：**

简单原理：rpm在查询的时候,其实查询的地方是在/var/lib/rpm/ 这个目录下的数据库文件 



rpm查询已安装软件,选项与参数：

-q :仅查询,后面接的软件名称是否有安装 

-qa :列出所有的,已经安装在本机Linux系统上面的所有软件名称 ！！！

-qi :列出该软件的详细信息,包含开发商、版本和说明等 !!

-ql :列出该软件所有的文件与目录所在完整文件名 !!

-qc :列出该软件的所有配置文件 !

-qd :列出该软件的所有说明文件 

-qR :列出和该软件有关的相依软件所含的文件 

-qf :由后面接的文件名,找出该文件属于哪一个已安装的软件 

#### 2、yum命令（解决依赖）

​        yum是一个在Fedora和RedHat以及SUSE中的Shell前端软件包管理器

查找软件包是否存在yum search pakeage

```
基於RPM包管理，能够从指定的服务器自动下载RPM包并且安装，可以自动处理依赖性关系，并且一次安装所有依赖的软体包，无须繁琐地一次次下载、安装。

yum提供了查找、安装、删除某一个、一组甚至全部软件包的命令，而且命令简洁而又好记
```

​        **语法**：

```
yum [options] [command] [package ...]
```

​        **常用参数**：

| 参数        | 参数描述                                                     |
| ----------- | ------------------------------------------------------------ |
| **options** | 可选，选项包括-h（帮助），-y（当安装过程提示选择全部为"yes"），-q（不显示安装的过程）等等 |
| **command** | 要进行的操作                                                 |
| **package** | 操作的对象                                                   |

​        **常用命令**：

1.列出所有可更新的软件清单命令：yum check-update
2.更新所有软件命令：yum update
3.<font color='red'>仅安装指定的软件命令：yum install <package_name></font>
4.<font color='red'>仅更新指定的软件命令：yum update <package_name></font>
5.列出所有可安裝的软件清单命令：yum list
6.<font color='red'>删除软件包命令：yum remove <package_name></font>（不会卸载依赖，因为其它软件可能也有依赖）
7.<font color='red'>查找软件包 命令：yum search <package_name></font>
   查看安装的语言：yum repolist，语言在/etc/yum.repos.d/
CentOS-Base.repo  CentOS-Debuginfo.repo  CentOS-Media.repo     CentOS-SCLo-scl-rh.repo  CentOS-Vault.repo
CentOS-CR.repo    CentOS-fasttrack.repo  CentOS-SCLo-scl.repo  CentOS-Sources.repo      CentOS-x86_64-kernel.repo
编辑第一个vim CentOS-Base.repo，13-18是base，21-26是updates，29-是extras。
8.清除缓存命令:
     yum clean packages: 清除缓存目录下的软件包
     yum clean headers: 清除缓存目录下的 headers
     yum clean oldheaders: 清除缓存目录下旧的 headers
     yum clean, yum clean all (= yum clean packages; yum clean oldheaders) :清除缓存目录下的软件包及旧的headers

<font color='red'> 9.查看软件包信息 yum info package </font>

可以看到python 3.6.8只有几十k，这是因为我们自己下载的安装包含有大量依赖，而这里的安装包只有核心代码，没有包含依赖。yum会自己解决依赖问题。

yum默认安装的是国外的rpm软件官网，可以自己改成国内的网站下载。阿里云镜像网站有详细介绍。选centos7后的第二个curl的网址复制。阿里云的yum语言有好几个站点，除了centos还有epel（企业开发软件包）

pip install XXX 追加 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

tsinghua
python3 -m pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple

douban
python3 -m pip install xxx -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com





<font color='red'>安装make命令：yum install gcc automake autoconf libtool make</font>

安装g++ :yum install gcc-c++

安装pip：yum -y install python3-pip

Setup script exited with error: command 'gcc' failed with exit status 1

由于没有正确安装Python开发环境导致。

### Centos/Fedora

sudo yum install python3-devel

sudo yum install libevent-devel

easy_install gevent`
或者
`pip3 install gevent
把环境更新下
`sudo yum install groupinstall 'development tools'

fasttext需要安装numpy scipy和pybind11，git，wget

安装sud pip3 install pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 

安装gcc8.3并设置为默认的gcc

scl enable devtoolset-8 bash

sudo python3 -m pip install --upgrade --force pip升级pip

pip3 install --upgrade setuptools升级setuptools

安装指定版本包pip install numpy==1.2.3





tar (child): bzip2：无法 exec: 没有那个文件或目录
tar (child): Error is not recoverable: exiting now
tar: Child returned status 2
tar: Error is not recoverable: exiting now

原因：缺少bzip2包

```
yum install -y bzip2
```
