# Python 全栈文档

## 第三章 Linux文件与路径

#### 1、文件结构

​        Windows和Linux文件系统区别

​        在windows平台下，打开“此电脑”，我们可以看到盘符分区

![](imgs/04_107.png)

​        每个驱动器都有自己的根目录结构，这样形成了多个树并列的情形

​        但是在 Linux 下，我们是看不到这些驱动器盘符，我们看到的是文件夹（目录）：

![](imgs/04_108.png)

​        Centos没有盘符这个概念，只有一个根目录/，所有文件都在它下面

```bash
[root@localhost ~]# ls /
bin  boot  dev  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var

```

​        当我们输入ls /  可以查看更目录下的文件查看根目录下的系统文件

| 目录      | 说明                           | 备注                                                         |
| --------- | ------------------------------ | ------------------------------------------------------------ |
| bin       | 存放普通用户可执行的指令       | 即使在单用户模式下也能够执行处理                             |
| boot      | 开机引导目录                   | 包括Linux内核文件与开机所需要的文件                          |
| dev       | 设备目录                       | 所有的硬件设备及周边均放置在这个设备目录中                   |
| **etc**   | **各种配置文件目录**           | 大部分配置属性均存放在这里                                   |
| lib/lib64 | 开机时常用的动态链接库         | bin及sbin指令也会调用对应的lib库                             |
| media     | 可移除设备挂载目录             | 类似软盘 U盘 光盘等临时挂放目录                              |
| mnt       | 用户临时挂载其他的文件系统     | 额外的设备可挂载在这里,相对临时而言                          |
| opt       | 第三方软件安装目录             | 现在习惯性的放置在/usr/local中                               |
| proc      | 虚拟文件系统                   | 通常是内存中的映射,特别注意在误删除数据文件后，比如DB，只要系统不重启,还是有很大几率能将数据找回来 |
| root      | 系统管理员家目录               | 除root之外,其他用户均放置在/home目录下                       |
| run       | 系统运行时所需文件             | 以前防止在/var/run中,后来拆分成独立的/run目录。重启后重新生成对应的目录数据 |
| sbin      | 只有root才能运行的管理指令     | 跟bin类似,但只属于root管理员                                 |
| srv       | 服务启动后需要访问的数据目录   |                                                              |
| sys       | 跟proc一样虚拟文件系统         | 记录核心系统硬件信息                                         |
| tmp       | 存放临时文件目录               | 所有用户对该目录均可读写                                     |
| usr       | 应用程序放置目录               |                                                              |
| var       | 存放系统执行过程经常改变的文件 |                                                              |

​        在 Linux 系统中，有几个目录是比较重要的，平时需要注意不要误删除或者随意更改内部文件。

​        **/etc**： 上边也提到了，这个是系统中的配置文件，如果更改该目录下的某个文件可能会导致系统不能启动。

​        **/bin, /sbin, /usr/bin, /usr/sbin**: 这是系统预设的执行文件的放置目录，比如 ls 就是在/bin/ls 目录下的。

​        值得提出的是，/bin, /usr/bin 是给系统用户使用的指令（除root外的通用户），而/sbin, /usr/sbin 则是给root使用的指令。（所有绿色显示的都是命令）

​        **/var**： 这是一个非常重要的目录，系统上跑了很多程序，那么每个**程序都会有相应的日志产生**，而这些日志就被记录到这个目录下，具体在/var/log 目录下，另外mail的预设放置也是在这里。启动日志是boot.log。输入cd /var/log回车输入more boot.log查看启动日志。

#### 2、基本概念

​        **用户目录：位于 /home/user，称之为用户工作目录或家目录,表示方式：**

```bash
# 在home有一个user  这里就是之前创建的msb123用户
[root@localhost ~]# cd /home
[root@localhost home]# ls
user
# 使用~回到root目录，使用/是回到根目录下
[root@localhost ~]# cd ~当前用户的家目录，相当于cd /root
[root@localhost user]# cd~user 表示切换到user的家目录，即/home/user
```

​        **登录信息**

```python
[root@localhost /]#  /代表根目录，如果是~代表家目录
Linux的bash解析器终端用来显示主机名和当前用户的标识；
# root表示bai当前用户叫root（系统管理员账户）
# localhost表示当前使用的主机名叫localhost（没有设置系bai统名字的时候默认名称是localhost）
# / 表示你当前所处的目录位置 (这里的'/'表示你当前在根目录下)
```

​        **相对路径和绝对路径**

​        **绝对路径**

​                从/目录开始描述的路径为绝对路径，如：

```bash
[root@localhost /]# cd /home/user
[root@localhost /]#su user切换到user目录，即user的家目录，user拥有很高的权限。企业里每个人都是把私人文件放在自己的家目录里。超级管理员家目录是/root。切换到root账户输入su root。
[root@localhost /]# ls /usr
```

​        **相对路径**

​                从当前位置开始描述的路径为相对路径，如：

```bash
[root@localhost /]# cd ../postfix 表示由当前目录/var/spool/mail去到/var/spool/postfix目录
[root@localhost /]# ls abc/def
```

​        **. 和 ..**

​        每个目录下都有**.**和**..**

```
./ 表示当前目录
../ 表示上一级目录，即父目录
```

​        例如这里切换路径时候

```python
# 从 / 根目录切换到 home目录
[root@localhost /]# cd home
# 确认路径/home
[root@localhost home]# pwd
/home
# 切换到当前目录cd .  目录无变化
[root@localhost home]# cd .还是当前目录
# 切换到当前目录cd ..  上一级目录
[root@localhost home]# cd ..回到上一级根目录,
                          ../目录名   就表示父目录下的子目录，其实就是同一父目录下其它目录，相对路径
[root@localhost /]# cd-表示回到刚刚那个目录


```

（注意  **根目录下的  .  和  ..  都表示当前目录**）

​    **文件权限**

​        文件权限就是文件的访问控制权限，即哪些用户和组群可以访问文件以及可以执行什么样的操作。

​        Unix/Linux系统是一个典型的多用户系统，不同的用户处于不同的地位，对文件和目录有不同的访问权限。为了保护系统的安全性Unix/Linux系统除了对用户权限作了严格的界定外，还在用户身份认证、访问控制、传输安全、文件读写权限等方面作了周密的控制。

​        <font color='orange'>在 Unix/Linux中的每一个文件或目录都包含有访问权限</font>，这些访问权限决定了谁能访问和如何访问这些文件和目录。

​    **访问用户**

​        通过设定权限可以从以下三种访问方式限制访问权限：

```
- <font color='red'>只允许用户自己访问（所有者）</ 所有者就是创建文件的用户，用户是所有用户所创建文件的所有者，用户可以允许所在的用户组能访问用户的文件。
- <font color='red'>允许一个预先指定的用户组中的用户访问（用户组）</ 用户都组合成用户组，例如，某一类或某一项目中的所有用户都能够被系统管理员归为一个用户组，一个用户能够授予所在用户组的其他成员的文件访问权限。
- <font color='red'>允许系统中的任何用户访问（其他用户）</ 用户也将自己的文件向系统内的所有用户开放，在这种情况下，系统内的所有用户都能够访问用户的目录或文件。在这种意义上，系统内的其他所有用户就是 other 用户类
```

​    **访问权限**

​        用户能够控制一个给定的文件或目 录的访问程度，一个文件或目录可能有读、写及执行权限：

```
- ​        读权限（r） 对文件而言，具有读取文件内容的权限；对目录来说，具有浏览目录的权限。
- ​        写权限（w） 对文件而言，具有新增、修改文件内容的权限；对目录来说，具有删除、移动目录内文件的权限。
- ​        可执行权限（x） 对文件而言，具有执行文件的权限；对目录了来说该用户具有进入目录的权限。
```

​        注意：通常，Unix/Linux系统只允许文件的属主(所有者)或超级用户改变文件的读写权限。

```bash
[root@localhost /]# ls -l（ls后面有个空格，查看文件列表）
总用量 20
lrwxrwxrwx.   1 root root    7 8月  31 15:48 bin -> usr/bin
dr-xr-xr-x.   5 root root 4096 8月  31 15:58 boot
...

```

​        我们来拆解结构，这里面我只列了根目录下的一部分内容

​        用到<font color='red'> ls -l 命令查看当前文件夹下详细信息</font>，具体的命令和参数，后面会深入讲解

lrwxrwxrwx.   1 root root    7 5月  22 07:59 bin -> usr/bin
dr-xr-xr-x.   5 root root 4096 5月  22 08:09 boot
dr-xr-xr-x. 133 root root    0 5月  22 13:41 proc
dr-xr-x---.   2 root root  135 5月  22 08:09 root（root目录对于root用户和root组都只能访问和读入，不可改写）

​        我们需要关注的是文件或目录的权限情况（第一位是类型，后面9位分成三组，分别表示所有者、用户组和其他用户）

```bash
l  rwx  rwx  rwx（所有者、用户组、其他用户）
d  r-x  r-x  r-x

# 首先第一个字母 在Linux中第一个字符代表这个文件是目录、文件或链接文件等等。
[ d ] 表示目录
[ l ] 表示为链接文档(link file，即快捷方式，颜色不一样)
[ - ] 表示为文件
[ b ] 表示为装置文件里面的可供储存的接口设备(可随机存取装置)
[ c ] 表示为装置文件里面的串行端口设备，例如键盘、鼠标(一次性读取装置)

# 其次接下来的字符中，以三个为一组，且均为 [ rwx ] 的三个参数的组合
[ r ]代表可读(read)
[ w ]代表可写(write)，可以用mv编辑，如mv dev/abc/修改dev目录名为abc
[ x ]代表可执行(execute)对于目录就是访问
[ - ]

# 要注意的是，这三个权限的位置不会改变，如果没有权限，就会出现减号[ - ]而已。

  此时问题来了那么这三组一样是有什么区分尼？
# 这里就涉及到刚才所描述的访问用户权限
# 所有者    所有者表示该文件的所有者（属主）
# 用户组    表示当前用户再同一组（属组）
# 其他用户  允许系统中的任何用户访问，系统内的其他所有用户就是 other 用户类
```

​    **文件属主与属组**

​        对于文件来说，它都有一个特定的所有者，也就是对该文件具有所有权的用户。

​        同时，在Linux系统中，用户是按组分类的，一个用户属于一个或多个组。

​        文件所有者以外的用户又可以分为文件所有者的同组用户和其他用户。

​        因此，Linux系统按文件所有者、文件所有者同组用户和其他用户来规定了不同的文件访问权限。

​       <font color='red'> more etc/passwd打开passwd文件可以查看所有的用户 </font>

root:x:0:0:root:/root:/bin/bash  root用户的id即uid为0，用户组id即gid为0

user:x:1000:1000:user:/home/user:/bin/bash user用户编号100，组编号1000

​       <font color='red'>more etc/group打开group文件查看所有的组</font>

root:x:0:root用户编号0

user:x:1000:user用户编号1000

```bash
[root@localhost /]# ls -l
总用量 20
...
dr-xr-xr-x.   5 root root 4096 8月  31 15:58 boot
...

[root@localhost /]# cd /home
[root@localhost home]# ls -l
总用量 0
drwx------. 2 msb123 msb123 83 9月   2 15:54 msb123


# 在以上实例中，msb123 文件是一个目录文件，属主和属组都为 msb123，属主有可读、可写、可执行的权限；与属主同组的用户无权限读写执行；其他用户也无权限读写执行

# 对于 root 用户来说，一般情况下，文件的权限对其不起作用。
```

#### 3、基本命令信息

​        熟悉一些入门的命令

​        1、ls

```bash
ls 命令
作用：Linux ls命令列出目前工作目录所含之文件及子目录，就是list的意思
语法： ls   [-alrtAFR](可接多个选项)    [name...](参数必须是文件)（ls后面一定要空格）
参数：蓝色为目录

-a 显示所有文件及目录 (l文件名或目录名开头为"."的视为隐藏档，不会列出) 示例如下：
[root@localhost ~]# ls -a
.  ..  anaconda-ks.cfg  .bash_history  .bash_logout  .bash_profile  .bashrc  .cshrc  

-l 除文件名称外，亦将文件型态、权限、拥有者、文件大小等资讯详细列出  示例如下：
[root@localhost ~]# ls -l#详细列出，称为长格式
总用量 4
-rw-------. 1 root root 1437 8月  31 15:54 anaconda-ks.cfg

-r 将文件以相反次序显示(原定依英文字母次序) 示例如下：
[root@localhost ~]# ls -ra
.tcshrc  .cshrc  .bashrc  .bash_profile  .bash_logout  .bash_history  anaconda-ks.cfg  ..  .

-t 将文件依建立时间之先后次序列出   示例如下：
[root@localhost ~]# ls -lt
总用量 4
-rw-------. 1 root root 1437 8月  31 15:54 anaconda-ks.cfg

-A 同 -a ，但不列出 "." (目前目录) 及 ".." (父目录)   示例如下：
[root@localhost ~]# ls -A
anaconda-ks.cfg  .bash_history  .bash_logout  .bash_profile  .bashrc  .cshrc  .tcshrc

-F 在列出的文件名称后加一符号；例如可执行档则加 "*", 目录则加 "/"   示例如下：
[root@localhost ~]# ls -F /home
msb123/

-R 若目录下有文件，则以下之文件亦皆依序列出  示例如下：
[root@localhost ~]# ls -R /home
/home:
msb123
/home/msb123:

```

```bash
常用组合
[1]查看文件详情：ls -l 或 ll
[2]增强对文件大小易读性，以人类可读的形式显示文件大小： ls -lh
[3]对文件或者目录进行从大到小的排序： ls -lhs
[4]查看当前目录下的所有文件或者目录，包括隐藏文件： ls -la等同于ls -al或者ls -a -l（两空格）
[5]只查看当前目录下的目录文件： ls -d .
[6]按照时间顺序查看，从上到倒下时间越来越近： ls -ltr
[7]查看文件在对应的inode信息：ls -li
ls -f /表示所有目录后面加/
```

​        2、cd

```bash
cd 命令
作用：变换当前目录到dir。默认目录为home，可以使用绝对路径、或相对路径。
语法：cd [dir](路径)

# 跳到用户目录下
[root@localhost ~]# cd /home/msb123
[root@localhost msb123]# 
 
# 回到home目录
[root@localhost msb123]# cd ~当前用户家目录
[root@localhost ~]# 

# 跳到上次所在目录
[root@localhost ~]# cd -
/home/msb123
[root@localhost msb123]#

# 跳到父目录(也可以直接使用 cd ..)
[root@localhost msb123]# cd ./..
[root@localhost home]# 
#cd 目录名 切换到子目录

# 再次跳到上次所在目录
[root@localhost home]# cd -
/home/msb123
[root@localhost msb123]# 

# 跳到当前目录的上两层
[root@localhost msb123]# cd ../..
[root@localhost /]#
#cd ./切换到当前目录

# 把上个命令的最后参数作为dir
这里我们先将文件夹cd 到python2.7路径
[root@localhost /]# cd /usr/include/python2.7/
[root@localhost python2.7]#

# 这里使用cd ./..参数作为引子
[root@localhost python2.7]# cd ./..

# 这里我们使用命令，重复最后一个命令参数，直到回到了根目录
[root@localhost include]# cd !$
cd ./..
[root@localhost usr]# cd ./..
[root@localhost /]# 

```

3.<font color='red'> pushd命令创建目录栈 </font>

​      1.比如现在在etc/sysconfig目录下，想切换到var目录，可以输入 <font color='red'> pushd /var/lock </font>直接切换过去了，且创建了一个目录栈 /var/lock（在顶部）   /etc/sysconfig。此时直接输入pushd就可以在最顶端两个目录之间切换。目录栈顶端永远是当前目录。 每次 pushd 命令执行完成之后，默认都会执行一个 dirs 命令来显示目录栈的内容 

​     2.dirs显示目录栈。选项 -p 每行显示一条记录

​                                           -v 每行显示一条记录，同时展示该记录在栈中的index

​                                           -c 清空目录栈

​    其中，`-p` 与 `-v` 选项的区别是，`-v` 选项将显示每条记录在栈中的索引，除此之外完全一样

0  /etc/rc0.d
 1  /var/lock
 2  /etc/sysconfig
 3  /var/run

此时<font color='red'>输入pushd+n或者pushd-n可以快速任意切换目录栈</font>，+表从上往下查找，-为从下往上查找。

​      3.输入<font color='red'>popd /目录名</font>可以在目录栈中删除此目录。

4、pwd

```bash
pwd  命令

作用：可立刻得知目前所在的工作目录的绝对路径名称
语法：pwd [--help][--version]

参数说明:
--help 在线帮助。
--version 显示版本信息。
查看当前所在目录：
[root@localhost /]# cd /home
[root@localhost home]# pwd
/home
[root@localhost home]# 


```

5.其它用法

命令 --help 查看命令的全部参数和用法

命令 --version查看命令版本

shift+g跳到末尾，pageup切换到上一个命令

比如一个文件名为mashibin.txt ，操作的时候可以只输入ma再按Tab键，就直接键入mashibin.txt。快速操作