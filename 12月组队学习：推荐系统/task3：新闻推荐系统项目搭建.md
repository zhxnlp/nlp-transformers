@[toc]
## 0	项目运行环境
0.1	获取软件安装包
软件安装包地址：https://share.weiyun.com/u3ZIjZfg 

0.2	使用软件版本
操作系统：Windows10
MySQL：8.0.25
Redis：5.0.14
Mongodb：5.0.5
Mini-Conda Python 3.8
Node.js：16.13.1
前端IDE：WebStorm 2021.1
 
后端IDE：PyCharm Professional 2021.1
 
访问MySQL和Mongodb的数据库工具：DataGrip 2021.1
 
访问Redis的工具：redis-desktop-manager-0.9.9.99.exe
 
## 1	项目下载与IDE导入
项目地址：
https://github.com/datawhalechina/fun-rec 
1.1	前端项目导入
使用WebStrom IDE工具，导入前端项目
 
1.2	后端项目导入
使用PyCharm IDE工具，导入后端项目
 
## 2	数据库安装与使用（Windows10）
### 2.1	MySQL数据库安装与使用
卸载mysql：[帖子1](https://blog.csdn.net/dh12313012/article/details/87274385?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163976655316780269838594%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=163976655316780269838594&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-87274385.pc_search_insert_es_download_v2&utm_term=centos%E5%AE%89%E8%A3%85%E5%8D%B8%E8%BD%BDmysql&spm=1018.2226.3001.4187)、[帖子2](https://blog.csdn.net/weixin_44443884/article/details/106231811?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163976655316780269838594%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=163976655316780269838594&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-106231811.pc_search_insert_es_download_v2&utm_term=centos%E5%AE%89%E8%A3%85%E5%8D%B8%E8%BD%BDmysql&spm=1018.2226.3001.4187)
参考胡瑞峰文档和帖子[《Windows环境安装 安装mysql-8.0.18-winx64详细图解(zip包版本)》](https://blog.csdn.net/hu10131013/article/details/107711192?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163957716016780269846658%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163957716016780269846658&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-107711192.pc_search_insert_es_download_v2&utm_term=windows%E5%AE%89%E8%A3%85mysql&spm=1018.2226.3001.4187)

2.0 centos装的mysql8无法启动，运行`service mysql start`显示这个命令找不到

```python
#不装这个下面安装报错，缺少安装包
wget http://www.percona.com/redir/downloads/Percona-XtraDB-Cluster/5.5.37-25.10/RPM/rhel6/x86_64/Percona-XtraDB-Cluster-shared-55-5.5.37-25.10.756.el6.x86_64.rpm
rpm -ivh Percona-XtraDB-Cluster-shared-55-5.5.37-25.10.756.el6.x86_64.rpm

yum install -y mariadb-server
```

#### 2.1.1	MySQL数据库安装
（1）安装包下载
下载地址：https://dev.mysql.com/downloads/mysql/ 
安装包版本：8.0.25

（2）配置环境变量
变量名：MYSQL_HOME
变量值：D:\mysql-8.0.25-winx64
在桌面上点击"此电脑–右击–选择属性–选择高级–环境变量"，上方点新建系统变量

 ![在这里插入图片描述](https://img-blog.csdnimg.cn/70bb500defa44234bacf8095ed29e9e7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

在下方环境变量PATH添加：%MYSQL_HOME%\bin
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2ed4bd1facab45aaa5f5f31c4bfa10bb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)


（3）生成data文件
在解压后mysql-8.0.18-winx64的文件下创建my.ini配置文件
具体内容如下
将下面的内容复制到刚创建的文件中 ，主要需要修改的字段为basedir和datadir
basedir=自己的mysql目录
datadir=mysql的data存储的目录

```python
[mysqld]
# 设置3306端口
port=3306
# 设置mysql的安装目录
basedir=D:/Java/Database/mysql-8.0.18-winx64
# 设置mysql数据库的数据的存放目录 (data文件夹如果没有的话会自动创建)
datadir=D:/Java/Database/mysql-8.0.18-winx64/data
# 允许最大连接数
max_connections=200
# 允许连接失败的次数。这是为了防止有人从该主机试图攻击数据库系统
max_connect_errors=10
# 服务端使用的字符集默认为UTF8
character-set-server=utf8
# 创建新表时将使用的默认存储引擎
default-storage-engine=INNODB
# 默认使用“mysql_native_password”插件认证
default_authentication_plugin=mysql_native_password
[mysql]
# 设置mysql客户端默认字符集
default-character-set=utf8
[client]
# 设置mysql客户端连接服务端时默认使用的端口
port=3306
default-character-set=utf8
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/691660bdac224b65999438778b5a2136.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

打开CMD，进入D:\mysql-8.0.25-winx64\bin目录，执行如下命令初始化创建data目录。
cd D:\mysql-8.0.25-winx64\bin
mysqld --initialize-insecure --user=mysql


（5）启动MySQL服务，并配置成系统服务
![在这里插入图片描述](https://img-blog.csdnimg.cn/a03d94e194ac4975806f082908c1b9dd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

使用系统管理员身份，启动CMD，执行如下命令将MySQL配置成Windows系统服务：

```python
mysqld -install --serviceName "MySQL"
Service successfully installed.
```
安装mysql服务方便以后启动：

```python
D:\Java\Database\mysql-8.0.18-winx64\bin>mysqld.exe install mysql
Service successfully installed
```
在服务列表中能找到刚刚安装的mysql服务，可设置其启动的方式 

右键单击此电脑打开任务管理器的服务，启动MySQL服务。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/dea6a6cef5d4492aa5941fe029a440f6.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/b425fee1e9a44921807b7f3df002101a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

`net start mysql`：启动mysql服务 
`net stop mysql`： 停止mysql服务

```python
# 启动mysql服务需要使用管理员角色
# 通过net start命令启动mysql服务 (net stop mysql --终止mysql服务命令)
  D:\Java\Database\mysql-8.0.18-winx64\bin>net start mysql
  mysql 服务正在启动 .
  mysql 服务已经启动成功
```

#### 2.1.2	设置root用户密码
（1）登录MySQL
在CMD中，输入以下命令登录MySQL（新安装的MySQL，可以无密码登录）：
`mysql -u root` -p

（2）设置root用户密码
输入如下命令，键入回车后执行SQL语句，设置root用户密码为123456：

```python
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123456';
```

（3）刷新保存配置
输入如下命令，保存配置并生效：`flush privileges;` 
输入`quit`退出数据库。

#### 2.1.3	使用DataGrip连接MySQL数据库
DataGrip2021安装参考[帖子1](https://blog.csdn.net/qq_31762741/article/details/115134775?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163957944416780265418841%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163957944416780265418841&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-115134775.pc_search_insert_es_download_v2&utm_term=datagrip%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4187)、[帖子2](https://blog.csdn.net/weixin_45078818/article/details/116054375?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163957944416780265418841%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163957944416780265418841&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-116054375.pc_search_insert_es_download_v2&utm_term=datagrip%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4187)。
（1）打开DataGrip工具，新建MySQL连接
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/2c81891e8fa144e293df9d0fbdc37cf2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

（2）配置MySQL连接
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/0f8c2899749f493195e7b5f486b8dd56.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)


（3）连接MySQL数据库
第一次连接mysql会报错，提示你缺少驱动，点击Download Driver Files就会自动帮你安装连接驱动。![在这里插入图片描述](https://img-blog.csdnimg.cn/09b34b59e4e34be7a5fec1394f5dd1fe.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
（4）更换中文语言教程，参考[此贴](https://blog.csdn.net/qq_31762741/article/details/115134775?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163957944416780265418841%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163957944416780265418841&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-115134775.pc_search_insert_es_download_v2&utm_term=datagrip%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4187)。
（5）创建userinfo和loginfo数据库
新建一个console窗口，在mysql窗口中输入如下SQL语句，创建数据库：
create database userinfo;
create database loginfo;
（此时不能创建mongodb连接，因为还没装mongodb，也没有启动。装了也连不上）
 
2.2	MongoDB数据库安装与使用
参考帖子[《【2021/8/19-最新教程】Windows安装MongoDB及配置（超详细）》](https://blog.csdn.net/qq_46092061/article/details/119811965?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163958533516780271568946%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=163958533516780271568946&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-4-119811965.pc_search_insert_es_download_v2&utm_term=windows%E9%85%8D%E7%BD%AEmongodb%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4187)
2.2.1	MongoDB数据库安装
（1）安装包下载
下载地址：https://www.mongodb.com/try/download/community 
安装包版本：5.0.5

（2）配置环境变量
在PATH下添加环境变量：D:\mongodb-win32-x86_64-windows-5.0.5\bin
 

（3）创建目录及配置文件
在bin目录同级的目录创建data目录，继续在data目录下创建db以及log，log目录中还需要创建mongod.log文件。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/b03fdda88be74e648b00a70204c5cfd5.png)


然后在bin目录的同级目录创建mongod.cfg文件：

```python
systemLog:
    destination: file
    path: D:\mongodb-win32-x86_64-windows-5.0.5\data\log\mongod.log
storage:
    dbPath: D:\mongodb-win32-x86_64-windows-5.0.5\data\db
net:
    port: 27017
```

- path是配置打印日志的目录
- dbpath是配置数据的存储位置
- port是配置的端口号

（4）启动MongoDB服务，并配置成系统服务

使用系统管理员身份，启动CMD，在D:\mongodb-win32-x86_64-windows-5.0.5\bin目录下执行如下命令将MongoDB配置成Windows系统服务：
```python
mongod --config D:\mongodb-win32-x86_64-windows-5.0.5\mongod.cfg --install --serviceName "MongoDB"
```

打开任务管理器的服务，查看MongoDB服务。
 此时就可以通过`net start MongoDB`和`net stop MongoDB`以及`net delete MongoDB`开启、关闭、删除MongoDB。

2.2.2	使用DataGrip连接MongoDB数据库
（1）打开DataGrip工具，新建MongoDB连接
 

（2）配置MongoDB连接
 
（3）连接MongoDB数据库
连接MongoDB数据库，在console中输入语句创建两个库（由于库中没有数据，在MongoDB中还看不到这两个库，等完成项目部署并运行调试之后，刷新MongoDB之后会出来这两个库）：
use NewsRecSys;
use SinaNews;

 

### 2.3	Redis数据库安装与使用
2.3.1	Redis数据库安装
（1）安装包下载
下载地址：https://github.com/tporadowski/redis/releases 
安装包版本：5.0.14

（2）启动Redis服务，并配置成系统服务
使用系统管理员身份，启动CMD，执行如下命令将Redis配置成Windows系统服务：

```python
cd D:\Redis-x64-5.0.14#命令地址
redis-server.exe --service-install redis.windows.conf --serviceName "Redis5.0.14"
#或者运行下面这个
redis-server --service-install redis.windows.conf --loglevel verbose
```

打开任务管理器的服务，查看Redis服务。
 
启动Redis：`net start Redis`
#### 2.3.2	使用redis-desktop-manager连接Redis数据库
>参考[《Redis可视化工具Redis Desktop Manager使用教程》](https://blog.csdn.net/weixin_40668023/article/details/91905748?ops_request_misc=&request_id=&biz_id=102&utm_term=Redis%20Desktop%20Manager%E6%95%99%E7%A8%8B&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-91905748.pc_search_insert_es_download_v2&spm=1018.2226.3001.4187)
[《Redis DeskTop Manager 使用教程》](https://blog.csdn.net/weixin_33859504/article/details/93832144?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163958926516780255243373%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163958926516780255243373&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-93832144.pc_search_insert_es_download_v2&utm_term=Redis%20Desktop%20Manager%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4187)


（1）安装Redis Desktop Manager软件
下载地址：从腾讯微云中获取，参见0.1节

（2）连接Redis数据库
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2b34a7914af54615bf17a245e738a4b8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
 最后点ok就行。点击左侧Redis-@localhost出现下拉列表：
![在这里插入图片描述](https://img-blog.csdnimg.cn/332005dea44d4513b09e0d126f8e5e33.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

## 3	前端项目运行
[安装WebStorm-2021](https://blog.csdn.net/xudali_1012/article/details/117534094?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163959359416780274139155%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163959359416780274139155&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-117534094.pc_search_insert_es_download_v2&utm_term=webstorm%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187)
![在这里插入图片描述](https://img-blog.csdnimg.cn/122c2d83f4484699a0adfefe7aa58c84.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

导入前端项目：open打开vue文件夹就行
![在这里插入图片描述](https://img-blog.csdnimg.cn/a39350a1ceac4cf5b27541f3c675f4e2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
### 3.1	安装依赖包
安装
安装node
![在这里插入图片描述](https://img-blog.csdnimg.cn/bbad58c531214107817c7e3a0406a0b4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

首先安装淘宝的npm，在Terminal中执行如下命令：

```python
npm install -g cnpm --registry=https://registry.npm.taobao.org
cnpm install
```


### 3.2	修改前端访问IP和端口
打开文件package.json，修改第49行的IP和端口，修改内容如下：

```python
"scripts": {
  "test": "echo \"Error: no test specified\" && exit 1",
  "dev": "webpack-dev-server --open --port 8686 --contentBase src --hot --host 0.0.0.0",
  "start": "nodemon src/main.js"
},
#锐锋是127.0.0.1
```

127.0.0.1表示游览器的访问IP（也称为本地IP），8686表示访问端口
### 3.3	修改访问后端API接口的IP和端口
打开文件main.js，文件路径：src/main.js，修改第23行的IP和端口，修改内容如下：

```python
// Vue.prototype.$http = axios
Vue.use(VueAxios, axios);
// axios公共基路径，以后所有的请求都会在前面加上这个路径
// axios.defaults.baseURL = "http://10.170.4.60:3000";
// axios.defaults.baseURL = "http://47.108.56.188:3000";
axios.defaults.baseURL = "http://127.0.0.1:5000"
```

127.0.0.1表示后端项目的访问IP（也称为本地IP），5000表示访问端口
### 3.4	运行前端项目
在Terminal中执行命令运行前端项目:`npm run dev`
![在这里插入图片描述](https://img-blog.csdnimg.cn/0993baf2421249babfed2927ecaeb7bf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

 
浏览器会自动访问地址：http://127.0.0.1:8686/#/
![在这里插入图片描述](https://img-blog.csdnimg.cn/83ecc2fc191940a3be7e5e1e9d5b34a1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

 
通过打开“开发者工具”，调节设备工具栏，显示正常比例的页面
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/59ef280c39df421d814fa4174b65f836.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

## 4	后端项目运行
### 4.1	配置环境,安装依赖
1. 安装conda环境，并创建虚拟环境
创建指定路径的Python环境：`conda create --prefix venv python=3.8`

虚拟环境位置：
![在这里插入图片描述](https://img-blog.csdnimg.cn/f5457bb18bdd400383c037add2dc7b52.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_13,color_FFFFFF,t_70,g_se,x_16)
包装在libs下面的site-pakeages:
![在这里插入图片描述](https://img-blog.csdnimg.cn/6156da64eced4666966c649f2091d7c6.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_14,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/d54e9b59faaa44dbb4a1418d5103bfee.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)


在PyCharm中，设置Python解释器
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/c6c89560e48f47348d21338b5be58911.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

2. 安装依赖文件
在Terminal中执行命令安装依赖包：`pip install -r requirements.txt`

![在这里插入图片描述](https://img-blog.csdnimg.cn/6e4b9062a1064cfcbabc9aba8baf960e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

 
 ### 4.2 修改端口，配置文件
1. 修改后端项目的IP和端口
打开文件server.py，修改第233行的IP和端口，修改内容如下：

```python
if __name__ == '__main__':
    # 允许服务器被公开访问
    # app.run(debug=True, host='0.0.0.0', port=3000, threaded=True)
    # 只能被自己的机子访问
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
```

127.0.0.1表示后端提供给前端的IP（也称为本地IP），5000表示端口。

2. 修改项目路径配置文件proj_path.py
因为没有配置home路径，所以改为读取项目地址。修改项目路径配置文件proj_path.py，文件路径：conf/proj_path.py

```python
# home_path = os.environ['HOME']
# proj_path = home_path + "/fun-rec/codes/news_recsys/news_rec_server/"
proj_path = os.path.join(sys.path[1], '')
```

3. 核对数据库配置文件dao_config.py
打开数据库配置文件dao_config.py，文件路径：conf/dao_config.py，核对以下配置：

```python
# MySQL默认配置
mysql_username = "root"
mysql_passwd = "123456"
mysql_hostname = "localhost"
mysql_port = "3306"

# MongoDB配置
mongo_hostname = "127.0.0.1"
mongo_port = 27017

# Redis配置
redis_hostname = "127.0.0.1"
redis_port = 6379
```

### 4.3 	启动雪花算法服务
在Terminal中执行命令启动雪花算法服务，用于生成用户ID，启动命令如下：

```python
snowflake_start_server --address=127.0.0.1 --port=8910 --dc=1 --worker=1
```


### 4.4	创建logs目录
在根目录下，创建logs目录，如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/3582ab8e23ff4b07b3a9989699e6c4d2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

 
### 4.5 启动后端项目
启动server.py程序（注：在此之前，必须启动数据库并创建数据库，详见2.1.3节和2.2.2节），执行如下命令：
python server.py
![在这里插入图片描述](https://img-blog.csdnimg.cn/12c5b5947bd84a71aaab73cf9e51670e.png)

 
## 5	项目整体运行与调试
注册用户
 
### 5.1	爬取新浪新闻
通过查看crawl_news.sh文件（文件路径：scheduler/crawl_news.sh），可知爬取新浪新闻的代码在如下目录
/materials/news_scrapy/sinanews/run.py
使用PyCharm的Run按钮，手动执行该代码，需要配置参数：
—pages=30

![在这里插入图片描述](https://img-blog.csdnimg.cn/3a0d7d8f91774f90b1ca2f709db08c52.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/c65de70dd5f9404abdc7d56d20804897.png)

 
 

### 5.2	更新物料画像
通过查看offline_material_and_user_process.sh文件（文件路径：scheduler/offline_material_and_user_process.sh），可知更新物料画像的代码在如下目录：
materials/process_material.py
使用PyCharm的Run按钮，手动执行该代码
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/17cd2ccaf3d94bf5ac8d8b93a167f6e5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)


### 5.3	更新用户画像
通过查看offline_material_and_user_process.sh文件（文件路径：scheduler/offline_material_and_user_process.sh），可知更新用户画像的代码在如下目录：
materials/process_user.py
使用PyCharm的Run按钮，手动执行该代码
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/ba2509dd448a450caff81142bfe76b6f.png)


### 5.4	清除前一天redis中的数据，更新最新今天最新的数据
通过查看offline_material_and_user_process.sh文件（文件路径：scheduler/offline_material_and_user_process.sh），可知清除前一天redis中的数据，更新最新今天最新的数据的代码在如下目录：
materials/update_redis.py
使用PyCharm的Run按钮，手动执行该代码
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/f4265cf10e72450da4b50457a9c301c4.png)

### 5.5	离线将推荐列表和热门列表存入redis
通过查看run_offline.sh文件（文件路径：scheduler/run_offline.sh），可知离线将推荐列表和热门列表存入redis的代码在如下目录：
recprocess/offline.py
使用PyCharm的Run按钮，手动执行该代码
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/6fd3cdaa10e64cc19040601fa4313ea9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

### 5.6	重新登录用户查看新闻
 
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/65be2740b23243a6b677e8e0fad1f1b6.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)








