@[toc]
## 0.解决npm命令语法不正确问题
### 0.1  powershell报错
![在这里插入图片描述](https://img-blog.csdnimg.cn/2a4d50e19ca7476a8ac33976be4071c0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
解决方案：
根据上面提升报错的环境变量把环境变量Path中含有 ； 的分开写
![在这里插入图片描述](https://img-blog.csdnimg.cn/99d033d1644845a38a4ddf0fcda8e270.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

2. 第二个报错
![在这里插入图片描述](https://img-blog.csdnimg.cn/10342ec9d4534a2ea2828cf81b5953ef.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
- PowerShell默认禁止运行脚本，但是因为安装Anaconda后再启动PowerShell时需要运行脚本，所以会报错。
-  以管理员身份运行PowerShell，执行 `set-ExecutionPolicy RemoteSigned`，然后输入 `Y`，重启PowerShell：
![在这里插入图片描述](https://img-blog.csdnimg.cn/cd8fb0e578a24c169c0592a9476b0281.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
### 0.2 win10家庭版升级
参考帖子[《win10激活密钥》](https://blog.csdn.net/weixin_30540691/article/details/101459865?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164069499916780265470211%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164069499916780265470211&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-101459865.pc_search_insert_es_download_v2&utm_term=windows10%E6%BF%80%E6%B4%BB%E5%AF%86%E9%92%A5&spm=1018.2226.3001.4187)。
专业版秘钥用不了，装的教育版。直接此电脑——右键属性——产品密钥，输入合适的密钥就安装对应的win10版本。（教育版：NW6C2-QMPVW-D7KKK-3GKT6-VCFB2）
![在这里插入图片描述](https://img-blog.csdnimg.cn/256f5169649c47089d4edf7832ce946f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
激活步骤：
管理员打开powershell，输入以下命令：（cmd打开会提示没有slmgr.vbs命令）
```python
slmgr.vbs /upk                               //卸载秘钥
slmgr /ipk NW6C2-QMPVW-D7KKK-3GKT6-VCFB2     //产品密钥可用并且需要连接互联网
slmgr /skms kms.03k.org                     //指定KMS服务器的地址和端口,服务器地址和端口请根据实际情况修改。
slmgr /ato                                  //进行激活
```

## nodejs
### 1.1 centos安装nodejs
切换到目录：`cd /usr/local`
压缩包解压:`tar -zxvf node-v15.10.0-linux-x64.tar.gz` 
mv命令将解压文件夹重命名为nodejs
```python
tar -xvf file.tar //解压 tar包
tar xf node-v16.13.1-linux-x64.tar.xz//解压 tar.xz包
tar -xzvf file.tar.gz //解压tar.gz
tar -xjvf file.tar.bz2   //解压 tar.bz2
tar –xZvf file.tar.Z   //解压tar.Z
unrar e file.rar //解压rar
unzip file.zip //解压zip

rm-f　　-force　　忽略不存在的文件，强制删除，无任何提示
-i　　　--interactive　　　 进行交互式地删除
rm -ivrf dirname 删除目录
```
建立软连接：

```python
ln -s /usr/local/nodejs/bin/node /usr/local/bin
ln -s /usr/local/nodejs/bin/npm /usr/local/bin
```
查看是否正确安装
```python
[root@192 local]# node -v
v15.10.0
[root@192 local]# npm -v
7.5.3
```
各种报错：
1. 安装nodejs15后执行`npm install -g cnpm --registry=https://registry.npm.taobao.org`报错：
![在这里插入图片描述](https://img-blog.csdnimg.cn/f050c16ed60942cfbb875cac6d2df79e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
这是npm的版本太低了版本过低不支持nodejs。

2. 结果再次`npm install -g cnpm --registry=https://registry.npm.taobao.orgnpm`报错
![在这里插入图片描述](https://img-blog.csdnimg.cn/57130bfa4bde44d99b2a3ad0128d26a2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
怀疑是上一次的nodejs没删干净，但其实是ping www.baidu.com都有问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/cfab59f2a1254876a6a8a2a82a83d7d2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

之后参考：[linux里面ping www.baidu.com ping不通的问题](https://blog.csdn.net/weixin_43700340/article/details/88393833?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163965925416780274135285%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163965925416780274135285&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-5-88393833.pc_search_insert_es_download_v2&utm_term=ping%20www.baidu.com&spm=1018.2226.3001.4187)。就ok了
配置DNS：打开文件 vim /etc/resolv.conf，注释第二行并输入：
![在这里插入图片描述](https://img-blog.csdnimg.cn/f149a64bfda74cba96a0855d5bf95291.png)

```python
#Google
nameserver 8.8.8.8
nameserver 8.8.4.4
```
现在ping registry.npmjs.org也ok了。
输入`npm config set registry http://registry.npm.taobao.org`。测试`ping registry.npm.taobao.org`成功。
- 查看npm源地址
```python
npm config get registry
#http://registry.npmjs.org 为国外镜像地址
#设置阿里云镜像
npm config set registry http://registry.npm.taobao.org
#http://registry.npm.taobao.org/现在是淘宝镜像
npm config set registry http://registry.npm.taobao.org#安装cnpm
```
这个是不知道中间执行了啥报错了。即越做越错。
![在这里插入图片描述](https://img-blog.csdnimg.cn/7ea41e5a166c4a01a36e5575aa615546.png)

装完nodejs16，弄好DNS服务。`npm install -g npm@8.3.0`升级npm到8.3。

### 1.2  win10安装nodejs
官网下载安装node-v16.13.1-x64后，cmd运行`npm -v`一直显示命令语法不正确。升级win10家庭版到专业版，各种操作都不行。打开git-bash，运行就ok了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/5ccfc4ee6f3c4252a31b4d6b51e0467a.png)
运行`npm config set registry http://registry.npm.taobao.org`
运行`npm install -g npm@8.3.0`升级npm到8.3。
安装vue：`pip install vue`
安装vue-cli：`npm install -g @vue/cli`

```python
$ npm install joi
npm ERR! code EPERM
npm ERR! syscall mkdir
npm ERR! path D:\
npm ERR! errno -4048
npm ERR! Error: EPERM: operation not permitted, mkdir 'D:\'
npm ERR!  [Error: EPERM: operation not permitted, mkdir 'D:\'] {
npm ERR!   errno: -4048,
npm ERR!   code: 'EPERM',
npm ERR!   syscall: 'mkdir',
npm ERR!   path: 'D:\\'
npm ERR! }
npm ERR!
npm ERR! The operation was rejected by your operating system.
npm ERR! It's possible that the file was already in use (by a text editor or ant
ivirus),
npm ERR! or that you lack permissions to access it.
npm ERR!
npm ERR! If you believe this might be a permissions issue, please double-check t
he
npm ERR! permissions of the file and its containing directories, or try running
npm ERR! the command again as root/Administrator.

npm ERR! A complete log of this run can be found in:
npm ERR!     C:\Users\LS\AppData\Local\npm-cache\_logs\2021-12-30T16_48_16_213Z-
debug-0.log
```
bash没有管理员权限。此时右键单击git-bash.exe，以管理员身份运行`npm install joi`。
![在这里插入图片描述](https://img-blog.csdnimg.cn/4e900e52f9e54dbc8a01ecfeff76941b.png)

## 2. vue
### 2.1 安装vue
安装vue：`pip install vue`
安装vue-cli：`npm install -g @vue/cli`
![在这里插入图片描述](https://img-blog.csdnimg.cn/ee2645568b0444388b67156108860054.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
报错一大堆，我一直以为是安装失败。后来才知道是有些包过时弃用了不碍事。改了两处：

```python
npm install uuid@8.3.2
npm install joi
```
之后建立软连接：`ln -s /usr/local/nodejs/bin/vue /usr/local/bin/vue`
输入 `vue -v`
![在这里插入图片描述](https://img-blog.csdnimg.cn/cdf0eea5932d4a5fb9e12c6e627865a0.png)
原来已经装成功了。

wget http://www.percona.com/redir/downloads/Percona-XtraDB-Cluster/5.5.37-25.10/RPM/rhel6/x86_64/Percona-XtraDB-Cluster-shared-55-5.5.37-25.10.756.el6.x86_64.rpm
wget http://www.percona.com/redir/downloads/Percona-XtraDB-Cluster/5.5.37-25.10/RPM/rhel6/x86_64/Percona-XtraDB-Cluster-shared-55-5.5.37-25.10.756.el6.x86_64.rpm
rpm -ivh Percona-XtraDB-Cluster-shared-55-5.5.37-25.10.756.el6.x86_64.rpm

### 2.2 创建vue项目
官网教程：[创建一个项目](https://cli.vuejs.org/zh/guide/creating-a-project.html#%E4%BD%BF%E7%94%A8%E5%9B%BE%E5%BD%A2%E5%8C%96%E7%95%8C%E9%9D%A2)

```python
vue create hello-world
```
可以选默认的包含了基本的 Babel + ESLint 设置的 preset，也可以选“手动选择特性”来选取需要的特性。
![在这里插入图片描述](https://img-blog.csdnimg.cn/ae49532f1d9640b2b91814fbaec5ff33.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
vue create 命令有一些可选项，你可以通过运行以下命令进行探索：`vue create --help`
你也可以通过 vue ui 命令以图形化界面创建和管理项目：`vue ui`
上述命令会打开一个浏览器窗口，并以图形化界面将你引导至项目创建的流程。
![在这里插入图片描述](https://img-blog.csdnimg.cn/4e72d8d9381b47aa9e84f7e6cbae730e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
Vue CLI >= 3 和旧版使用了相同的 vue 命令，所以 Vue CLI 2 (vue-cli) 被覆盖了。如果你仍然需要使用旧版本的 vue init 功能，你可以全局安装一个桥接工具：

```python
npm install -g @vue/cli-init
# `vue init` 的运行效果将会跟 `vue-cli@2.x` 相同
vue init webpack my-project
```
Vue项目目录和说明见教程：
### 2.3 使用Vue开发H5页面
```python
# vue create创建项目
vue create test

# 进入项目具体路径
cd test

# 下载依赖
npm install

# 启动运行项目
npm run serve 
```
此时输不了命令，也没有app界面弹出（minicentos没有图形界面）
ctrl+z退出后输入npm run build：
![在这里插入图片描述](https://img-blog.csdnimg.cn/193c0c2d582b4cc59f17cd4e80bc8ed5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/0b33543fc78340a8b0ec8f70d659adad.png)
构建完成后，可以看到前端项目根目录下多了一个dist文件夹，这就是要部署的前台文件。
>参考[《centos部署vue项目》](https://blog.csdn.net/weixin_30367543/article/details/99511206?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163968487616780265472282%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163968487616780265472282&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-99511206.pc_search_insert_es_download_v2&utm_term=centos%E9%83%A8%E7%BD%B2vue%E9%A1%B9%E7%9B%AE&spm=1018.2226.3001.4187)[《Centos7部署Vue项目》](https://blog.csdn.net/qq_41082746/article/details/106698019?ops_request_misc=&request_id=&biz_id=102&utm_term=centos%E9%83%A8%E7%BD%B2vue%E9%A1%B9%E7%9B%AE&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-106698019.pc_search_insert_es_download_v2&spm=1018.2226.3001.4187)
>参考链接：
https://www.runoob.com/vue3/vue3-directory-structure.html
https://blog.csdn.net/weixin_41887155/article/details/107648969
https://www.cnblogs.com/jhpy/p/11873270.html
https://blog.csdn.net/chao2458/article/details/81284522

### 2.4 部署新闻推荐前端项目
1. 跳转到前端项目文件目录：cd Vue-newsinfo

2. 本地安装node环境，在项目根目录命令行输入命令`npm install`安装依赖包

如果因为版本或者网络问题下载失败请执行npm install -g cnpm -registry=https://registry.npm.taobao.org/ 和cnpm install

3. 启动前端服务：`npm run dev`

本机访问地址http://localhost:8686/#/

4. 根据需要修改package.json下`"scripts": { "dev": "webpack-dev-server --open --port 8686 --contentBase src --hot --host 0.0.0.0"}`,中的ip和端口号)

```python
npm ERR! JSON.parse package.json must be actual JSON, not just JavaScript
#修改package.json错误导致
```
5. 修改main.js：`vim /home/Vue-newsinfo/src/main.js`
这个是后端访问的地址，也就是网页F12时用户操作的地址。
```python
Vue.use(VueAxios, axios);
// axios公共基路径，以后所有的请求都会在前面加上这个路径
//axios.defaults.baseURL = "http://47.108.56.188:3000";
//axios.defaults.baseURL = "http://127.0.0.1:3000"
axios.defaults.baseURL = "http://0.0.0.0:3000"
```
centos本地服务启动后想退出按`ctrl+c`不是ctrl+z。
`netstat -tunlp | grep 8080`查看端口是否被占用
`systemctl stop firewalld.service`关闭防火墙
`yum install telnet httpd`安装telnet、httpd。
`netstat -nlp` 查看是否映射成功
`telnet 192.168.112.1 10022`连接主机
`telnet 192.168.112.1 6379`连接redis

4. 虚拟机端口映射
参考[此贴](https://blog.csdn.net/aod83029/article/details/102163544?ops_request_misc=&request_id=&biz_id=102&utm_term=%E8%99%9A%E6%8B%9F%E6%9C%BA%E7%AB%AF%E5%8F%A3%E6%98%A0%E5%B0%84&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-102163544.nonecase&spm=1018.2226.3001.4187)
![在这里插入图片描述](https://img-blog.csdnimg.cn/dacfd75c157e49ffaf91f4dc348df679.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_19,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/14430c3bc0fe4bcfa8b8ffd3a0e8ed5f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_12,color_FFFFFF,t_70,g_se,x_16)
最后输入`http://localhost:8686/#/`访问。

5. 点击F12或者右键选择检查打开开发者模式,选中移动端浏览（点击左上角箭头右边的手机按钮）开始体验


## 3. 后端配置
### 3.1 创建conda虚拟环境
输入conda命令，如正常返回，说明conda安装成功。
查看已有环境列表，*表示当前环境，base表示默认环境：`conda env list`
![在这里插入图片描述](https://img-blog.csdnimg.cn/4aa8a52871144e4e9bb81ee9f7fd7159.png)
`python --version`查看python版本

1. 使用命令`conda create -n sina python=3.6.8`创建环境，这里创建了名称为sina的python版本号为3.6.8的虚拟环境，稍微等待，过程中输入“y”。

将后端文件全部拷贝到centos上，切换到目录：`cd /home/news_rec_server`

2. 创建指定路径的Python环境：`conda create --prefix venv python=3.6.8`指定路径venv，名字是sina。
![在这里插入图片描述](https://img-blog.csdnimg.cn/8422680ac4a14d999f41439b1fa161fa.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

进入其他环境需要使用`conda activate`手动切换（一开始没有命令，结果悲剧了）
退出当前环境，使用`conda deactivate`，默认回到base 环境

- 在新的环境中使用下面的代码安装旧的环境中对应的包：`pip install -r requirements.txt`
- 直接安装失败报错;

```python
Could not fetch URL https://pypi.tuna.tsinghua.edu.cn/simple/pip/: There was a problem confirming the ssl certificate: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:852) - skipping
```
解决方法:
```python
python -m pip install --upgrade pip --user -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
```

中间报错一次，将selenium==4.0.0改成3.14.1就ok了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/80c20c68c2624b679714085067ec21af.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
删除虚拟环境，直接找到环境所在的目录，手动删除

```python
conda remove -n your_env_name(虚拟环境名称) --all 删除虚拟环境
conda remove --name your_env_name package_name 删除虚拟环境中的某个包
```
旧环境导出：

```python
#切换到需要生成requirements.txt 的虚拟环境中;在终端运行下面的代码
pip freeze >requirements.txt

pip list --format=freeze > requirements.txt
#该命令可以得到安装在本地的库的版本名
```

- 进入环境后，可使用如下命令安装依赖的包，使用的是已经配置好的清华的源，这里以“opencv-python”包为例，由于使用了清华大学的镜像源，下载速度很快。

```python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
```

- 换回conda默认的源，访问起来可能有些慢，但总比无法访问好。
conda config --remove-key channels
### 3.2 后端文件配置
 ### 4.2 修改端口，配置文件
1. 修改后端项目的IP和端口
后端获取网页地址的代码在最后一行（点击登录的时候login信息）
![在这里插入图片描述](https://img-blog.csdnimg.cn/524998e228654f8ab9af0f49894fe994.png)

打开文件server.py，修改第233行的IP和端口，修改内容如下：

```python
if __name__ == '__main__':
    # 允许服务器被公开访问
     app.run(debug=True, host='0.0.0.0', port=3000, threaded=True)#我的这个不变
    # 只能被自己的机子访问
    #app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
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
news_rec_server下在Terminal中执行命令启动雪花算法服务，用于生成用户ID，启动命令如下：

```python
snowflake_start_server --address=127.0.0.1 --port=8910 --dc=1 --worker=1
```
### 4.4 创建数据库

```python
mysql -u root -p #登录mysql
grep "password" /var/log/mysqld.log#查看密码，直接复制登录
CREATE DATABASE loginfo；
CREATE DATABASE userinfo;
```
创建mongodb库

```python
mongo
use SinaNews
use NewsRecSys
```

### 4.4	创建logs目录
在根目录下，创建logs目录，如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/3582ab8e23ff4b07b3a9989699e6c4d2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

 
### 4.5 启动后端项目
启动server.py程序（注：在此之前，必须启动数据库并创建数据库，详见2.1.3节和2.2.2节），执行如下命令：
先在sina虚拟环境下安装flask：

```python
conda activate sina
pip install Flask
```
python server.py
![在这里插入图片描述](https://img-blog.csdnimg.cn/12c5b5947bd84a71aaab73cf9e51670e.png)

 
## 5	项目整体运行与调试
注册用户，注册的用户数据在mysql的userinfo table下面：
![在这里插入图片描述](https://img-blog.csdnimg.cn/6f7d11017cfa4727921003d6aa673d25.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
可以直接删除就没有用户注册信息了（datagrap删除后要刷新）
 现在就可以登录网址注册登录了，但是没有新闻展示，因为还没有开始推送新闻，所以用户看不到新闻。终端报错：![在这里插入图片描述](https://img-blog.csdnimg.cn/6736c30e0356408d97c8184ea616c0c8.png)

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

