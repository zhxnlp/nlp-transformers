视频讲解地址：[Task01：熟悉推荐系统基本流程](https://datawhale.feishu.cn/minutes/obcnzns778b725r5l535j32o)
[推荐系统项目网页地址](http://47.108.56.188:8686/#/recLists)

## 一、简介
本项目是离线的推荐系统。即不是实时的通过用户 ID 等信息通过模型实时、动态地获取用户的推荐列表，而是==提前已经把用户的这个推荐列表计算好了，用Redis存到了这个倒排索引表里面。我们线下线上真正要做的事情就是从这个 Redis 里面去拉取就够了。所以这里可能就会存在一个 T + 1 的延迟。（后面一天才会更新前一天的动态数据）==

推荐系统架构如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/7dd55b57470b4889b907e1b7cea0bb95.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
除了离线offline部分，系统还有online部分，跟后端数据交互。
onlineserver：offline得到的的倒排索引数据经过online server处理后才给到online。如果不处理直接拉过来，会产生问题：
- 推荐页和热门页挤在一起重复曝光（推荐页看过的热门页又看了一遍）
- 同的类别的很多东西会摞在一起，影响体验（同类产品score可能相近）
- 新用户需要冷启动策略
recsysserver：和onlineserver一样处理后端数据

offline：得到用户第二天需要展示的推荐列表。
- 爬取新物料并处理成物料画像
- 获得用户操作的动态信息，更新用户画像（年龄性别、物料侧标签得到的长短期兴趣等）
- 根据模型做出推荐列表

画像处理有两部分：处理新来物料，更新旧物料动态属性。处理完后从Mongo DB 存到Redis（前端新闻展示信息）。如果还是在Mongo DB操作会很卡。
热门页：直接更新物料库里所有新闻的热度，做出倒排索引存入Redis。
推荐页：用模型得出倒排索引，涉及冷启动问题

## 二、网页
打开http://47.108.56.188:8686/#/之后按下F12键，然后登陆系统。在network里面有login、user_id等字典信息，是通过json的前后端交互得到的。
code表示前后端处理状态/进度，这里表示登陆成功login success。
![在这里插入图片描述](https://img-blog.csdnimg.cn/b8ef0c7375144a0d9b411452d187529e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
现在跳转到推荐页。rec_list?user_id=zhxscut：表示当前页展示信息。也有状态和json。
登陆之后跳转到推荐页也是按设定逻辑前后端交互好的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/f5bc5d79199d420fa77ac37ad41811f7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
data点开之后是一个数组，每个数组点开是一条新闻。包括类别、时间、标题、收藏、喜欢、阅读次数、url等。
![在这里插入图片描述](https://img-blog.csdnimg.cn/4ef2b5364bcf49c6bfa11a12530bc86e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/916adedbb25f4d1b8fef12006497639d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_14,color_FFFFFF,t_70,g_se,x_16)
左侧推荐页面展示10条数据，往下拉的时候不够了不够展示了，再往下拉的时候他会重新请求重新去后端拉 10 个数据，右侧显示几个新的rec_list。
点击热门页，右侧显示hot_list?user_id=zhxscut：。
打开一篇新闻，会有news_detail?和一个action。点击喜欢或者收藏会有一个action。
![在这里插入图片描述](https://img-blog.csdnimg.cn/a80a2ba84cfd4449be881a60f24b02ce.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
前后端交互核心：通过url链接。前端展示的 URL （这个项目的网址）和我们后端资源访问的 URL （推荐新闻页面）是不一样的
## 三、项目代码解读
### 3.1 前端url
![在这里插入图片描述](https://img-blog.csdnimg.cn/35b7d1fc8b024bf6afb52ae91857aa6e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

router.js：路由，path：‘/’就是根目录，默认是signin界面。对应的是componence下面的signin.vue页面,展示的样式、处理逻辑等。
![在这里插入图片描述](https://img-blog.csdnimg.cn/44dbec30656441d483551bd666dd8b45.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
除了登陆页面还有注册、hot_list、rec_list页面等。

前后端交互的时候，我们可以大框架都可以看不懂，但是我们可以。但至少要知道比如说我推荐什么东西之后，一些我要搞的东西可能有些变化。那我们应该去看哪块的代码，他怎么是写的，我应该怎么改，大概需要了解一下这个

![在这里插入图片描述](https://img-blog.csdnimg.cn/9a0b9612f6214c1b8a6efe1983bcd58e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

上图表示post后端请求验证用户名和密码，登陆后跳转到推荐页。
状态码：
![在这里插入图片描述](https://img-blog.csdnimg.cn/3039bde92f3642f58b9dbe50de19d2cd.png)
### 3.2 后端url
![在这里插入图片描述](https://img-blog.csdnimg.cn/4c53729673c84bdfa5c5fb4e5d55619b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
后端url通过get list 方法获取，即通过 get 请求去后端拉取数据。
get 请求它就是这种格式：一个问号后面加上一些参数，比如说用户名或者是年龄性别之类的，当然还有 post 的请求。
get 请求：前端往后端发东西的时候会把一些具体的参数写在这个 URL 里面。

flask ： Python 的一个 web 框架，做20w用户的网站是没问题的，再多就不行了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/56648b9d7b6a4c748efbcc201fa5f0c2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
server：app.route是一个路由，装饰器，第一个参数是url，第二个参数是get请求（对应下面的arg.get）。然后绑定一个具体的函数rec_list。
绑定的函数通过 flask 路由，根据当前的这个 URL 实现它要处理的具体逻辑，比如说现在是获取推荐列表（上一步reclist.vue里面的推荐列表）

try：
	rec_new_list=：表示从后端Redis拿取数据，即onlineserver、recsysserver后端请求这块。

signin没有url，不用get请求，而是post请求直接request.get_data。

## 四、推荐系统流程
### 4.1 物料处理
物料处理自动化主要体现在使用 clone Tab 然后把整个链路给连串起来了：
![在这里插入图片描述](https://img-blog.csdnimg.cn/8ce84dd9357840a3bf380a9e1c61adbc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
上图表示：每天0点会运行很多命令。
- 爬虫获取物料，代码在materials下的news_scrapy。爬取完存在mongoDB。
- 写入日志
- 离线处理物料（新闻物料和用户画像）
- 处理完写入日志
- 通过算法策略等用处理后物料生成排序列表，存入Redis
- 再写入日志。
整个过程就串起来了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/d5398c3bab1040e6b7a6e9635fba29d5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
终端输入mongo，进入后输入show dbs，显示所有数据库。
第一个是存推荐系统相关的物料画像、用户画像。
输入use SinaNews切换数据库、show collections显示数据。
通过以下命令查看爬取的新闻：
![在这里插入图片描述](https://img-blog.csdnimg.cn/b50bc5cb5707436c8b96bea51cff5037.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
每天爬取的数据会存盘一次，方便回溯。
### 4.2 物料处理
爬取的物料通过下面的sh文件，运行里面的三个py代码，更新画像和更新Redis。第三个py文件没有排序列表，而是新闻的详细信息。
![在这里插入图片描述](https://img-blog.csdnimg.cn/33ff79550d8547dea94218577908e27d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
#### 4.2.1 更新物料画像，代码如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/86d84fd455c84809b4f8bfd265c1ba25.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
更新存入mongo是为了备份，避免Redis数据被清空。

mongo NewsRecSys下面有三个集合。Feature集合是特征画像（物料池），只存了一份（方便处理，也可以分布式存储），包含每天用户交互的一些新闻产生的动态特征。
第二个部分update_news_items()代码如下：（点击函数跳转）
![在这里插入图片描述](https://img-blog.csdnimg.cn/ca5611c0c224475e899880b24850c8e3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
遍历今天爬去的所有数据（SinaNews），去重，初始化等生成特征画像，存入物料池（比如点击收藏开始都是0，热度是1000，逐渐衰减）
以上对应：
![在这里插入图片描述](https://img-blog.csdnimg.cn/bebbbf63bba442f88b25d89d6127c2e5.png)
#### 4.2.2 当天新闻动态画像更新
- 用户浏览新闻时，会有点击、收藏、喜欢等操作。这个是需要实时反馈的，但是又不可能每次都去MongoDB 那个物料库里面去把他拉回来展示。把新闻展示的物料分成了两部分，一部分是静态的，一部分是动态的，都是存在 Redis 里面。
- 静态：标题、类别、详情页
- 动态：阅读次数、喜欢次数和收藏次数。
- 用户在前端交互完之后，立马修改动态的信息，并且再拉回来再展示。此时物料池并没有更新
- Redis 在清除缓存前，需要遍历动态画像，更新到物料画像池。

动态信息更新到物料池代码：
![在这里插入图片描述](https://img-blog.csdnimg.cn/018c45ebe4d9446b8cfcece9e97e4513.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

![在这里插入图片描述](https://img-blog.csdnimg.cn/b21498be38cd47c1ae173921256a000f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
dao_config：注释了每个数据库存了什么。
![在这里插入图片描述](https://img-blog.csdnimg.cn/361ad3949e8c4829a2eb4826c3030895.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
上面显示动态数据在2号库，静态在1号库。redis就是一个k-v系统。
![在这里插入图片描述](https://img-blog.csdnimg.cn/76823696fd5d43b7b1c3e3d5e2c7bb47.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
通过get得到静态信息。
==通过新闻 ID get对应的静态信息和动态信息，拼接起来送到这个前端展示==

redis-cli --raw参数设置可以看中文，否则有些信息显示乱码。redis-cli 无空格。2号库动态信息展示如图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/7fd2bb930b6e4dc7babe6a4cf16206a8.png)

