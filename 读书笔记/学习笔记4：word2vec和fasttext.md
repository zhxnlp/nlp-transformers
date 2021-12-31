FastText：快速的文本分类器
@[toc]
## 一、word2vec
参考文档[《word2vec原理和gensim实现》](https://maxiang.io/note/#%E4%B8%80-cbow%E4%B8%8Eskip-gram%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80)、[《深入浅出Word2Vec原理解析》](https://zhuanlan.zhihu.com/p/114538417)

### 1.1 word2vec为什么 不用现成的DNN模型

1.  最主要的问题是DNN模型的这个处理过程非常耗时。我们的词汇表一般在百万级别以上，==从隐藏层到输出的softmax层的计算量很大，因为要计算所有词的softmax概率，再去找概率最大的值==。解决办法有两个：霍夫曼树和负采样。
2. 对于从输入层到隐藏层的映射，没有采取神经网络的线性变换加激活函数的方法，而是采用简单的对所有输入词向量求和并取平均的方法。输入从多个词向量变成了一个词向量
3.  在word2vec中，由于使用的是随机梯度上升法，所以并没有把所有样本的似然乘起来得到真正的训练集最大似然，仅仅每次只用一个样本更新梯度，这样做的目的是减少梯度计算量
### 1.2 word2vec两种模型：CBOW和Skip-gram
&#8195;&#8195;Word2Vec是轻量级的神经网络，其模型仅仅包括输入层、隐藏层和输出层，模型框架根据输入输出的不同，主要包括CBOW和Skip-gram模型。 
- CBOW的方式是在知道词$w_{t}$的上下文$w_{t-2}$、$w_{t-1}$和$w_{t+1}$、$w_{t+2}$的情况下预测当前词$w_{t}$。
- Skip-gram是在知道了词$w_{t}$的情况下,对词的上下文进行预测，如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/8486814a2075418a807be2402c23abf1.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

### 1.2 word2vec两种优化解法：霍夫曼树和负采样
1. 霍夫曼树解法：
	- 采用霍夫曼树来代替隐藏层和输出层的神经元，霍夫曼树的叶子节点起到输出层神经元的作用，叶子节点的个数即为词汇表的小大。 而内部节点则起到隐藏层神经元的作用。
	- 把之前计算所有词的softmax概率变成了查找二叉霍夫曼树。那么我们的softmax概率计算只需要沿着树形结构进行，从根节点一直走到我们的叶子节点的词。<font color='red'>将每个节点向左或向右走的概率连乘就是最终预测的概率。训练时只更新对应通路的w，与全连接W相比大大减少。
	- <font color='red'>因为涉及连乘，每次乘的概率都是小于1，所以越到深层概率越低。所以其实存在一个词与词之间概率不对等的问题。
	- 霍夫曼编码：由于权重高的叶子节点越靠近根节点，编码值较短。而权重低的叶子节点会远离根节点，编码值较长。这保证的树的带权路径最短，也符合我们的信息论，即我们希望==越常用的词拥有更短的编码，查找就更快==。如何编码呢？参见上面提的文档
	
2. 负采样：
	-  使用霍夫曼树可以提高模型训练的效率。但是如果我们的训练样本里的中心词是一个很生僻的词，那么就得在霍夫曼树中辛苦的向下走很久了。
	- Negative Sampling：word2vec用神经网络解法时，输出是计算V类的概率，其中1类是中心词，概率往大的方向走，剩下一类是V-1个其它词，概率往小的方向走。==真正计算复杂的就是负类别。负采样法就是从V-1个负样本中随机挑几个词做负样本==。每个词被选为负样本的概率和其词频正相关00
	- Negative Sampling由于没有采用霍夫曼树，每次只是通过采样neg个不同的中心词做负例，利用这一个正例和neg个负例，我们进行二元逻辑回归，就可以训练模型，因此整个过程要比Hierarchical Softmax简单。二元逻辑回归算法见文档。
	- <font color='deeppink'>负采样中每个词有两套向量，分别作为输入和预测时使用。
3. <font color='deeppink'>两种解法进行一定优化，牺牲了一定的分类的准确度。比如负采样的负样本是随机选取的，所以相对已经没那么准了。
#### 1.2.2 基于Hierarchical Softmax的CBOW模型算法流程：
- 输入：根据词向量的维度大小M，以及CBOW的上下文大小2c，步长$\eta$，得到训练样本。
- 建立霍夫曼树，整体语料的各个词频 决定 huffman树。
- 随机初始化所有的模型参数$\theta$，所有的词向量w。这些训练样本所用的huffman树是一棵
- 随机梯度上升法，对于训练集中的每一个样本$(context(w), w)$中的每一个词向量$x_i$(共2c个)进行迭代更新。
- 如果梯度收敛，则结束梯度迭代，否则回到上一步继续迭代
$$h=\sum_{i=1}^{2c} embedding_{i}$$
$$y=softmax(d)=softmax(Wh)=\frac{1}{\sum_{i=1}^{V}e^{d_{i}}}\begin{bmatrix}
e^{d_{1}}\\ 
e^{d_{2}}\\ 
...\\
e^{d_{V}}\end{bmatrix}$$
W为全连接层参数，将词向量维度映射为V维（词表大小），表示预测词的概率。
#### 1.2.3 负采样方法
如果<font color='red'>词汇表的大小为$V$,那么我们就将一段长度为1的线段分成$V$份，每份对应词汇表中的一个词。</font>高频词对应的线段长，低频词对应的线段短(高频词数量多，分子count就大)。<font color='deeppink'>每个词$w$的线段长度</font>由下式决定：$$len(w) = \frac{count(w)}{\sum\limits_{u \in vocab} count(u)}$$

　　　　在word2vec中，分子和分母都取了3/4次幂（经验参数，提高低频词被选取的概率）如下：$$len(w) = \frac{count(w)^{3/4}}{\sum\limits_{u \in vocab} count(u)^{3/4}}$$

　　　　在采样前，我们将这段长度为1的线段划分成$M$等份，这里$M >> V$，这样可以保证每个词对应的线段都会划分成对应的小块。而M份中的每一份都会落在某一个词对应的线段上。在采样的时候，我们只需要==从$M$个位置中采样出$neg$个位置就行，此时采样到的每一个位置对应到的线段所属的词就是我们的负例词==
![在这里插入图片描述](https://img-blog.csdnimg.cn/3753e2f61df74e54878f6726356c0b50.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
**在word2vec中，$M$取值默认为$10^8$。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/68d66db9b92141d9addd2a4831098b61.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

### 1.3 总结：
1. one-hot：词表大大时内存不够。且所有词相似度都是一样的没有区别
2. word embedding：考虑使用使用神经网络语言模型，通过训练，将每个词都映射到一个较短的词向量上来
3. 神经网络语言模型的输入输出，有连续词袋模型CBOW(Continuous Bag-of-Words） 和Skip-Gram两种模型。
	- CBOW模型的训练输入是某个中心词的上下文词向量，输出是词表所有词的softmax概率，训练的目标是期望中心词对应的softmax概率最大。
	- Skip-Gram模型和CBOW的思路是反着来的，即输入中心词词向量，而输出是中心词对应的上下文词向量。比如窗口大小为4，就是输出softmax概率排前8的8个词。
4. word2vec有两种解法，霍夫曼树和负采样。负采样用得较多，因为构建霍夫曼树比较麻烦。
5. 一般来说， <font color='deeppink'>Skip-Gram模型比CBOW模型更好，因为：
	- Skip-Gram模型有更多的训练样本。Skip-Gram是一个词预测n个词，而CBOW是n个词预测一个词。
	- 误差反向更新中，CBOW是中心词误差更新n个周边词，这n个周边词被更新的力度是一样的。而Skip-Gram中，每个周边词都可以根据误差更新中心词，所以Skip-Gram是更细粒度的学习方法。
	- Skip-Gram效果更好（默认Skip-Gram模型）但是缺点就是训练次数更多，时间更长。 


## 二、fasttext
### 2.1、简介
&#8195;&#8195;fasttext是facebook开源的一个词向量与文本分类工具，在2016年开源，典型应用场景是“带监督的文本分类问题”。提供简单而高效的文本分类和表征学习的方法，性能比肩深度学习而且速度更快。

&#8195;&#8195;fastText的核心思想：==将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。这中间涉及到两个技巧：字符级n-gram特征的引入以及分层Softmax分类==。叠加词向量背后的思想就是传统的词袋法，即将文档看成一个由词构成的集合。

这些不同概念被用于两个不同任务：
•	有效文本分类 ：有监督学习（短文本）
•	学习词向量表征：无监督学习

### 2.2 FastText原理

fastText方法包含三部分，模型架构，层次SoftMax和N-gram特征。用词向量的叠加代表文档向量，全连接之后softmax分类。
#### 2.2.1 模型架构
fastText的架构和word2vec中的CBOW的架构类似，因为它们的作者都是Facebook的科学家Tomas Mikolov，而且确实fastText（2016）也算是words2vec（2014）所衍生出来的。
Continuous Bog-Of-Words： 

![在这里插入图片描述](https://img-blog.csdnimg.cn/4e880a15d1b14c82a7f1797676a87bf8.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f9891d5be1134c50aded04345f5e8b72.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
==隐藏层就是叠加后的句子（文档）向量==
参考[《理解文本分类利器fastText》](https://zhuanlan.zhihu.com/p/375614469)
- 序列中的词和词组组成特征向量，特征向量通过线性变换映射到中间层，中间层再映射到标签。 
- fastText 模型架构和 Word2Vec 中的 CBOW 模型很类似。不同之处在于，fastText 预测标签，而 CBOW 模型预测中间词。
- 所以fastText只有CBOW模型，对应fastText.train_supervised 没有model参数。 Word2Vec有两种模型，所以fastText.train_unsupervised可以选择model={cbow, skipgram} ，默认skipgram。

#### 2.2.2 层次SoftMax
层次softmax的基本思想是根据类别的频率构造霍夫曼树来代替扁平化的标准softmax。通过层次softmax，获得概率分布的时间复杂度可以从O(N)降至O(logN)。(多分类转成一系列二分类）

下图为层次softmax的一个具体示例：
![在这里插入图片描述](https://img-blog.csdnimg.cn/267e028268f346a184e211105c3fd76d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/eb9f5beaf0f14ec4b076c663037ad927.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

（见速通一书162页）
#### 2.2.3 N-gram特征
&#8195;&#8195; <font color='red'>n-gram解决词袋模型没有词序的问题，Hash解决n-gram膨胀问题。最大问题是有Hash冲突，但是实际中问题不大。

&#8195;&#8195; <font color='deeppink'>fastText 本身是词袋模型，为了分类的准确性，所以加入了 N-gram 特征提取词序信息。</font>“我 爱 她”如果加入 2-Ngram，第一句话的特征还有 “我-爱” 和 “爱-她”，这两句话 “我 爱 她” 和 “她 爱 我” 就能区别开来了。当然啦，为了提高效率，我们需要过滤掉低频的 N-gram。
&#8195;&#8195; <font color='deeppink'>n-gram的问题是词表会急剧扩大，变为$|V|^n$，没有机器扛得住。所以使用散列法（Hash）对n-gram特征进行压缩。</font>
&#8195;&#8195; Hash：使用Hash函数将字符串映射到某个整数。这样不管n-gram词表有多大，最后整数范围都是函数输出范围（比如4000亿词表。hash函数是对10526取余，最后输出就10526个数值，数值再转成向量）

#### 2.2.4 subword
- word2vec中每个词都是一个基本信息单元，不可再切分。忽略了词内部特征。fasttext采样子词模型表示词，可以从词的构造上学习词义，解决未登录词的问题。
- fasttext中子词的n-gram长度在minn和maxn之间。如果模型输入是ID之类的特征，子词没有任何意义，应取消子词。即minn=maxn=0。
- 中文中子词是两个相邻的字，英文中是词根和词缀。


#### 2.2.5 fasttext文本分类总结
- 一个句子进行分词，每个词进行embedding转换成一个词向量，默认100维。
- 每个词按位相加成一个新的100维向量。再过一个全连接矩阵，100行(词向量维度)22列（分类数）
- 经过softmax得到每一类的类别概率。

## 三、fastText和word2vec对比总结
### 3.1 fastText和word2vec的区别
相似处：
1.图模型结构很像，都是采用embedding向量的形式，得到word的隐向量表达。
2.都采用很多相似的优化方法，比如使用Hierarchical softmax优化训练和预测中的打分速度。

不同处：
==word2vec用词预测词，而且是词袋模型，没有n-gram。fasttext用文章/句子词向量预测类别，加入了n-gram信息==。所以有：
1.	模型的输入层：word2vec的输入层，是 context window 内的词；而fasttext 对应的整个sentence的内容，包括<font color='deeppink'>word、n-gram、subword。
2.	模型的输出层：word2vec的输出层，计算某个词的softmax概率最大；而fasttext的输出层对应的是 分类的label；
3.	两者本质的不同，体现在 h-softmax的使用：
	- word2vec用的负采样或者霍夫曼树解法（计算所有词概率，类别过大）。
	-  <font color='deeppink'>fasttext用的softmsx全连接分类（类别少）
4. word2vec主要目的的得到词向量，该词向量 最终是在输入层得到（不关注预测的结果准不准，因为霍夫曼树和负采样解法虽然优化了训练速度，但是分类结果没那么准了）。fasttext主要是做分类 ，虽然也会生成一系列的向量，但最终都被抛弃，不会使用。
5. word2vec有两种模型cbow和 skipgram，fasttext只有cbow模型。
6. word2vec属于监督模型，但是不需要标注样本。fasttext也属于监督模型，但是需要标注样本。


### 3.2 小结
#### 3.2.1 fasttext适用范围
总的来说，fastText的学习速度比较快，效果还不错。
1. ==fastText适用与分类类别比较大而且数据集足够多的情况，当分类类别比较小或者数据集比较少的话，很容易过拟合==。
2. 适用于短文本。因为第一步是多个向量相加，文本越长，高频词越多，最后相加结果越趋于相同。（比如关键词只有那么几个，如果长文本词向量相加，关键词就被淹没了）如果非要用于长文本分类，就先去停用词或者干脆提取关键词（这个软件没有分开计算词的权重）
#### 3.2.2 fasttext应用场景
1. 可以完成无监督的词向量的学习，可以学习出来词向量，来保持住词和词之间，相关词之间是一个距离比较近的情况；
2. 也可以用于有监督学习的文本分类任务，（新闻文本分类，垃圾邮件分类、情感分析中文本情感分析，电商中用户评论的褒贬分析）
3. 封装的特别好，用了很多加速模块包括多线程实现。非常简单。Keras可以做模型，定制化，很灵活，但是需要自己搭。Fasttext任务单一，用起来方便。
#### 3.2.3 fastText优点
fastText是一个快速文本分类算法，与基于神经网络的分类算法相比有两大优点：
1. fastText在保持高精度的情况下加快了训练速度和测试速度
2. fastText不需要预训练好的词向量，fastText会自己训练词向量
3. fastText两个重要的优化：Hierarchical Softmax、N-gram
训练代码中，如果电脑一开始训练就卡了，可以设置线程thread=2。（卡住只能kill进程ps - aux│grep python,kill – 9 1531(进程数）

fasttext已经嵌入word2vec，可以用它做有监督和无监督（就是word2vec）。涉及到离散特征都可以用fasttext。比如招聘网站预测求职者和职位的匹配度。（求职者和职位分别提取关键词特征，然后用fasttext训练，输出录用和不录用的概率。但是求职者简历写本科就是本科学位，职位要求的本科是指本科及以上。二者还是有些不一样。需要把求职者关键字/标签加P，职位标签加J予以区分。即当数据来源不同纬度时，语义可能不同，前面加一个field予以区分）

## 四、用gensim学习word2vec
参考文档[《word2vec原理和gensim实现》](https://maxiang.io/note/#%E4%B8%80-cbow%E4%B8%8Eskip-gram%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80)
### 4.1 使用技巧
1. 用哪种方法看需求：
      1.使用时需要将多个向量相加（文本向量化） 用cbow
      2.使用时都是单个词向量使用（找近义词） 用skip-gram
     大原则：使用的过程和训练的过程越一致 ，效果一般越好
如果实在不知道怎么选，一般来说<font color='deeppink'>skip-gram+ns负采样效果好一点点。</font>

2. <font color='deeppink'>同一批词分别进行两次训练，embedding也不在同一语义空间，不同语义空间的向量没有可比性。word2vec不能进行增量更新，有新词只能全量训练，因为语料库变了one-hot也变了，V也变了。</font>
3. 孤岛效应：有一堆词，明明不相关，训练出来确是显示相似的。
	- 某部分词总是一起出现，另一堆词也是一起出现，但是这两堆词互相没有任何交集，虽然在一起训练是一个向量空间，但实际上是两个向量空间。这两堆词互相比较是没有意义的。
	- 孤岛效应本质是由一些不相关语料或者弱相关语料组成。Word2vec本身不能解决这个问题，这个只能在样本选取上下功夫，让训练样本尽可能相关。==所以各领域自己训练自己的，不要把一堆不相关的东西放到一起训练。几个行业几套词向量==。
### 4.2 推荐系统中的Word2vec
- word2vec可以计算向量之间的相似度，所以可以在其它领域广泛使用。比如视频分类
- nlp和推荐系统中最大区别是nlp的词向量比较固定，而推荐系统中用户不断推陈出新，用户向量变化很快。
- 可以使用Hash技术，将用户ID(如手机设备号）进行hash作为类别。
- 将视频ID作为词，用户的点击序列作为句子（一连串视频），用word2vec对点击序列进行训练。最后每个视频ID对应一个embedding，用来计算不同视频的相似度，或者作为视频向量输入后续模型。
## 五、 基于fastText实现文本分类

直接pip安装报错：“Microsoft Visual C++ 14.0 or greater is required”。在[此页面](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)下载fasttext文件，然后安装：pip install C:\Users\LS\Downloads\fasttext-0.9.2-cp38-cp38-win_amd64.whl

FastText可以快速的在CPU上进行训练，最好的实践方法就是[github教程](https://github.com/facebookresearch/fastText/tree/master/python)，以及[官网教程](https://fasttext.cc/docs/en/cheatsheet.html)。
### 5.1 fasttext参数：
参考官方文档[《Python模块》](https://fasttext.cc/docs/en/python-module.html)、[《FastText代码详解》](https://zhuanlan.zhihu.com/p/52154254?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_campaign=shareopn)
- FUNCTIONS
    load_model(path)：加载给定文件路径的模型并返回模型对象。
    read_args(arg_list, arg_dict, arg_names, default_values)
    tokenize(text)：给定一串文本，对其进行标记并返回一个标记列表
    ==train_supervised(*kargs, **kwargs)：监督训练，样本包含标签，即fasttext==。
     train_unsupervised(*kargs, **kwargs)：无监督训练，样本没有标签，即word2vec。
    

- fasttext.train_unsupervised函数：调用此函数学习词向量，即word2vec模型。
	- 维度 ( dim ) ：向量维度的大小，defult=100 ，也可以选100-300 。
	- 子词是包含在最小大小 ( minn ) 和最大大小 ( maxn )之间的单词中的所有子字符串。默认minn=3， maxn=6。
	- minn和maxn分别代表subwords的最小长度和最大长度
	- bucket表示可容纳的subwords和wordNgrams的数量，可以理解成是它们存放的表，与word存放的表是分开的。
	- t表示过滤高频词的阈值，像"the"，"a"这种高频但语义很少的词应该过滤掉。
```python
input             # training file path (required)
model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
lr                # 学习率 [0.05]
dim               # 词向量维度 [100]
ws                # 上下文窗口大小 [5]
epoch             # 训练轮数 [5]
minCount          # 最少单词词频，过滤过少的单词 [5]
minn              # min length of char ngram [3]
maxn              # max length of char ngram [6]
neg               # 负采样个数 [5]
wordNgrams        # 词ngram最大长度 [1]
loss              # loss function {ns, hs, softmax, ova}[ns]
                  #（负采样、霍夫曼树、softmax和多分类采用多个二分类计算，即loss one-vs-all） 
bucket            # number of buckets，放的是subwords [2000000]
thread            # cpu线程 [number of cpus]
lrUpdateRate      # change the rate of updates for the learning rate，实现阶梯动态学习率 [100]
t                 # sampling threshold，过滤高频词，越大被保留的概率越大 [0.0001]
verbose           # verbose [2]
```


train_supervised 参数：

```python
input             # training file path (required)
lr                # 学习率 [0.05]
dim               # 词向量维度 [100]
ws                # 上下文窗口大小 [5]
epoch             # 训练轮数 [5]
minCount          # 最小词频 [1]
minCountLabel     # minimal number of label occurences [1]
minn              # min length of char ngram [0]
maxn              # max length of char ngram [0]
neg               # 负采样个数 [5]
wordNgrams        # n-gram [1]
loss              # loss function {ns, hs, softmax, ova} [softmax]
bucket            # number of buckets [2000000]
thread            # cpu线程数 [number of cpus]
lrUpdateRate      # change the rate of updates for the learning rate [100]
t                 # sampling threshold [0.0001]
label             # 标签前缀 ['__label__']
verbose           # verbose [2]
pretrainedVectors # 从 (.vec file)加载预训练的词向量，用于监督训练 []
```
model属性
 

```python
get_dimension           # 获取向量（隐藏层）的维度（大小）.这等价于 `dim` 属性           
get_input_vector        # 给定一个索引，得到输入矩阵对应的向量 
get_input_matrix        # 获取模型的完整输入矩阵的副本
get_labels              # 获取字典的整个标签列表，这相当于 `labels` 属性。
get_line                # 将一行文本拆分为单词和标签
get_output_matrix       # 获取模型的完整输出矩阵的副本。
get_sentence_vector     # 给定一个字符串，获得向量表示。这个函数
                        # assumes to be given a single line of text. We split words on
                        # whitespace (space, newline, tab, vertical tab) and the control
                        # characters carriage return, formfeed and the null character.
get_subword_id          # 给定一个subword，获取字典中的词 id hashes to.
get_subwords            # 给定一个词，获取子词及其索引。
get_word_id             # 给定一个词，获取字典中的词 id
get_word_vector         # 获取训练好的词向量。
get_words               # 获取字典的整个单词列表，这相当于 `words` 属性。
is_quantized            # 模型是否已经量化过
predict                 # 给定一个字符串，得到一个标签列表和一个对应概率列表
quantize                # 量化模型，减少模型的大小和内存占用
save_model              # 保存模型
test                    # Evaluate supervised model using file given by path
test_label              # 返回每个标签的准确率和召回率。
```

### 5.2 基本使用
当 fastText 运行时，进度和预计完成时间会显示在您的屏幕上。训练完成后，model变量包含有关训练模型的信息，可用于查询：

```python
import fasttext
model = fasttext.train_unsupervised('data/fil9')#维基百科文件
model.words

[u'the', u'of', u'one', u'zero', u'and', u'in', u'two', u'a', u'nine', u'to', u'is', ...
```

==获得词向量==：（它返回词汇表中的所有单词，按频率递减排序。）

```python
model.get_word_vector("the")
array([-0.03087516,  0.09221972,  0.17660329,  0.17308897,  0.12863874,
        0.13912526, -0.09851588,  0.00739991,  0.37038437, -0.00845221,
        ...
       -0.21184735, -0.05048715, -0.34571868,  0.23765688,  0.23726143],
      dtype=float32)
```
==保存模型（二进制），后续加载==
```python
 model.save_model("result/fil9.bin")
 model = fasttext.load_model("result/fil9.bin")
```
cobw和skipgram：
```python
import fasttext
model = fasttext.train_unsupervised('data/fil9', "cbow")
```
==预测结果==
```python
#读取测试集，预测模型输出
test_df=pd.read_csv('./train_set.csv',sep='\t',nrows=10000)
results=[model.predict(x)  for x in test_df['text']]
results

[(('__label__2',), array([0.99827653])),
 (('__label__11',), array([0.84706676])),
 (('__label__3',), array([0.99988556])),
 (('__label__2',), array([0.99980879])),

...
(('__label__2',), array([0.9998678])),
 (('__label__1',), array([0.87650901])),
 (('__label__3',), array([1.00001013])),
 ...]
```
所以输出结果是带前缀的标签和分类概率。想只得到类别，可以这样写：

```python
result=[model.predict(x)[0][0].split('__')[-1] for x in test_df['text']]
result
['2',
 '11',
 '3',
 '2',
 '3',
 '9',
 '3',
 '10',
 '12',
 '3',
 '0',
...]
```
### 5.3 bin格式词向量转换为vec格式
参考[《fasttext训练的bin格式词向量转换为vec格式词向量》](https://blog.csdn.net/huyidu/article/details/112712526?utm_source=app&app_version=4.16.0&code=app_1562916241&uLinkId=usr1mkqgl919blen)

```python
#加载的fasttext预训练词向量都是vec格式的，但fasttext无监督训练后却是bin格式，因此需要进行转换
# 以下代码为fasttext官方推荐：
# 请将以下代码保存在bin_to_vec.py文件中
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division, absolute_import, print_function

from fasttext import load_model
import argparse
import errno

if __name__ == "__main__":
    # 整个代码逻辑非常简单
    # 以bin格式的模型为输入参数
    # 按照vec格式进行文本写入
    # 可通过head -5 xxx.vec进行文件查看
    parser = argparse.ArgumentParser(
        description=("Print fasttext .vec file to stdout from .bin file")
    )
    parser.add_argument(
        "model",
        help="Model to use",
    )
    args = parser.parse_args()

    f = load_model(args.model)
    words = f.get_words()
    print(str(len(words)) + " " + str(f.get_dimension()))
    for w in words:
        v = f.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        try:
            print(w + vstr)
        except IOError as e:
            if e.errno == errno.EPIPE:
                pass
```

```python
# 打开cmd，在bin_to_vec.py路径下执行该命令，生成unsupervised_data.vec
python bin_to_vec.py word15000.bin > word15000.vec
```

==在实践中，我们观察到 skipgram 模型在处理子词信息方面比 cbow 更好==
## 六、新闻文本分类——fasttext
### 6.1 正常fasttext分类
单纯的fasttext分类，参数用讨论区默认参数，没有调整。分数0.9151。
fasttext训练很快，大概十来分钟吧。
```python
import pandas as pd
train_df=pd.read_csv('./train_set.csv',sep='\t')
train_df['label_ft']='__label__'+train_df['label'].astype(str)
train_df[['text','label_ft']].to_csv('./train.csv',index=None,header=None,sep='\t')

import fasttext
model=fasttext.train_supervised('./train.csv',lr=1.0,wordNgrams=2, 
verbose=2,minCount=1,epoch=25,loss="hs")

test_df=pd.read_csv('./test_a.csv',sep='\t')
result=[model.predict(x)[0][0].split('__')[-1] for x in test_df['text']]
result[:100]

pd.DataFrame({'label':result}).to_csv('fasttext.csv',index=None)
```
最终上传，得分0.9151。
调整部分参数后，最终得分0.9358。
```python
model=fasttext.train_supervised('./train.csv',lr=0.8,wordNgrams=3, 
verbose=2,minCount=1,epoch=25,loss="softmax")
```


### 6.2 小数据集：word2vec+fasttext+首尾截断
首先拿15000条数据进行试验，前10000条fasttext训练，后5000条测试，代码见讨论区：[《Task4 基于深度学习的文本分类1-fastText》](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.12.64063dadaDY31g&postId=118255)（其实就是上面代码改了点数据集）：
1. 试验正常fasttext效果，f1 score=0.8272
```python
import pandas as pd
from sklearn.metrics import f1_score

# 转换为FastText需要的格式
train_df = pd.read_csv('../data/train_set.csv', sep='\t', nrows=15000)
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-5000].to_csv('train.csv', index=None, header=None, sep='\t')

import fasttext
model = fasttext.train_supervised('train.csv', lr=1.0, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=25, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
```
2. 试验word2vec+fasttext效果，f1 score=0.8426
```python
#先进行word2vec训练，含全部15000条数据
train_df[['text','label_ft']].to_csv('train15000.csv', index=None, header=None, sep='\t')
model1 = fasttext.train_unsupervised('train15000.csv', lr=0.1, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=8, loss="hs")
#保存模型转为词向量
model1.save_model("word15000.bin")
#cmd命令行执行python bin_to_vec.py result1000.bin < result1000.vec，转换为vec词向量
```

```python
#fasttext进行训练，词向量为前一步训练好的词向量，训练数据为10000条
model2 = fasttext.train_supervised('train.csv',pretrainedVectors='word15000.vec',lr=1.0, wordNgrams=2, 
#                                  verbose=2, minCount=1, epoch=16, loss="hs")
#预测结果
val_pred = [model2.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
```
3. 试验首尾截断效果，f1 score=0.8222(首尾各50词），0.8304（首尾各100词）
```python
#首尾截断实验效果
#准备将text文本首尾截断，各取100tokens
def slipt2(x):
  ls=x.split(' ')
  le=len(ls)
  if le<201:
    return x
  else:
    return ' '.join(ls[:100]+ls[-100:])
    
trains_df['summary']=trains_df['text'].apply(lambda x:slipt2(x))
train_df[['summary','label_ft']].iloc[:-5000].to_csv('trains_summary10000.csv', index=None, header=None, sep='\t')

model3 = fasttext.train_supervised('trains_summary10000.csv',pretrainedVectors='word15000.vec',lr=1.0, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=16, loss="hs")
#预测结果
val_pred = [model3.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
```

### 6.3 全数据集：word2vec+fasttext+首尾截断
1. 数据处理
```python
#读取训练测试集数据
import pandas as pd
from sklearn.metrics import f1_score

# 转换为FastText需要的格式
train_df = pd.read_csv('./train_set.csv', sep='\t')
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].to_csv('train_20w.csv', index=None, header=None, sep='\t')

test_df = pd.read_csv('./test_a.csv', sep='\t')
df=pd.concat([train_df,test_df])
df[['text']].to_csv('train_25w.csv', index=None, header=None, sep='\t')
```
2. 用word2vec进行train+test数据的词向量训练，这一步花了2个小时。
```python
import fasttext

model1 = fasttext.train_unsupervised('train_25w.csv', lr=0.1, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=8, loss="hs")

model1.save_model("word_25w.bin")
#cmd下运行python bin_to_vec.py word_25w.bin > word_25w.vec
```
3. fasttext进行有监督训练，相当于分类微调。最终上传，得分0.9162，吐血。
```python
model2=fasttext.train_supervised('train_20w.csv',pretrainedVectors='word_25w.vec',lr=0.8, wordNgrams=2, verbose=2, minCount=1, epoch=18, loss="hs")

import pandas as pd
test_df = pd.read_csv('./test_a.csv', sep='\t')
test_pred = [model2.predict(x)[0][0].split('__')[-1] for x in test_df['text']]

pd.DataFrame({'label':test_pred}).to_csv('word_fast.csv',index=None)
```
4. 接下来进行首尾截断测试：

```python
#首尾截断进行训练
train_df = pd.read_csv('./train_set.csv', sep='\t')
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df['summary']=train_df['text'].apply(lambda x:slipt2(x))
train_df[['summary','label_ft']].to_csv('train_summary_20w.csv', index=None, header=None, sep='\t')

model3 = fasttext.train_supervised('train_summary_20w.csv',pretrainedVectors='word_25w.vec',lr=0.8, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=18, loss="hs")
#预测结果
test_df['summary']=test_df['text'].apply(lambda x:slipt2(x))
test_pred = [model3.predict(x)[0][0].split('__')[-1] for x in test_df['summary']]
pd.DataFrame({'label':test_pred}).to_csv('word_fast_cut.csv',index=None)
```
最终得分0.9203，至少证明了长文本分类，数据集够多的时候，进行部分截断比较好。

数据量    | fasttext|word2vec+fasttext|word2vec+fasttext+首尾截断
-------- | -----| ----- | -----
10000+5000  | 0.8272|0.8426|0.8304
20w+5w   |0.9151（没调参）|0.9162（没调参）|0.9203（没调参）
20w+5w    | 0.9358（已调参）||0.9421（已调参）
截断比不截断高0.4-0.6个点。

5. 下面是部分调参记录
继续首尾截断试验，训练集前19w为悬链数据，最后1w为测试数据。

首尾截断     | f1|loss|n-gram
-------- | ----- |-------- | -----
各30，同时epoch=18，lr=0.8，下同 | 0.9190 |hs|2
各30 | 0.9352 |softmax|2
各30 | 0.9388 |softmax|3
各30 | 0.9382 |softmax|4
各30 |  |softmax|5
各30，同时epoch=18，lr=0.5 |  |softmax|4
各30，同时epoch=27，lr=0.5 |  |softmax|4
-----|----- |-----|-----
各50  | 0.9192 |hs|2
各80  | 0.9170 |hs|2
各100  | 0.9200/0.9184 |hs|2
各150  | 0.9226 |hs|2
各150 | 0.9371 |softmax|2
各150 | 0.9436 |softmax|3
各150 | 0.9417 |softmax|4
各200  | 0.9212 |hs|2
不截断 | 0.9158 |hs|2
不截断，加和平均的词向量太多，无用信息冲淡了关键信息。
fasttext分类的loss必须选择softmex，不需要hs和ng，因为类别少。
n-gram中，n增大可以表示一部分词序，有利于文本表征。但是太大的话，词向量和n-gram向量太多，分类效果也不好（参数过多学不好或者是无用信息过多）。

初步选择以下参数：

```python
#首尾截断各150个词
model3=fasttext.train_supervised('train_summary_20w.csv',pretrainedVectors='word_25w.vec',
lr=0.8,wordNgrams=3,verbose=2,minCount=1,epoch=18,loss="softmax")
```
最终分数f1=0.9421。
