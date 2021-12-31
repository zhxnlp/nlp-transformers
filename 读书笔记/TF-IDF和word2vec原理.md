@(NLP)
## 文本挖掘
@[toc]
###(一) 文本挖掘的分词原理
　　　　在做文本挖掘的时候，首先要做的预处理就是分词。英文单词天然有空格隔开容易按照空格分词，但是也有时候需要把多个单词做为一个分词，比如一些名词如“New York”，需要做为一个词看待。而中文由于没有空格，分词就是一个需要专门去解决的问题了。无论是英文还是中文，分词的原理都是类似的，本文就对文本挖掘时的分词原理做一个总结。

#### 1. 分词的基本原理
　　　　现代分词都是基于统计的分词，而统计的样本内容来自于一些标准的语料库。假如有一个句子：“小明来到荔湾区”，我们期望语料库统计后分词的结果是："小明/来到/荔湾/区"，而不是“小明/来到/荔/湾区”。那么如何做到这一点呢？

　　　　从统计的角度，我们期望"小明/来到/荔湾/区"这个分词后句子出现的概率要比“小明/来到/荔/湾区”大。如果用数学的语言来说说，如果有一个句子$S$,它有m种分词选项如下：$$A_{11}A_{12}...A_{1n_1}$$$$A_{21}A_{22}...A_{2n_2}$$$$......  ......$$$$A_{m1}A_{m2}...A_{mn_m}$$

　　　　其中下标$n_i$代表第$i$种分词的词个数。如果我们从中选择了最优的第$r$种分词方法，那么这种分词方法对应的统计分布概率应该最大，即：$$r = \underbrace{arg\;max}_iP(A_{i1},A_{i2},...,A_{in_i}) $$

　　　　但是我们的概率分布$P(A_{i1},A_{i2},...,A_{in_i})$并不好求出来，因为它涉及到$n_i$个分词的联合分布。在NLP中，为了简化计算，我们通常使用<font color='red'>马尔科夫假设，即每一个分词出现的概率仅仅和前一个分词有关</font>，即：$$P(A_{ij}|A_{i1},A_{i2},...,A_{i(j-1)}) = P(A_{ij}|A_{i(j-1)})$$

　　　　在前面我们讲MCMC采样时，也用到了相同的假设来简化模型复杂度。使用了马尔科夫假设，则我们的联合分布就好求了，即：$$P(A_{i1},A_{i2},...,A_{in_i}) = P(A_{i1})P(A_{i2}|A_{i1})P(A_{i3}|A_{i2})...P(A_{in_i}|A_{i(n_i-1)})$$

　　　　而通过我们的标准语料库，我们可以近似的计算出所有的分词之间的二元条件概率，比如任意两个词$w_1,w_2$，它们的条件概率分布可以近似的表示为：$$P(w_2|w_1) = \frac{P(w_1,w_2)}{P(w_1)} \approx \frac{freq(w_1,w_2)}{freq(w_1)}$$$$P(w_1|w_2) = \frac{P(w_2,w_1)}{P(w_2)} \approx \frac{freq(w_1,w_2)}{freq(w_2)}$$

　　　　<font color='red'>其中$freq(w_1,w_2)$表示$w_1,w_2$在语料库中相邻一起出现的次数</font>，而其中$freq(w_1),freq(w_2)$分别表示$w_1,w_2$在语料库中出现的统计次数。

　　　　利用语料库建立的统计概率，对于一个新的句子，我们就可以通过计算各种分词方法对应的联合分布概率，找到最大概率对应的分词方法，即为最优分词。

#### 2. N元模型
　　　　当然，你会说，只依赖于前一个词太武断了，我们能不能依赖于前两个词呢？即：$$P(A_{i1},A_{i2},...,A_{in_i}) = P(A_{i1})P(A_{i2}|A_{i1})P(A_{i3}|A_{i1}，A_{i2})...P(A_{in_i}|A_{i(n_i-2)}，A_{i(n_i-1)})$$

　　　　这样也是可以的，只不过这样联合分布的计算量就大大增加了。我们一般称<font color='red'>只依赖于前一个词的模型为二元模型(Bi-Gram model)</font>，而依赖于前两个词的模型为三元模型。以此类推，我们可以建立四元模型，五元模型,...一直到通用的$N$元模型。越往后，概率分布的计算复杂度越高。当然算法的原理是类似的。

　　　　在实际应用中，$N$一般都较小，一般都小于4，主要原因是<font color='red'>N元模型概率分布的空间复杂度为$O(|V|^N)$，其中$|V|$为语料库大小</font>，而$N$为模型的元数，当$N$增大时，复杂度呈指数级的增长。(二元模型前后两个词都有V种选择)常用汉字三四千，但是常用词是20w，二元模型就是400亿可能。

　　　　$N$元模型的分词方法虽然很好，但是要在实际中应用也有很多问题，首先，某些生僻词，或者相邻分词联合分布在语料库中没有，概率为0。这种情况我们一般会使用　<font color='red'>拉普拉斯平滑，即给它一个较小的概率值，</font>这个方法在朴素贝叶斯算法原理小结也有讲到。第二个问题是如果句子长，分词有很多情况，计算量也非常大，这时我们可以用下一节维特比算法来优化算法时间复杂度。

#### 3. 维特比算法与分词
　　　　为了简化原理描述，我们本节的讨论都是以二元模型为基础。

　　　　对于一个有很多分词可能的长句子，我们当然可以用暴力方法去计算出所有的分词可能的概率，再找出最优分词方法。但是用维特比算法可以大大简化求出最优分词的时间。

　　　　大家一般知道维特比算法是用于隐式马尔科夫模型HMM解码算法的，但是它是一个通用的求序列最短路径的方法，不光可以用于HMM，也可以用于其他的序列最短路径算法，比如最优分词。

　　　　维特比算法采用的是动态规划来解决这个最优分词问题的，动态规划要求局部路径也是最优路径的一部分，很显然我们的问题是成立的。首先我们看一个简单的分词例子："人生如梦境"。它的可能分词可以用下面的概率图表示：
[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-x9KdSdW0-1631636573467)(./1624897703642.png)]
　　　　图中的箭头为通过统计语料库而得到的对应的各分词位置BEMS（开始位置，结束位置，中间位置，单词）的条件概率。比如P(生|人)=0.17。有了这个图，维特比算法需要找到从Start到End之间的一条最短路径。对于在End之前的任意一个当前局部节点，我们需要得到到达该节点的最大概率$\delta$，和记录到达当前节点满足最大概率的前一节点位置$\Psi$。

　　　　我们先用这个例子来观察维特比算法的过程。首先我们初始化有：$$\delta(人) = 0.26\;\;\Psi(人)=Start\;\;\delta(人生) = 0.44\;\;\Psi(人生)=Start$$

　　　　对于节点"生"，它只有一个前向节点，因此有：$$\delta(生) = \delta(人)P(生|人) = 0.0442 \;\; \Psi(生)=人 $$

 　　　　对于节点"如"，就稍微复杂一点了，因为它有多个前向节点，我们要计算出到“如”概率最大的路径：$$\delta(如) = max\{\delta(生)P(如|生)，\delta(人生)P(如|人生)\} = max\{0.01680, 0.3168\} = 0.3168 \;\; \Psi(如) = 人生 $$

　　　　类似的方法可以用于其他节点如下：$$\delta(如梦) = \delta(人生)P(如梦|人生) = 0.242 \;\; \Psi(如梦)=人生 $$$$\delta(梦) = \delta(如)P(梦|如) = 0.1996 \;\; \Psi(梦)=如 $$$$\delta(境) = max\{\delta(梦)P(境|梦) ,\delta(如梦)P(境|如梦)\}= max\{0.0359, 0.0315\} = 0.0359 \;\; \Psi(境)=梦 $$$$\delta(梦境) = \delta(如)P(梦境|如) = 0.1616 \;\; \Psi(梦境)=如 $$

　　　　最后我们看看最终节点End:$$\delta(End) = max\{\delta(梦境)P(End|梦境), \delta(境)P(End|境)\} = max\{0.0396, 0.0047\} = 0.0396\;\;\Psi(End)=梦境$$

　　　　由于最后的最优解为“梦境”，现在我们开始用$\Psi$反推:$$\Psi(End)=梦境 \to \Psi(梦境)=如 \to \Psi(如)=人生 \to \Psi(人生)=start $$

　　　　从而最终的分词结果为"人生/如/梦境"。是不是很简单呢。

　　　　由于维特比算法我会在后面讲隐式马尔科夫模型HMM解码算法时详细解释，这里就不归纳了。

#### 4. 常用分词工具
　　　　对于文本挖掘中需要的分词功能，一般我们会用现有的工具。简单的英文分词不需要任何工具，通过空格和标点符号就可以分词了，而进一步的英文分词推荐使用nltk。对于中文分词，则推荐用结巴分词（jieba）。这些工具使用都很简单。你的分词没有特别的需求直接使用这些分词工具就可以了。

　　　　分词是文本挖掘的预处理的重要的一步，分词完成后，我们可以继续做一些其他的特征工程，比如向量化（vectorize），TF-IDF以及Hash trick，这些我们后面再讲。

### （二）文本挖掘预处理之向量化与Hash Trick
　　　　在文本挖掘的分词原理中，我们讲到了文本挖掘的预处理的关键一步：“分词”，而在做了分词后，如果我们是做文本分类聚类，则后面关键的特征预处理步骤有向量化或向量化的特例Hash Trick，本文我们就对向量化和特例Hash Trick预处理方法做一个总结。

#### 1. 词袋模型
　　　　在讲向量化与Hash Trick之前，我们先说说词袋模型(Bag of Words,简称BoW)。<font color='red'>词袋模型假设我们不考虑文本中词与词之间的上下文关系，仅仅只考虑所有词的权重。而权重与词在文本中出现的频率有关。

　　　　词袋模型首先会进行分词，在分词之后，通过统计每个词在文本中出现的次数，我们就可以得到该文本基于词的特征，如果<font color='red'>将各个文本样本的这些词与对应的词频放在一起，就是我们常说的向量化。</font>向量化完毕后一般也会使用<font color='red'>TF-IDF进行特征的权重修正，再将特征进行标准化</font>。 再进行一些其他的特征工程后，就可以将数据带入机器学习算法进行分类聚类了。

　　　　<font color='deeppink'>总结下词袋模型的三部曲：分词（tokenizing），TF-IDF修订词特征值（counting）与标准化（normalizing）。

　　　　与词袋模型非常类似的一个模型是词集模型(Set of Words,简称SoW)，和词袋模型唯一的不同是它仅仅考虑词是否在文本中出现，而不考虑词频。也就是一个词在文本在文本中出现1次和多次特征处理是一样的。在大多数时候，我们使用词袋模型，后面的讨论也是以词袋模型为主。

　　　　当然，词袋模型有很大的局限性，因为它仅仅考虑了词频，没有考虑上下文的关系，因此会丢失一部分文本的语义。但是大多数时候，<font color='deeppink'>如果我们的目的是分类聚类，则词袋模型表现的很好。

#### 2. 词袋模型之向量化
　　　　在词袋模型的统计词频这一步，我们会得到该文本中所有词的词频，有了词频，我们就可以用词向量表示这个文本。这里我们举一个例子，例子直接用scikit-learn的CountVectorizer类来完成，这个类可以帮我们完成文本的词频统计与向量化，代码如下：

　　　　完整代码参见我的github:https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/hash_trick.ipynb

	from sklearn.feature_extraction.text import CountVectorizer  
	vectorizer=CountVectorizer()
	corpus=["I come to China to travel", 
	    "This is a car polupar in China",          
	    "I love tea and Apple ",   
	    "The work is to write some papers in science"] 
	print vectorizer.fit_transform(corpus)
　　　　我们看看对于上面4个文本的处理输出如下：

	  (0, 16)	1
	  (0, 3)	1
	  (0, 15)	2
	  (0, 4)	1
	  (1, 5)	1
	  (1, 9)	1
	  (1, 2)	1
	  (1, 6)	1
	  (1, 14)	1
	  (1, 3)	1
	  (2, 1)	1
	  (2, 0)	1
	  (2, 12)	1
	  (2, 7)	1
	  (3, 10)	1
	  (3, 8)	1
	  (3, 11)	1
	  (3, 18)	1
	  (3, 17)	1
	  (3, 13)	1
	  (3, 5)	1
	  (3, 6)	1
	  (3, 15)	1
　　　　可以看出4个文本的词频已经统计出，在输出中，左边的括号中的第一个数字是文本的序号，第2个数字是词的序号，注意词的序号是基于所有的文档的。第三个数字就是我们的词频。

　　　　我们可以进一步看看每个文本的词向量特征和各个特征代表的词，代码如下：

	print vectorizer.fit_transform(corpus).toarray()
	print vectorizer.get_feature_names()
　　　　输出如下：
	
	[[0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0]
	 [0 0 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0]
	 [1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0]
	 [0 0 0 0 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1]]
	[u'and', u'apple', u'car', u'china', u'come', u'in', u'is', u'love', u'papers', u'polupar', u'science', u'some', u'tea', u'the', u'this', u'to', u'travel', u'work', u'write']
　　　　可以看到我们一共有19个词，所以4个文本都是19维的特征向量。而每一维的向量依次对应了下面的19个词。另外由于词"I"在英文中是停用词，不参加词频的统计。

　　　　由于大部分的文本都只会使用词汇表中的很少一部分的词，因此我们的词向量中会有大量的0。也就是说<font color='deeppink'>词向量是稀疏的。在实际应用中一般使用稀疏矩阵来存储。

　　　　将文本做了词频统计后，我们一般会通过TF-IDF进行词特征值修订，这部分我们后面再讲。

　　　　向量化的方法很好用，也很直接，但是在有些场景下很难使用，比如分词后的词汇表非常大，达到100万+，此时如果我们直接使用向量化的方法，将对应的样本对应特征矩阵载入内存，有可能将内存撑爆，在这种情况下我们怎么办呢？第一反应是我们要进行特征的降维，说的没错！而<font color='deeppink'>Hash Trick就是非常常用的文本特征降维方法。

#### 3.  Hash Trick
　　　　在大规模的文本处理中，由于特征的维度对应分词词汇表的大小，所以维度可能非常恐怖，此时需要进行降维，不能直接用我们上一节的向量化方法。而最常用的文本降维方法是Hash Trick。说到Hash，一点也不神秘，学过数据结构的同学都知道。这里的Hash意义也类似。

　　　　在Hash Trick里，我们会定义一个特征Hash后对应的哈希表的大小，这个哈希表的维度会远远小于我们的词汇表的特征维度，因此可以看成是降维。具体的方法是，对应任意一个特征名，我们会用Hash函数找到对应哈希表的位置，然后将该特征名对应的词频统计值累加到该哈希表位置。如果用数学语言表示,假如<font color='deeppink'>哈希函数$h$使第$i$个特征哈希到位置$j$,即$h(i)=j$,则第$i$个原始特征的词频数值$\phi(i)$将累加到哈希后的第$j$个特征的词频数值$\bar{\phi}$上，即：$$\bar{\phi}(j) = \sum_{i\in \mathcal{J}; h(i) = j}\phi(i)$$

　　　　其中$\mathcal{J}$是原始特征的维度。

　　　　但是上面的方法有一个问题，有可能两个原始特征的哈希后位置在一起导致词频累加特征值突然变大，为了解决这个问题，出现了hash Trick的变种signed hash trick,此时除了哈希函数$h$,我们多了一个一个哈希函数：$$\xi : \mathbb{N} \to {\pm 1}$$

　　　　此时我们有$$\bar{\phi}(j) = \sum_{i\in \mathcal{J}; h(i) = j}\xi(i)\phi(i)$$

　　　　这样做的好处是，哈希后的特征仍然是一个无偏的估计，不会导致某些哈希位置的值过大。

　　　　当然，大家会有疑惑，这种方法来处理特征，哈希后的特征是否能够很好的代表哈希前的特征呢？从实际应用中说，由于文本特征的高稀疏性，这么做是可行的。如果大家对理论上为何这种方法有效，建议参考论文：Feature hashing for large scale multitask learning.这里就不多说了。

　　　　在scikit-learn的HashingVectorizer类中，实现了基于signed hash trick的算法，这里我们就用HashingVectorizer来实践一下Hash Trick，为了简单，我们使用上面的19维词汇表，并哈希降维到6维。当然在实际应用中，19维的数据根本不需要Hash Trick，这里只做一个演示，代码如下：

	from sklearn.feature_extraction.text import HashingVectorizer 
	vectorizer2=HashingVectorizer(n_features = 6,norm = None)
	print vectorizer2.fit_transform(corpus)
　　　　输出如下：

	  (0, 1)	2.0
	  (0, 2)	-1.0
	  (0, 4)	1.0
	  (0, 5)	-1.0
	  (1, 0)	1.0
	  (1, 1)	1.0
	  (1, 2)	-1.0
	  (1, 5)	-1.0
	  (2, 0)	2.0
	  (2, 5)	-2.0
	  (3, 0)	0.0
	  (3, 1)	4.0
	  (3, 2)	-1.0
	  (3, 3)	1.0
	  (3, 5)	-1.0
　　　　大家可以看到结果里面有负数，这是因为我们的哈希函数$\xi$可以哈希到1或者-1导致的。

　　　　和PCA类似，Hash Trick降维后的特征我们已经不知道它代表的特征名字和意义。此时我们不能像上一节向量化时候可以知道每一列的意义，所以Hash Trick的解释性不强。

#### 4. 向量化与Hash Trick小结
　　　　这里我们对向量化与它的特例Hash Trick做一个总结。在特征预处理的时候，我们什么时候用一般意义的向量化，什么时候用Hash Trick呢？标准也很简单。

　　　　一般来说，<font color='deeppink'>只要词汇表的特征不至于太大，大到内存不够用，肯定是使用一般意义的向量化比较好。因为向量化的方法解释性很强，我们知道每一维特征对应哪一个词，进而我们还可以使用TF-IDF对各个词特征的权重修改，进一步完善特征的表示。

　　　　而Hash Trick用大规模机器学习上，此时我们的词汇量极大，使用向量化方法内存不够用，而使用Hash Trick降维速度很快，降维后的特征仍然可以帮我们完成后续的分类和聚类工作。当然由于分布式计算框架的存在，其实一般我们不会出现内存不够的情况。因此，实际工作中我使用的都是特征向量化。

　　　　向量化与Hash Trick就介绍到这里，下一篇我们讨论TF-IDF。

### （三）文本挖掘预处理之TF-IDF
　　　　在文本挖掘预处理之向量化与Hash Trick中我们讲到在文本挖掘的预处理中，向量化之后一般都伴随着TF-IDF的处理，那么什么是TF-IDF，为什么一般我们要加这一步预处理呢？这里就对TF-IDF的原理做一个总结。

#### 1. 文本向量化特征的不足
　　　　在将文本分词并向量化后，我们可以得到词汇表中每个词在各个文本中形成的词向量，比如在文本挖掘预处理之向量化与Hash Trick这篇文章中，我们将下面4个短文本做了词频统计：

	corpus=["I come to China to travel", 
	    "This is a car polupar in China",          
	    "I love tea and Apple ",   
	    "The work is to write some papers in science"] 
　　　　不考虑停用词，处理后得到的词向量如下：

	[[0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0]
	 [0 0 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0]
	 [1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0]
	 [0 0 0 0 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1]]
　　　　如果我们直接将统计词频后的19维特征做为文本分类的输入，会发现有一些问题。比如第一个文本，我们发现"come","China"和“Travel”各出现1次，而“to“出现了两次。似乎看起来这个文本与”to“这个特征更关系紧密。但是实际上”to“是一个非常普遍的词，几乎所有的文本都会用到，因此虽然它的词频为2，但是重要性却比词频为1的"China"和“Travel”要低的多。如果我们的向量化特征仅仅用词频表示就无法反应这一点。因此我们需要进一步的预处理来反应文本的这个特征，而这个预处理就是TF-IDF。

#### 2. TF-IDF概述
　　　　TF-IDF是Term Frequency -  Inverse Document Frequency的缩写，即<font color='deeppink'>“词频-逆文本频率”。它由两部分组成，TF和IDF。

　　　　前面的<font color='red'>TF也就是我们前面说到的词频</font>，我们之前做的向量化也就是做了文本中各个词的出现频率统计，并作为文本特征，这个很好理解。关键是后面的这个IDF，即“逆文本频率”如何理解。在上一节中，我们讲到几乎所有文本都会出现的"to"其词频虽然高，但是重要性却应该比词频低的"China"和“Travel”要低。我们的IDF就是来帮助我们来反应这个词的重要性的，进而修正仅仅用词频表示的词特征值。

　　　　概括来讲， <font color='red'>IDF反应了一个词在所有文本中出现的频率（也可以理解为词的信息量）</font>，如果一个词在很多的文本中出现，那么它的IDF值应该低，比如上文中的“to”。而反过来如果一个词在比较少的文本中出现，那么它的IDF值应该高。比如一些专业的名词如“Machine Learning”。这样的词IDF值应该高。一个极端的情况，如果一个词在所有的文本中都出现，那么它的IDF值应该为0。

　　　　上面是从定性上说明的IDF的作用，那么如何对一个词的IDF进行定量分析呢？这里直接给出<font color='red'>一个词$x$的IDF的基本公式如下：$$IDF(x) = log\frac{N}{N(x)}$$

　　　　其中，<font color='deeppink'>$N$代表语料库中文本的总数，而$N(x)$代表语料库中包含词$x$的文本总数。</font>为什么IDF的基本公式应该是是上面这样的而不是像$N/N(x)$这样的形式呢？这就涉及到信息论相关的一些知识了。感兴趣的朋友建议阅读吴军博士的《数学之美》第11章。

　　　　上面的IDF公式已经可以使用了，但是在一些特殊的情况会有一些小问题，比如某一个生僻词在语料库中没有，这样我们的分母为0， IDF没有意义了。所以常用的IDF我们需要做一些平滑，使语料库中没有出现的词也可以得到一个合适的IDF值。平滑的方法有很多种，<font color='red'>最常见的IDF平滑后的公式之一为：$$IDF(x) = log\frac{N+1}{N(x)+1} + 1$$

　　　　有了IDF的定义，我们就可以<font color='red'>计算某一个词的TF-IDF值了：$$TF-IDF(x) = TF(x) * IDF(x)$$

　　　　<font color='red'>这个值可以表示一个词在文档中的权重。</font>其中$TF(x)$指词$x$在<font color='red'>当前文本中的词频</font>。IDF是一个全量信息，综合全局文档得出每个词的IDF值。
&#8195 &#8195 &#8195  <font color='red'>TF-IDF的缺点是没有考虑词的组合搭配，优点是运算量小，符合直觉，解释性强。</font>在Solr  elastic-Search  和luence这些搜索引擎中广泛使用。


#### 3. 用scikit-learn进行TF-IDF预处理
　　　　在scikit-learn中，有两种方法进行TF-IDF的预处理。

　　　　完整代码参见我的github:https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/tf-idf.ipynb

　　　　第一种方法是在用CountVectorizer类向量化之后再调用TfidfTransformer类进行预处理。第二种方法是直接用TfidfVectorizer完成向量化与TF-IDF预处理。

　　　　首先我们来看第一种方法，CountVectorizer+TfidfTransformer的组合，代码如下：

复制代码
	from sklearn.feature_extraction.text import TfidfTransformer  
	from sklearn.feature_extraction.text import CountVectorizer  
	
	corpus=["I come to China to travel", 
	    "This is a car polupar in China",          
	    "I love tea and Apple ",   
	    "The work is to write some papers in science"] 
	
	vectorizer=CountVectorizer()

	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  
	print tfidf
复制代码
　　　　输出的各个文本各个词的TF-IDF值如下：

	  (0, 4)	0.442462137895
	  (0, 15)	0.697684463384
	  (0, 3)	0.348842231692
	  (0, 16)	0.442462137895
	  (1, 3)	0.357455043342
	  (1, 14)	0.453386397373
	  (1, 6)	0.357455043342
	  (1, 2)	0.453386397373
	  (1, 9)	0.453386397373
	  (1, 5)	0.357455043342
	  (2, 7)	0.5
	  (2, 12)	0.5
	  (2, 0)	0.5
	  (2, 1)	0.5
	  (3, 15)	0.281131628441
	  (3, 6)	0.281131628441
	  (3, 5)	0.281131628441
	  (3, 13)	0.356579823338
	  (3, 17)	0.356579823338
	  (3, 18)	0.356579823338
	  (3, 11)	0.356579823338
	  (3, 8)	0.356579823338
	  (3, 10)	0.356579823338
　　　　现在我们用TfidfVectorizer一步到位，代码如下：

	from sklearn.feature_extraction.text import TfidfVectorizer
	tfidf2 = TfidfVectorizer()
	re = tfidf2.fit_transform(corpus)
	print re
　　　　输出的各个文本各个词的TF-IDF值和第一种的输出完全相同。大家可以自己去验证一下。

　　　　由于第二种方法比较的简洁，因此在实际应用中推荐使用，一步到位完成向量化，TF-IDF与标准化。

#### 4. TF-IDF小结
　　　　TF-IDF是非常常用的文本挖掘预处理基本步骤，但是如果<font color='red'>预处理中使用了Hash Trick，则一般就无法使用TF-IDF了，因为Hash Trick后我们已经无法得到哈希后的各特征的IDF的值。</font>使用了IF-IDF并标准化以后，我们就可以使用各个文本的词特征向量作为文本的特征，进行分类或者聚类分析。

　　　　当然TF-IDF不光可以用于文本挖掘，在信息检索等很多领域都有使用。因此值得好好的理解这个方法的思想。

 ## word2vec原理和gensim实现
#### (一) CBOW与Skip-Gram模型基础
　　　　word2vec是google在2013年推出的一个NLP工具，它的特点是<font color='red'>将所有的词向量化，这样词与词之间就可以定量的去度量他们之间的关系，挖掘词之间的联系</font>。虽然源码是开源的，但是谷歌的代码库国内无法访问，因此本文的讲解word2vec原理以Github上的word2vec代码为准。本文关注于word2vec的基础知识。

1.词向量基础
    　　　　用词向量来表示词并不是word2vec的首创，在很久之前就出现了。最早的词向量是很冗长的，它使用是词向量维度大小为整个词汇表的大小，对于每个具体的词汇表中的词，将对应的位置置为1。比如我们有下面的5个词组成的词汇表，词"Queen"的序号为2， 那么它的词向量就是$(0,1,0,0,0)$。同样的道理，词"Woman"的词向量就是$(0,0,0,1,0)$。这种词向量的编码方式我们一般叫做1-of-N representation或者稀疏向量one hot representation（词的独热表示）.
    　　　　
    [外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-I7W6ZA5v-1631636699473)(./1624894569313.png)]
    
   One hot representation用来表示词向量非常简单，但是却有很多问题。最大的问题是我们的词汇表一般都非常大，比如达到百万级别，这样每个词都用百万维的向量来表示简直是内存的灾难。这样的向量其实除了一个位置是1，其余的位置全部都是0，表达的效率不高，能不能把词向量的维度变小呢？

　　　　密集向量Distributed representation（分布式表示）可以解决One hot representation的问题，它的思路是<font color='red'>通过训练，将每个词都映射到一个较短的词向量上来</font>。所有的这些词向量就构成了向量空间，进而可以用普通的统计学的方法来研究词与词之间的关系。这个较短的词向量维度是多大呢？这个一般需要我们在训练时自己来指定。

　　　　比如下图我们将词汇表里的词用"Royalty","Masculinity", "Femininity"和"Age"4个维度来表示，King这个词对应的词向量可能是$(0.99, 0.99,0.05, 0.7)$。当然在实际情况中，我们并不能对词向量的每个维度做一个很好的解释。
[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-LLx1voJA-1631636699475)(./1624894589712.png)]

　　　　有了用Distributed Representation表示的较短的词向量，我们就可以较容易的分析词之间的关系了，比如我们将词的维度降维到2维，有一个有趣的研究表明，用下图的词向量表示我们的词时，我们可以发现：$$\vec {King} - \vec {Man} + \vec {Woman} = \vec {Queen} $$
[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-RaV7GXOA-1631636699477)(./1624894615743.png)]
 　　　　可见我们只要得到了词汇表里所有词对应的词向量，那么我们就可以做很多有趣的事情了。不过，怎么训练得到合适的词向量呢？一个很常见的方法是使用神经网络语言模型。

2.CBOW与Skip-Gram用于神经网络语言模型
					<font color='deeppink'>语言模型是用周边的词来预测一个位置出现词的概率（完型填空）</font>
    　　　　在word2vec出现之前，已经有用神经网络DNN来用训练词向量进而处理词与词之间的关系了。采用的方法一般是一个三层的神经网络结构（当然也可以多层），分为输入层，隐藏层和输出层(softmax层)。

　　　　这个模型是如何定义数据的输入和输出呢？一般分为连续词袋模型CBOW(Continuous Bag-of-Words 与Skip-Gram两种模型。

　　　　CBOW模型的训练<font color='red'>输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量</font>。比如下面这段话，我们的上下文大小取值为4，特定的这个词是"Learning"，也就是我们需要的输出词向量,上下文对应的词有8个，前后各4个，这8个词是我们模型的输入。由于CBOW使用的是词袋模型，因此这8个词都是平等的，也就是不考虑他们和我们关注的词之间的距离大小，只要在我们上下文之内即可。
[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-XyWzYnOT-1631636699480)(./1624894635288.png)]
　　　　这样我们这个CBOW的例子里，<font color='red'>我们的输入是8个词向量，输出是所有词的softmax概率 </font>（训练的目标是期望训练样本特定词对应的softmax概率最大），对应的CBOW神经网络模型输入层有8个神经元，输出层有词汇表大小个神经元。隐藏层的神经元个数我们可以自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。这样当我们有新的需求，要求出某8个词对应的最可能的输出中心词时，我们可以通过一次DNN前向传播算法并通过softmax激活函数找到概率最大的词对应的神经元即可。
　　　　
　　　　Skip-Gram模型和CBOW的思路是反着来的，即<font color='red'>输入是特定的一个词的词向量，而输出是特定词对应的上下文词向量</font>。还是上面的例子，我们的上下文大小取值为4， 特定的这个词"Learning"是我们的输入，而这8个上下文词是我们的输出。

　　　　这样我们这个Skip-Gram的例子里，我们的<font color='red'>输入是特定词， 输出是softmax概率排前8的8个词</font>，对应的Skip-Gram神经网络模型输入层有1个神经元，输出层有词汇表大小个神经元。隐藏层的神经元个数我们可以自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。这样当我们有新的需求，要求出某1个词对应的最可能的8个上下文词时，我们可以通过一次DNN前向传播算法得到概率大小排前8的softmax概率对应的神经元所对应的词即可。

　　　　以上就是神经网络语言模型中如何用CBOW与Skip-Gram来训练模型与得到词向量的大概过程。但是这和word2vec中用CBOW与Skip-Gram来训练模型与得到词向量的过程有很多的不同。

　　　　word2vec为什么 不用现成的DNN模型，要继续优化出新方法呢？<font color='red'>最主要的问题是DNN模型的这个处理过程非常耗时。我们的词汇表一般在百万级别以上，这意味着我们DNN的输出层需要进行softmax计算各个词的输出概率的的计算量很大</font>。有没有简化一点点的方法呢？

3. word2vec基础之霍夫曼树
    　　　　word2vec也使用了CBOW与Skip-Gram来训练模型与得到词向量，但是并没有使用传统的DNN模型。最先优化使用的数据结构是<font color='red'>用霍夫曼树来代替隐藏层和输出层的神经元，霍夫曼树的叶子节点起到输出层神经元的作用，叶子节点的个数即为词汇表的小大。 而内部节点则起到隐藏层神经元的作用。</font>

　　　　具体如何用霍夫曼树来进行CBOW和Skip-Gram的训练我们在下一节讲，这里我们先复习下霍夫曼树。

　　　　霍夫曼树的建立其实并不难，过程如下：

　　　　输入：权值为$(w_1,w_2,...w_n)$的$n$个节点

　　　　输出：对应的霍夫曼树

　　　　1）将$(w_1,w_2,...w_n)$看做是有$n$棵树的森林，每个树仅有一个节点。

　　　　2）在森林中选择根节点权值最小的两棵树进行合并，得到一个新的树，这两颗树分布作为新树的左右子树。新树的根节点权重为左右子树的根节点权重之和。

　　　　3） 将之前的根节点权值最小的两棵树从森林删除，并把新树加入森林。

　　　　4）重复步骤2）和3）直到森林里只有一棵树为止。

　　　　下面我们用一个具体的例子来说明霍夫曼树建立的过程，我们有(a,b,c,d,e,f)共6个节点，节点的权值分布是(20,4,8,6,16,3)。

　　　　首先是最小的b和f合并，得到的新树根节点权重是7.此时森林里5棵树，根节点权重分别是20,8,6,16,7。此时根节点权重最小的6,7合并，得到新子树，依次类推，最终得到下面的霍夫曼树。
[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-J7zVL1f6-1631636699483)(./1624894667403.png)]
　　　　那么霍夫曼树有什么好处呢？一般得到霍夫曼树后我们会<font color='red'>对叶子节点进行霍夫曼编码，由于权重高的叶子节点越靠近根节点，而权重低的叶子节点会远离根节点，这样我们的高权重节点编码值较短，而低权重值编码值较长。这保证的树的带权路径最短，也符合我们的信息论，即我们希望越常用的词拥有更短的编码</font>。如何编码呢？一般对于一个霍夫曼树的节点（根节点除外），可以约定左子树编码为0，右子树编码为1.如上图，则可以得到c的编码是00。

　　　　在word2vec中，约定编码方式和上面的例子相反，即约定<font color='red'>左子树编码为1，右子树编码为0，同时约定左子树的权重不小于右子树的权重</font>。

　　　　我们在下一节的Hierarchical Softmax中再继续讲使用霍夫曼树和DNN语言模型相比的好处以及如何训练CBOW&Skip-Gram模型。

## （二）Hierarchical Softmax模型

#### 2.1.基于Hierarchical Softmax的模型概述

　　　　我们先回顾下传统的神经网络词向量语言模型，里面一般有三层，输入层（词向量），隐藏层和输出层（softmax层）。里面最大的问题在于<font color='red'>从隐藏层到输出的softmax层的计算量很大，因为要计算所有词的softmax概率，再去找概率最大的值。</font>这个模型如下图所示。其中$V$是词汇表的大小，

<div align=center> [外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-jMPUWH2T-1631636699484)(./1625149093024.png)]

　　　　word2vec对这个模型做了改进，首先，<font color='red'>对于从输入层到隐藏层的映射，没有采取神经网络的线性变换加激活函数的方法，而是采用简单的对所有输入词向量求和并取平均的方法</font>。比如输入的是三个4维词向量：$(1,2,3,4), (9,6,11,8),(5,10,7,12)$,那么我们word2vec映射后的词向量就是$(5,6,7,8)$。由于这里是<font color='red'>从多个词向量变成了一个词向量。</font>

　　　　第二个改进就是从隐藏层到输出的softmax层这里的计算量个改进。为了<font color='red'>避免要计算所有词的softmax概率，word2vec采样了霍夫曼树来代替从隐藏层到输出softmax层的映射</font>。我们在上一节已经介绍了霍夫曼树的原理。如何映射呢？这里就是理解word2vec的关键所在了。

　　　　由于我们**把之前所有都要计算的从输出到softmax层的概率计算变成了一颗二叉霍夫曼树，那么我们的softmax概率计算只需要沿着树形结构进行就可以了**。如下图所示，我们可以<font color='red'>沿着霍夫曼树从根节点一直走到我们的叶子节点的词$w_2$。</font>

<div align=center>[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-3w0XuEmp-1631636699485)(./1625149073565.png)]



　　　　和之前的神经网络语言模型相比，**我们的霍夫曼树的所有内部节点就类似之前神经网络隐藏层的神经元,其中，根节点的词向量对应我们的投影后的词向量，而所有叶子节点就类似于之前神经网络softmax输出层的神经元，叶子节点的个数就是词汇表的大小。在霍夫曼树中，隐藏层到输出层的softmax映射不是一下子完成的，而是沿着霍夫曼树一步步完成的，因此这种softmax取名为"Hierarchical Softmax"。**


　　　　如何“沿着霍夫曼树一步步完成”呢？在word2vec中，我们采用了<font color='red'>二元逻辑回归的方法，即规定如果是负类(霍夫曼树编码1)沿着左子树走，如果是正类(霍夫曼树编码0)沿着右子树走</font>。<font color='deeppink'>判别正类和负类的方法是使用sigmoid函数</font>，即：$$P(+) = \sigma(x_w^T\theta) = \frac{1}{1+e^{-x_w^T\theta}}$$

　　　　其中$x_w$是当前内部节点的词向量，而$\theta$则是我们需要从训练样本求出的逻辑回归的模型参数。

　　　　使用霍夫曼树有什么好处呢？首先，<font color='red'>由于是二叉树，之前计算量为$V$,现在变成了$log_2V$。第二，由于使用霍夫曼树是高频的词靠近树根，这样高频词需要更少的时间会被找到</font>，这符合我们的贪心优化思想。（相当于二分查找，对于某个叶节点样本的训练，只需要训练它经过的节点就行，而不需要遍历整棵树）

　　　　容易理解，被划分为左子树而成为负类的概率为$P(-) =  1-P(+)$。<font color='red'>在某一个内部节点，要判断是沿左子树还是右子树走的标准就是看$P(-),P(+)$谁的概率值大。</font>而<font color='deeppink'>控制$P(-),P(+)$谁的概率值大的因素一个是当前节点的词向量，另一个是当前节点的模型参数$\theta$。</font>

　　　　对于上图中的$w_2$，如果它是一个训练样本的输出，那么我们期望对于里面的隐藏节点$n(w_2,1)$的$P(-)$概率大，$n(w_2,2)$的$P(-)$概率大，$n(w_2,3)$的$P(+)$概率大。

　　　　回到基于Hierarchical Softmax的word2vec本身，我们的目标就是<font color='red'>找到合适的所有节点的词向量和所有内部节点$\theta$, 使训练样本达到最大似然。</font>那么如何达到最大似然呢？

#### 2. 2 基于Hierarchical Softmax的模型梯度计算

　　　我们<font color='red'>使用最大似然法来寻找所有节点的词向量和所有内部节点$\theta$。</font>先拿上面的$w_2$例子来看，我们期望最大化下面的似然函数：$$\prod_{i=1}^3P(n(w_i),i) = (1- \frac{1}{1+e^{-x_w^T\theta_1}})(1- \frac{1}{1+e^{-x_w^T\theta_2}})\frac{1}{1+e^{-x_w^T\theta_3}}$$

　　　　<font color='red'>对于所有的训练样本，我们期望最大化所有样本的似然函数乘积。</font>

　　　　为了便于我们后面一般化的描述，我们定义输入的词为$w$,其从输入层词向量求和平均后的霍夫曼树根节点词向量为$x_w$, 从根节点到$w$所在的叶子节点，包含的节点总数为$l_w$, $w$在霍夫曼树中从根节点开始，经过的第$i$个节点表示为$p_i^w$,对应的霍夫曼编码为$d_i^w \in \{0,1\}$,其中$i =2,3,...l_w$。而该节点对应的模型参数表示为$\theta_i^w$, 其中$i =1,2,...l_w-1$，<font color='red'>没有$i =l_w$是因为模型参数仅仅针对于霍夫曼树的内部节点</font>。

　　　　<font color='red'>定义$w$经过的霍夫曼树某一个节点j的逻辑回归概率为$P(d_j^w|x_w, \theta_{j-1}^w)$，其表达式为：

$$P(d_j^w|x_w, \theta_{j-1}^w)=
\begin{cases}
 \sigma(x_w^T\theta_{j-1}^w)& {d_j^w=0}\\
1-  \sigma(x_w^T\theta_{j-1}^w) & {d_j^w = 1}
\end{cases}$$

　　　　那么对于某一个目标输出词$w$,其最大似然为：$$\prod_{j=2}^{l_w}P(d_j^w|x_w, \theta_{j-1}^w) = \prod_{j=2}^{l_w} [\sigma(x_w^T\theta_{j-1}^w)] ^{1-d_j^w}[1-\sigma(x_w^T\theta_{j-1}^w)]^{d_j^w}$$

　　　　在word2vec中，由于使用的是<font color='red'>随机梯度上升法，所以并没有把所有样本的似然乘起来得到真正的训练集最大似然，仅仅每次只用一个样本更新梯度，这样做的目的是减少梯度计算量。</font>这样我们可以得到$w$的对数似然函数$L$如下：

$$L= log \prod_{j=2}^{l_w}P(d_j^w|x_w, \theta_{j-1}^w) = \sum\limits_{j=2}^{l_w} ((1-d_j^w) log [\sigma(x_w^T\theta_{j-1}^w)]  + d_j^w log[1-\sigma(x_w^T\theta_{j-1}^w)])$$

　　　　要得到模型中$w$词向量和内部节点的模型参数$\theta$, 我们使用梯度上升法即可。首先我们求模型参数$\theta_{j-1}^w$的梯度：

$$ \begin{align} \frac{\partial L}{\partial \theta_{j-1}^w} & = (1-d_j^w)\frac{(\sigma(x_w^T\theta_{j-1}^w)(1-\sigma(x_w^T\theta_{j-1}^w)}{\sigma(x_w^T\theta_{j-1}^w)}x_w - d_j^w \frac{(\sigma(x_w^T\theta_{j-1}^w)(1-\sigma(x_w^T\theta_{j-1}^w)}{1- \sigma(x_w^T\theta_{j-1}^w)}x_w  \\ & =  (1-d_j^w)(1-\sigma(x_w^T\theta_{j-1}^w))x_w -  d_j^w\sigma(x_w^T\theta_{j-1}^w)x_w \\& = (1-d_j^w-\sigma(x_w^T\theta_{j-1}^w))x_w \end{align}$$

　　　　如果大家看过之前写的逻辑回归原理小结，会发现这里的梯度推导过程基本类似。

　　　　同样的方法，可以求出$x_w$的梯度表达式如下：$$\frac{\partial L}{\partial x_w} = \sum\limits_{j=2}^{l_w}(1-d_j^w-\sigma(x_w^T\theta_{j-1}^w))\theta_{j-1}^w $$

　　　　有了梯度表达式，我们就可以用梯度上升法进行迭代来一步步的求解我们需要的所有的$\theta_{j-1}^w$和$x_w$。

#### 2.3  基于Hierarchical Softmax的CBOW模型

　　　　由于word2vec有两种模型：CBOW和Skip-Gram,我们先看看基于CBOW模型时， Hierarchical Softmax如何使用。

　　　　**首先我们要定义词向量的维度大小$M$，以及CBOW的上下文大小$2c$,这样我们对于训练样本中的每一个词，其前面的$c$个词和后面的$c$个词作为了CBOW模型的输入,该词本身作为样本的输出，期望softmax概率最大。**

　　　　在做CBOW模型前，我们需要先将词汇表建立成一颗霍夫曼树。

　　　　对于从输入层到隐藏层（投影层），这一步比较简单，**就是对$w$周围的$2c$个词向量求和取平均即可**，即：$$x_w = \frac{1}{2c}\sum\limits_{i=1}^{2c}x_i$$

　　　　第二步，**通过梯度上升法来更新我们的$\theta_{j-1}^w$和$x_w$**，注意这里的$x_w$是由$2c$个词向量相加而成，我们做梯度更新完毕后会用梯度项直接**更新原始的各个$x_i(i=1,2,,,,2c)$，**即：

$$\theta_{j-1}^w = \theta_{j-1}^w + \eta  (1-d_j^w-\sigma(x_w^T\theta_{j-1}^w))x_w $$$$x_i= x_i +\eta  \sum\limits_{j=2}^{l_w}(1-d_j^w-\sigma(x_w^T\theta_{j-1}^w))\theta_{j-1}^w \;(i =1,2..,2c) $$

　　　　其中$\eta$为梯度上升法的步长。

　　　　这里总结下<font color='red'>基于Hierarchical Softmax的CBOW模型算法流程</font>，梯度迭代使用了随机梯度上升法：

　　　　输入：基于CBOW的语料训练样本，词向量的维度大小$M$，CBOW的上下文大小$2c$,步长$\eta$。<font color='red'>词向量大小M是什么？？？降维的稠密向量维度？</font>

　　　　输出：霍夫曼树的内部节点模型参数$\theta$，所有的词向量$w$

1. 基于语料训练样本建立霍夫曼树。
2. 随机初始化所有的模型参数$\theta$，所有的词向量$w$
3. 梯度上升迭代，对于训练集中的每一个样本$(context(w), w)$做如下处理：

　　　　　　a)  e=0， 计算$x_w= \frac{1}{2c}\sum\limits_{i=1}^{2c}x_i$

　　　　　　b)  for j = 2 to $ l_w$, 计算：$$f = \sigma(x_w^T\theta_{j-1}^w)$$$$g = (1-d_j^w-f)\eta$$$$e = e + g\theta_{j-1}^w$$$$\theta_{j-1}^w= \theta_{j-1}^w + gx_w$$

&#8194 &#160 &#8194   &#8194 &#8194 &#8194 &#8194 &#160  c) 对于$context(w)$中的每一个词向量$x_i$(共2c个)进行更新：$$x_i = x_i + e$$　

　　　　　　d) 如果梯度收敛，则结束梯度迭代，否则回到步骤3继续迭代。

#### 2.4. 基于Hierarchical Softmax的Skip-Gram模型

　　　　现在我们先看看基于Skip-Gram模型时， Hierarchical Softmax如何使用。此时输入的只有一个词$w$,输出的为$2c$个词向量$context(w)$。

　　　　**我们对于训练样本中的每一个词，该词本身作为样本的输入， 其前面的$c$个词和后面的$c$个词作为了Skip-Gram模型的输出,，期望这些词的softmax概率比其他的词大。**

　　　　Skip-Gram模型和CBOW模型其实是反过来的，在上一篇已经讲过。

　　　　在做CBOW模型前，我们需要先将词汇表建立成一颗霍夫曼树。

　　　　对于从输入层到隐藏层（投影层），这一步比CBOW简单，由于只有一个词，所以，即$x_w$就是词$w$对应的词向量。

　　　　第二步，通过梯度上升法来更新我们的$\theta_{j-1}^w$和$x_w$，注意这里的$x_w$周围有$2c$个词向量，此时如果我们期望$P(x_i|x_w), i=1,2...2c$最大。此时我们注意到由于上下文是相互的，在期望$P(x_i|x_w), i=1,2...2c$最大化的同时，反过来我们也期望$P(x_w|x_i), i=1,2...2c$最大。那么是<font color='red'>使用$P(x_i|x_w)$好还是$P(x_w|x_i)$好呢，word2vec使用了后者，这样做的好处就是在一个迭代窗口内，我们不是只更新$x_w$一个词，而是$x_i, i=1,2...2c$共$2c$个词。这样整体的迭代会更加的均衡。</font>因为这个原因，Skip-Gram模型并没有和<font color='deeppink'>CBOW模型一样对输入进行迭代更新，而是对$2c$个输出进行迭代更新。</font>

　　　　这里总结下基于Hierarchical Softmax的Skip-Gram模型算法流程，梯度迭代使用了随机梯度上升法：

　　　　输入：基于Skip-Gram的语料训练样本，词向量的维度大小$M$，Skip-Gram的上下文大小$2c$,步长$\eta$

　　　　输出：霍夫曼树的内部节点模型参数$\theta$，所有的词向量$w$

   　 1. 基于语料训练样本建立霍夫曼树。
        &#8194 &#160  2. 随机初始化所有的模型参数$\theta$，所有的词向量$w$,
         &#8194 &#160 3. 进行梯度上升迭代过程，对于训练集中的每一个样本$(w, context(w))$做如下处理：

　　　　　    a)  for i =1 to 2c:

　　　　　　　　i) e=0

　　　　　　　　ii)for j = 2 to $ l_w$, 计算：$$f = \sigma(x_i^T\theta_{j-1}^w)$$$$g = (1-d_j^w-f)\eta$$$$e = e + g\theta_{j-1}^w$$$$\theta_{j-1}^w= \theta_{j-1}^w+ gx_i$$

　　　　　　　　iii) $$x_i = x_i + e$$

　　　　　　b)如果梯度收敛，则结束梯度迭代，算法结束，否则回到步骤a继续迭代。

#### 2.5. Hierarchical Softmax的模型源码和算法的对应　
　　 &#160 1.整体语料的各个词频  决定 huffman树（网络结构）
			&#8194 &#160 &#8194 &#160   2.整体语料，会根据窗口，拆解成很多训练样本
			&#8194 &#160 &#8194 &#160 3.这些训练样本所用的huffman树是一棵
			&#8194 &#160 &#8194 &#160 4.每个训练样本所对应的目标词不一样，因此，不同训练样本在同一颗huffman树上走不同的路径

　　　　这里给出上面算法和word2vec源码中的变量对应关系。

　　　　在源代码中，基于Hierarchical Softmax的CBOW模型算法在435-463行，基于Hierarchical Softmax的Skip-Gram的模型算法在495-519行。大家可以对着源代码再深入研究下算法。

　　　　在源代码中，neule对应我们上面的$e$, syn0对应我们的$x_w$, syn1对应我们的$\theta_{j-1}^i$, layer1_size对应词向量的维度，window对应我们的$c$。

　　　　另外，vocab[word].code[d]指的是，当前单词word的第d个编码，编码不含Root结点。vocab[word].point[d]指的是：当前单词word第d个编码下，前置的结点。
##(三) 基于Negative Sampling的模型
在上一篇中我们讲到了基于Hierarchical Softmax的word2vec模型，本文我们我们再来看看另一种求解word2vec模型的方法：Negative Sampling。

####  3.1  Hierarchical Softmax的缺点与改进

　　　　在讲基于Negative Sampling的word2vec模型前，我们先看看Hierarchical Softmax的的缺点。的确，使用霍夫曼树来代替传统的神经网络，可以提高模型训练的效率。<font color='red'>但是如果我们的训练样本里的中心词$w$是一个很生僻的词，那么就得在霍夫曼树中辛苦的向下走很久了。</font>能不能不用搞这么复杂的一颗霍夫曼树，将模型变的更加简单呢？

　　　　Negative Sampling就是这么一种求解word2vec模型的方法，它摒弃了霍夫曼树，采用了Negative Sampling（负采样）的方法来求解，下面我们就来看看Negative Sampling的求解思路。

#### 3.2 基于Negative Sampling的模型概述
		word2vec用神经网络解法时，输出是计算V类的概率，其中1类是中心词，概率往大的方向走，剩下一类是V-1个其它词，概率往小的方向走。真正计算复杂的就是负类别。负采样法就是从V-1个负样本中随机挑几个词做负样本。
　　　　既然名字叫Negative Sampling（负采样），那么肯定使用了采样的方法。采样的方法有很多种，比如之前讲到的大名鼎鼎的MCMC。我们这里的Negative Sampling采样方法并没有MCMC那么复杂。

　　　　比如我们有一个训练样本，<font color='red'>中心词是$w$,它周围上下文共有$2c$个词，记为$context(w)$。</font>**由于这个中心词$w$的确和$context(w)$相关存在，因此它是一个真实的正例。通过Negative Sampling采样，我们得到neg个和$w$不同的中心词$w_i, i=1,2,..neg$，这样$context(w)$和$w_i$就组成了neg个并不真实存在的负例。利用这一个正例和neg个负例，我们进行二元逻辑回归，得到负采样对应每个词$w_i$对应的模型参数$\theta_{i}$，和每个词的词向量。**

　　　　从上面的描述可以看出，<font color='red'>Negative Sampling由于没有采用霍夫曼树，每次只是通过采样neg个不同的中心词做负例，就可以训练模型</font>，因此整个过程要比Hierarchical Softmax简单。

　　　　不过有两个问题还需要弄明白：
　　　　1）如何通过一个正例和neg个负例进行二元逻辑回归呢？
　　　　2）如何进行负采样呢？

　　　　我们在第三节讨论问题1，在第四节讨论问题2.

#### 3. 3 基于Negative Sampling的模型梯度计算

　　　　Negative Sampling也是采用了二元逻辑回归来求解模型参数，通过负采样，我们得到了neg个负例$(context(w), w_i) i=1,2,..neg$。为了统一描述，我们将正例定义为$w_0$。

　　　　在逻辑回归中，我们的正例应该期望满足：$$P(context(w_0), w_i) = \sigma(x_{w_0}^T\theta^{w_i}) ,y_i=1, i=0$$

　　　　我们的负例期望满足：$$P(context(w_0), w_i) =1-  \sigma(x_{w_0}^T\theta^{w_i}), y_i = 0, i=1,2,..neg$$

　　　　我们期望可以最大化下式：$$ \prod_{i=0}^{neg}P(context(w_0), w_i) = \sigma(x_{w_0}^T\theta^{w_0})\prod_{i=1}^{neg}(1-  \sigma(x_{w_0}^T\theta^{w_i}))$$

　　　　利用逻辑回归和上一节的知识，我们容易写出此时模型的似然函数为：$$\prod_{i=0}^{neg} \sigma(x_{w_0}^T\theta^{w_i})^{y_i}(1-  \sigma(x_{w_0}^T\theta^{w_i}))^{1-y_i}$$

　　　　此时对应的对数似然函数为：$$L = \sum\limits_{i=0}^{neg}y_i log(\sigma(x_{w_0}^T\theta^{w_i})) + (1-y_i) log(1-  \sigma(x_{w_0}^T\theta^{w_i}))$$

　　　　和Hierarchical Softmax类似，我们采用随机梯度上升法，仅仅每次只用一个样本更新梯度，来进行迭代更新得到我们需要的$x_{w_i}, \theta^{w_i},  i=0,1,..neg$, 这里我们需要求出$x_{w_0}, \theta^{w_i},  i=0,1,..neg$的梯度。

　　　　首先我们计算$\theta^{w_i}$的梯度：$$\begin{align} \frac{\partial L}{\partial \theta^{w_i} } &= y_i(1-  \sigma(x_{w_0}^T\theta^{w_i}))x_{w_0}-(1-y_i)\sigma(x_{w_0}^T\theta^{w_i})x_{w_0} \\ & = (y_i -\sigma(x_{w_0}^T\theta^{w_i})) x_{w_0} \end{align}$$

　　　　同样的方法，我们可以求出$x_{w_0}$的梯度如下：$$\frac{\partial L}{\partial x^{w_0} } = \sum\limits_{i=0}^{neg}(y_i -\sigma(x_{w_0}^T\theta^{w_i}))\theta^{w_i} $$

　　　　有了梯度表达式，我们就可以用梯度上升法进行迭代来一步步的求解我们需要的$x_{w_0}, \theta^{w_i},  i=0,1,..neg$。

#### 3.4 Negative Sampling负采样方法

　　　　现在我们来看看如何进行负采样，得到neg个负例。word2vec采样的方法并不复杂，如果<font color='red'>词汇表的大小为$V$,那么我们就将一段长度为1的线段分成$V$份，每份对应词汇表中的一个词。</font>当然每个词对应的线段长度是不一样的，高频词对应的线段长，低频词对应的线段短(高频词数量多，分子count就大)。<font color='deeppink'>每个词$w$的线段长度</font>由下式决定：$$len(w) = \frac{count(w)}{\sum\limits_{u \in vocab} count(u)}$$

　　　　在word2vec中，分子和分母都取了3/4次幂如下：$$len(w) = \frac{count(w)^{3/4}}{\sum\limits_{u \in vocab} count(u)^{3/4}}$$

　　　　在采样前，我们将这段长度为1的线段划分成$M$等份，这里$M >> V$，这样可以保证每个词对应的线段都会划分成对应的小块。而M份中的每一份都会落在某一个词对应的线段上。在采样的时候，我们只需要从$M$个位置中采样出$neg$个位置就行，此时采样到的每一个位置对应到的线段所属的词就是我们的负例词。

[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-Ve43Mcna-1631636699487)(./1624893880556.png)]
[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-Ur2y5bfg-1631636699488)(./1624893880595.png)]


　　　　**在word2vec中，$M$取值默认为$10^8$。**

##### 3.5  基于Negative Sampling的CBOW模型

　　　　有了上面Negative Sampling负采样的方法和逻辑回归求解模型参数的方法，我们就可以总结出基于Negative Sampling的CBOW模型算法流程了。梯度迭代过程使用了随机梯度上升法：

　　　　输入：基于CBOW的语料训练样本，词向量的维度大小$Mcount$，CBOW的上下文大小$2c$,步长$\eta$, 负采样的个数neg

　　　　输出：词汇表每个词对应的模型参数$\theta$，所有的词向量$x_w$
	    &#8194 &#160 &#8194 &#160  &#160  &#160 1. 随机初始化所有的模型参数$\theta$，所有的词向量$w$
	     &#8194 &#160 &#8194 &#160  &#160  &#160  2. 对于每个训练样本$(context(w_0), w_0)$,负采样出neg个负例中心词$w_i, i=1,2,...neg$
	     &#8194 &#160 &#8194 &#160  &#160  &#160  3. 进行梯度上升迭代过程，对于训练集中的每一个样本$(context(w_0), w_0,w_1,...w_{neg})$做如下处理：

　　　　　　a)  e=0， 计算$x_{w_0}= \frac{1}{2c}\sum\limits_{i=1}^{2c}x_i$

　　　　　　b)  for i= 0 to neg, 计算：$$f = \sigma(x_{w_0}^T\theta^{w_i})$$$$g = (y_i-f)\eta$$$$e = e + g\theta^{w_i}$$$$\theta^{w_i}= \theta^{w_i} + gx_{w_0}$$

&#8194 &#160 &#8194   &#8194 &#8194 &#8194 &#8194 &#160  c) 对于$context(w)$中的每一个词向量$x_k$(共2c个)进行更新：$$x_k = x_k + e$$　

　　　　　　d) 如果梯度收敛，则结束梯度迭代，否则回到步骤3继续迭代。

#### 3.6 基于Negative Sampling的Skip-Gram模型

　　　　有了上一节CBOW的基础和上一篇基于Hierarchical Softmax的Skip-Gram模型基础，我们也可以总结出基于Negative Sampling的Skip-Gram模型算法流程了。梯度迭代过程使用了随机梯度上升法：

　　　　输入：基于Skip-Gram的语料训练样本，词向量的维度大小$Mcount$，Skip-Gram的上下文大小$2c$,步长$\eta$， , 负采样的个数neg。

　　　　输出：词汇表每个词对应的模型参数$\theta$，所有的词向量$x_w$
   　　　 &#8194 1. 随机初始化所有的模型参数$\theta$，所有的词向量$w$
                &#8194 &#8194 &#160 &#8194 &#160 &#160  2. 对于每个训练样本$(context(w_0), w_0)$,负采样出neg个负例中心词$w_i, i=1,2,...neg$
                &#8194 &#8194 &#160 &#8194 &#160  &#160  3. 进行梯度上升迭代过程，对于训练集中的每一个样本$(context(w_0), w_0,w_1,...w_{neg})$做如下处理：

　　　　　　a)  for i =1 to 2c:

　　　　　　　　i)  e=0

　　　　　　　　ii)  for j= 0 to neg, 计算：$$f = \sigma(x_{w_{0i}}^T\theta^{w_j})$$$$g = (y_j-f)\eta$$$$e = e + g\theta^{w_j}$$$$\theta^{w_j}= \theta^{w_j} + gx_{w_{0i}}$$

　　　　　　　　iii)  词向量更新：$$x_{w_{0i}} = x_{w_{0i}} + e $$

　　　　　　b)如果梯度收敛，则结束梯度迭代，算法结束，否则回到步骤a继续迭代。

#### 3.7  Negative Sampling的模型源码和算法的对应　　

　　　　这里给出上面算法和word2vec源码中的变量对应关系。

　　　　在源代码中，基于Negative Sampling的CBOW模型算法在464-494行，基于Negative Sampling的Skip-Gram的模型算法在520-542行。大家可以对着源代码再深入研究下算法。

　　　　在源代码中，neule对应我们上面的$e$, syn0对应我们的$x_w$, syn1neg对应我们的$\theta^{w_i}$, layer1_size对应词向量的维度，window对应我们的$c$。negative对应我们的neg, table_size对应我们负采样中的划分数$M$。

　　　　另外，vocab[word].code[d]指的是：当前单词word的第d个编码，编码不含Root结点。vocab[word].point[d]指的是：当前单词word第d个编码下，前置的结点。这些和基于Hierarchical Softmax的是一样的。

　　　　以上就是基于Negative Sampling的word2vec模型，希望可以帮到大家，后面会讲解用gensim的python版word2vec来使用word2vec解决实际问题。
##(四) 用gensim学习word2vec
Word2vec的实现工具有很多：
1. 谷歌原生 效率高
2. Gensim
3. Fasttext  facebook搞出来的，可以直接用python实现，但是比较耗内存

这几个都不用分布式。spark支持分布式，但是bug特别多，尽量不用。分布式是多台机器并行计算。机器学习中涉及到很多算法需要频繁交互，此时分布式效率未必高，因为涉及到很大的通信成本。分布式适合于数据间是解耦的。

　　　　在word2vec原理篇中，我们对word2vec的两种模型CBOW和Skip-Gram，以及两种解法Hierarchical Softmax和Negative Sampling做了总结。这里我们就从实践的角度，使用gensim来学习word2vec。（word to vector，词转向量）

#### 1. gensim安装与概述

　　　　gensim是一个很好用的Python NLP的包，不光可以用于使用word2vec，还有很多其他的API可以用。它封装了google的C语言版的word2vec。当然我们可以可以直接使用C语言版的word2vec来学习，但是个人认为没有gensim的python版来的方便。

　　　　安装gensim是很容易的，使用"pip install gensim"即可。但是需要注意的是gensim对numpy的版本有要求，所以安装过程中可能会偷偷的升级你的numpy版本。而windows版的numpy直接装或者升级是有问题的。此时我们需要卸载numpy，并重新下载带mkl的符合gensim版本要求的numpy，下载地址在此：http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy。安装方法和scikit-learn 和pandas 基于windows单机机器学习环境的搭建这一篇第4步的方法一样。

　　　　安装成功的标志是你可以在代码里做下面的import而不出错：

from gensim.models import word2vec

#### 2. gensim word2vec API概述

　　　　在gensim中，word2vec 相关的API都在包gensim.models.word2vec中。和算法有关的参数都在类gensim.models.word2vec.Word2Vec中。<font color='red'>算法需要注意的参数有：

　　　　1) sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。后面我们会有从文件读出的例子。

　　　　2) size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。

　　　　3) window窗口：即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为$c$，window越大，上下文来预测中心词的词越多，默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。word2vec会在分好词的文本中不停的移动窗口来预测中心词，语料样本会非常大。

　　　　4) sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。

　　　　5) hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。

　　　　6) negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。

　　　　7) cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的$x_w$为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示$x_w$,默认值也是1,不推荐修改默认值。

　　　　8) min_count:最小计数阈值。若单词出现次数低于该阈值，则这个单词会被忽略，默认是5。如果是小语料，可以调低这个值。

　　　　9) iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。

　　　　10) alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为$\eta$，默认是0.025。

　　　　11) min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
　　　　 12）workers : 整型, 可选
            训练模型所采用的工作线程数量（=3使用多核机器进行训练将更快）

　　　　以上就是gensim word2vec的主要的参数，下面我们用一个实际的例子来学习word2vec。

#### 3. gensim  word2vec实战

　　　　我选择的《人民的名义》的小说原文作为语料，语料原文在这里。

　　　　完整代码参见我的github: https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/word2vec.ipynb

　　　　拿到了原文，我们首先要进行分词，这里使用结巴分词完成。在中文文本挖掘预处理流程总结中，我们已经对分词的原理和实践做了总结。因此，这里直接给出分词的代码，分词的结果，我们放到另一个文件中。代码如下, 加入下面的一串人名是为了结巴分词能更准确的把人名分出来。

复制代码
	# -*- coding: utf-8 -*-
	

	import jieba
	import jieba.analyse
	jieba.suggest_freq('沙瑞金', True)
	jieba.suggest_freq('田国富', True)
	jieba.suggest_freq('高育良', True)
	jieba.suggest_freq('侯亮平', True)
	jieba.suggest_freq('钟小艾', True)
	jieba.suggest_freq('陈岩石', True)
	jieba.suggest_freq('欧阳菁', True)
	jieba.suggest_freq('易学习', True)
	jieba.suggest_freq('王大路', True)
	jieba.suggest_freq('蔡成功', True)
	jieba.suggest_freq('孙连城', True)
	jieba.suggest_freq('季昌明', True)
	jieba.suggest_freq('丁义珍', True)
	jieba.suggest_freq('郑西坡', True)
	jieba.suggest_freq('赵东来', True)
	jieba.suggest_freq('高小琴', True)
	jieba.suggest_freq('赵瑞龙', True)
	jieba.suggest_freq('林华华', True)
	jieba.suggest_freq('陆亦可', True)
	jieba.suggest_freq('刘新建', True)
	jieba.suggest_freq('刘庆祝', True)
	
	with open('./in_the_name_of_people.txt') as f:
	    document = f.read()    
	#document_decode = document.decode('GBK')    
	document_cut = jieba.cut(document)
	#print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
	result = ' '.join(document_cut)
	result = result.encode('utf-8')
	with open('./in_the_name_of_people_segment.txt', 'w') as f2:
	    f2.write(result)

f.close()
f2.close()
复制代码
　　　　拿到了分词后的文件，<font color='red'>在一般的NLP处理中，会需要去停用词。由于word2vec的算法依赖于上下文，而上下文有可能就是停词。因此对于word2vec，我们可以不用去停词。

　　　　现在我们可以直接读分词后的文件到内存。这里使用了word2vec提供的LineSentence类来读文件，然后套用word2vec的模型。这里只是一个示例，因此省去了调参的步骤，实际使用的时候，你可能需要对我们上面提到一些参数进行调参。

复制代码
	# import modules & set up logging
	import logging
	import os
	from gensim.models import word2vec
	

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	sentences = word2vec.LineSentence('./in_the_name_of_people_segment.txt') 
	model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)  

复制代码
　　　　模型出来了，我们可以用来做什么呢？这里给出三个常用的应用。
　　　　<font color='red'>第一个是最常用的，找出某一个词向量最相近的词集合</font>,代码如下：
req_count = 5
for key in model.wv.similar_by_word('沙瑞金'.decode('utf-8'), topn =100):
    if len(key[0])==3:
        req_count -= 1
        print key[0], key[1]
        if req_count == 0:
            break
　　　　我们看看沙书记最相近的一些3个字的词（主要是人名）如下：

	高育良 0.967257142067
	李达康 0.959131598473
	田国富 0.953414440155
	易学习 0.943500876427
	祁同伟 0.942932963371

　　　　<font color='red'>第二个应用是看两个词向量的相近程度</font>，这里给出了书中两组人的相似程度：

	print model.wv.similarity('沙瑞金'.decode('utf-8'), '高育良'.decode('utf-8'))
	print model.wv.similarity('李达康'.decode('utf-8'), '王大路'.decode('utf-8'))

　　　　输出如下：

	0.961137455325
	0.935589365706

　　　　第三个应用是<font color='red'>找出不同类的词，这里给出了人物分类</font>题：

	print model.wv.doesnt_match(u"沙瑞金 高育良 李达康 刘庆祝".split())

　　　   word2vec也完成的很好，输出为"刘庆祝"。　　　    
#### 4.两种模型总结
一种是根据中心词上下文的各C个词来预测中心词，叫CBOW连续词袋模型
一种是根据中心词预测周围词的概率，叫skip-gram。
用哪种方法看需求：
1.使用时需要将多个向量相加（文本向量化） 用cbow
2.使用时都是单个词向量使用（找近义词） 用skip-gram
大原则：使用的过程和训练的过程越一致 ，效果一般越好
		如果实在不知道怎么选，一般来说skip-gram+ns负采样效果好一点点。同样是W1-W5这五个样本，如果window=2，对于CBOW只有一个训练样本，而skip-gram则有四个训练样本。即skip-gram训练样本更多，效果相对会更好。
		
#### 5.使用技巧
每次训练出来的词向量都在单独的语义空间，不同语义空间的向量没有可比性。比如第一次训练的一批词向量马云和第二次训练一批词向量的马化腾互相比较没有任何意义。word2vec词向量只能全量训练，因为语料库变了one-hot也变了，V也变了。
第一天 一堆文本 （没有W10 有 W1）->  词向量
第二天  一堆文本   (有W10  没有W1   )->  词向量
要比较W10和W1只能合在一起训练。
如果有一堆词，明明不相关，训练出来确是显示相似的，可能是因为孤岛效应。即某部分词总是一起出现，另一堆词也是一起出现，但是这两堆词互相没有任何交集，虽然在一起训练是一个向量空间，但实际上是两个向量空间。这两堆词互相比较是没有意义的。
孤岛效应本质是由一些不相关语料或者弱相关语料组成。Word2vec本身不能解决这个问题，这个只能在样本选取上下功夫，让训练样本尽可能相关。所以各领域自己训练自己的，不要把一堆不相关的东西放到一起训练。几个行业几套词向量。

以上就是用gensim学习word2vec实战的所有内容，希望对大家有所帮助。
