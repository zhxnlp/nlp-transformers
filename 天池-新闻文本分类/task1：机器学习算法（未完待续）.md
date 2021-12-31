@[toc]
本文重点使用TF-IDF+LightGBM

# 二、LightGBM介绍
LightGBM：一种高效的gbdt梯度提升决策树
gbdt 是决策树的集成模型，它基于三个重要原则：

- 弱学习器（决策树） 
- 梯度优化 
- 提升技巧
所以在gbdt方法中我们有很多决策树（弱学习器）。这些树是按顺序构建的：

- 第一棵树学习如何适应目标变量
- 第二棵树学习如何适应第一棵树的预测和基本事实之间的残差（差异）
- 第三棵树学习如何拟合第二棵树的残差，依此类推。
- 
所有这些树都是通过在整个系统中传播误差梯度来训练的。gbdt 的主要缺点是：在每个树节点中找到最佳分割点非常耗时且消耗很多内存，其他 boosting 方法试图解决这个问题。

LightGBM原理和参数详见：算法参照：[《LightGBM：一种高效的梯度提升决策树》](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)、[《了解 LightGBM 参数（以及如何调整它们）》](https://neptune.ai/blog/lightgbm-parameters-guide#slideDown)
摘要：
&#8195;&#8195;梯度提升决策树 (GBDT) 是一种流行的机器学习算法，并且有很多有效的实现，例如 XGBoost 和 pGBRT。虽然在这些实现中采用了很多工程优化，但在特征维数高、数据量大的情况下，效率和可扩展性仍然不尽如人意。一个主要原因是对于每个特征，他们需要扫描所有的数据实例来估计所有可能的分割点的信息增益，这是非常耗时的。
&#8195;&#8195;为了解决这个问题，我们提出了两种新技术：\emph{基于梯度的单边采样}（GOSS）和\emph{Exclusive Feature Bundling}（EFB）。使用 GOSS，我们排除了很大一部分具有小梯度的数据实例，仅使用其余部分来估计信息增益。我们证明，由于梯度较大的数据实例在信息增益的计算中起更重要的作用，因此GOSS可以用更小的数据量获得相当准确的信息增益估计。使用 EFB，我们捆绑了互斥的特征（即，它们很少同时采用非零值），以减少特征的数量。我们证明了寻找唯一特征的最优捆绑是 NP-hard 的，但是贪心算法可以达到相当好的逼近比（从而可以有效地减少特征的数量，而不会对分割点确定的准确性造成太大影响）。我们使用 GOSS 和 EFB \emph{LightGBM} 调用我们的新 GBDT 实现。我们在多个公共数据集上的实验表明。。。
1. dart gradient boosting：
[论文：DART: Dropouts meet Multiple Additive Regression Trees](https://arxiv.org/abs/1505.01866)中，介绍了DART 梯度提升，这是一种使用神经网络中的标准 dropout 来改进模型正则化并处理其他一些不太明显的问题的方法。 

也就是说，gbdt 存在过度专业化over-specialization的问题，这意味着在以后的迭代中添加的树往往只影响少数实例的预测，而对其余实例的贡献可以忽略不计。==添加 dropout 使以后迭代中的树更难以专注于那些少数样本，从而提高了性能。==

2. lgbm goss（基于梯度的单边采样）
	 - 之所以叫lightgbm，就是使用了基于论文[《LightGBM: A Highly Efficient Gradient Boosting
Decision Tree》](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)的[Goss方法](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)。Goss 是更新、更轻的 gbdt 实现（因此是“轻量级”gbm）。
	-  gbdt 是可靠的，但在大型数据集上速度不够快。因此，goss 提出了一种基于梯度的采样方法，以避免搜索整个搜索空间。我们知道，对于每个数据实例，当梯度很小时，这意味着不用担心数据训练得很好，而当梯度很大时，应该再次重新训练。所以我们这里有两个方面，具有大梯度和小梯度的数据实例。因此，==goss 保留具有大梯度的所有数据并对具有小梯度的数据进行随机采样（这就是为什么它被称为单侧采样）。这使得搜索空间更小，goss 可以更快地收敛==。最后，为了更深入地了解 goss，您可以查看这篇[博文](https://towardsdatascience.com/what-makes-lightgbm-lightning-fast-a27cf0d9785e)。

下表第二、第三行是Lgbm dart和Lgbm goss
![在这里插入图片描述](https://img-blog.csdnimg.cn/fcbde0f04ac04799acaac5afd334457a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
num_leaves：每个弱学习器拥有的最大叶子数。大 num_leaves 增加了训练集的准确性，也增加了因过度拟合而受到伤害的机会。根据文档，一种简单的方法是num_leaves = 2^(max_depth)
subsample：subsample（或 bagging_fraction），您可以指定每次树构建迭代使用的行的百分比。这意味着将随机选择一些行来拟合每个学习器（树）。这提高了泛化能力，但也提高了训练速度。
其它参照文档

# 三、代码解析
下面分类代码的本质，都是讲文本text经过TF/TF-IDF处理之后，转换成为文章向量（文章向量维度表示词向量的长度），之后将文章向量经过各种分类器，进行分类。本质上和fasttext、textcnn没有区别。（向量和label进行训练拟合，进行分类）

max_features表示每篇文章的词向量维度，即一篇文章设置为多少维向量来表示
ngram_range=(1,3)表示采样多元语言模型，从1元到3元。即包括单个词、双词到三个词。
其它参数参考：[文档《sklearn.feature_extraction.text.TfidfVectorizer》](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)、[《Python中的TfidfVectorizer参数解析》](https://blog.csdn.net/laobai1015/article/details/80451371?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163499954516780255218174%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163499954516780255218174&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-80451371.pc_search_all_es&utm_term=TfidfVectorizer&spm=1018.2226.3001.4187)
### 3.1 TF + RidgeClassifier
==max_features可以设置每篇文章的词向量维度，todense方法可以将稀疏矩阵转换为密集矩阵。==

```python
vectorizer = CountVectorizer() #构建一个计算词频（TF）的玩意儿，当然这里面不足是可以做这些

transformer = TfidfTransformer() #构建一个计算TF-IDF的玩意儿

tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

#vectorizer.fit_transform(corpus)将文本corpus输入，得到词频矩阵

#将这个矩阵作为输入，用transformer.fit_transform(词频矩阵)得到TF-IDF权重矩阵

TfidfTransformer + CountVectorizer  =  TfidfVectorizer
```
fit(raw_documents[,y])：从训练集学习词汇表和idf
transform(raw_documents)：将documents 转换为 文档词矩TF-IDF
fit_transform(raw_documents[, y])：学习词汇表和idf，返回文档词矩TF-IDF
```python
#从google云盘上加载数据
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/transformers/天池-入门NLP - 新闻文本分类')
```
 #### 3.1.2 max_features举例
```python
# Count Vectors词频
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#读取训练集前15000行数据测试
train_df = pd.read_csv('./train_set.csv', sep='\t', nrows=15000)
# 将一篇文章映射为3000维度的向量
vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])
print(train_test[:1000])
```
在大规模语料上训练TFIDF会得到非常多的词语，如果再使用了ngram_range，那么我们词表的大小就会爆炸。出于时间和空间效率的考虑，可以限制最多使用多少个词语，模型会优先选取词频高的词语留下。
```python
(0, 852)	6
(0, 2540)	4
(0, 1061)	1
(0, 446)	2
:	:
(999, 1387)	2
(999, 255)	1
(999, 2096)	2
(999, 759)	2
```
如果设置vectorizer = CountVectorizer(max_features=1000)，则结果如下：

```python
(0, 278)	6
(0, 143)	2
(0, 116)	1
:	:
(999, 835)	2
(999, 416)	2
(999, 103)	1
(999, 620)	1
```
#### 3.1.3 todense方法
用todense方法可以将稀疏矩阵转换为密集矩阵：

```python
print(train_test)
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 0]]
```
其实就是1-1000篇文章的列向量（二维列表，第一个维度是文章，第二个维度是文章的词）转换1000*3000的矩阵。

或者看这个例子：

```python
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer=CountVectorizer()
corpus=["I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"] 
print(vectorizer.fit_transform(corpus).todense())

[[0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0]
 [0 0 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0]
 [1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1]]
```

```python
print(vectorizer.fit_transform(corpus))

(0, 4)	1
(0, 15)	2
(0, 3)	1
(0, 16)	1
(1, 3)	1
(1, 14)	1
:	:
```
使用岭回归RidgeClassifier分类器训练模型
```python
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.7423139613717681
```
没有使用todense方法，整个训练时间是1分钟，准确率0.742。

```python
clf = RidgeClassifier()
clf.fit(train_test[:10000].todense(), train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:].todense())
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
#0.7411137174103894
```
==使用todense方法训练时间12秒，大大缩短，但是准确率0.741，稍有下降。==
#### 3.1.4 F1_score(sklearn)
```python
f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

#参数含义
average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']

'binary':默认值，表示二分类。
'micro' :通过先计算总体的TP，FN和FP的数量，再计算F1
'macro' :分别计算每个类别的F1，然后做平均（各类别F1的权重相同）当小类很重要时会出问题，因为该macro-averging方法是对性能的平均。另一方面，该方法假设所有分类都是一样重要的，因此macro-averaging方法会对小类的性能影响很大。
samples：应用在 multilabel问题上。它不会计算每个类，相反，它会在评估数据中，通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）
average：average=None将返回一个数组，它包含了每个类的得分.
weighted: 对于不均衡数量的类来说，计算二分类metrics的平均，通过在每个类的score上进行加权实现。
```
参考：[sklearn中 F1-micro 与 F1-macro区别和计算原理](https://blog.csdn.net/weixin_30802273/article/details/98816157?ops_request_misc=&request_id=&biz_id=&utm_medium=distribute.pc_search_result.none-task-blog-2~all~es_rank~default-9-98816157.pc_search_all_es&utm_term=sklearn%20f1_score%E5%8F%82%E6%95%B0&spm=1018.2226.3001.4187)、[F1_score(sklearn)](https://blog.csdn.net/Miraclecanbeachieve/article/details/89138066?ops_request_misc=&request_id=&biz_id=&utm_medium=distribute.pc_search_result.none-task-blog-2~all~es_rank~default-3-89138066.pc_search_all_es&utm_term=sklearn%20f1_score%E5%8F%82%E6%95%B0&spm=1018.2226.3001.4187)

### 3.2 TF-IDF +  RidgeClassifier

```python
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('../data/train_set.csv', sep='\t', nrows=15000)

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
print(train_test[:1000])
```
可以看到跟3.1TF相比，词频转换成了TF-IDF值
```python
(0, 2865)	0.021535770236833403
(0, 2806)	0.02001603028943581
:	:
(999, 1722)	0.044467254252004755
(999, 600)	0.0856488025571623
```

```python
print(train_test[:1000].todense())

[[0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.         ... 0.         0.         0.00481137]
 [0.         0.         0.         ... 0.         0.         0.        ]]
```
TfidfVectorizer参数：
ngram_range: tuple(min_n, max_n)
要提取的n-gram的n-values的下限和上限范围，在min_n <= n <= max_n区间的n的全部值
```python
clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
#0.8721598830546126
```
- 正常用时71秒，f1=0.87216。
- 使用todense方法训练时间70秒，f1=0.87194。

TF-IDF的计算时间比TF长很多。
关于RidgeClassifier岭回归分类器，参考[《Skleran-线性模型-Ridge/岭回归》](https://blog.csdn.net/qq_34356768/article/details/106818309)

### 3.3 TF-IDF+朴素贝叶斯

```python
# TF-IDF +  高斯朴素贝叶斯
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=30000)
# 将一篇文章映射为3000维度的向量
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
clf = GaussianNB()
clf.fit(train_test[:20000].toarray(), train_df['label'].values[:20000])

val_pred = clf.predict(train_test[20000:].toarray())
print(f1_score(train_df['label'].values[20000:], val_pred, average='macro'))
# 0.7286027535218247,用时126秒
```
### 3.4 TF-IDF+决策树

```python
# TF-IDF +  决策树
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=30000)
# 将一篇文章映射为3000维度的向量
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(train_test[:20000].todense(), train_df['label'].values[:20000])

val_pred = clf.predict(train_test[20000:].todense())
print(f1_score(train_df['label'].values[20000:], val_pred, average='macro'))

# 0.7019031297922049，用时193秒
```
### 3.5 TF-IDF+随机森林

```python
# TF-IDF +  随机森林

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=30000)
# 将一篇文章映射为3000维度的向量
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
clf = RandomForestClassifier(
        n_estimators=10, criterion='gini',
        max_depth=None,min_samples_split=2, 
        min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features='auto', max_leaf_nodes=None,
        min_impurity_split=1e-07,bootstrap=True,
        oob_score=False, n_jobs=1, 
        random_state=None, verbose=0,
        warm_start=False, class_weight=None)
clf.fit(train_test[:20000].todense(), train_df['label'].values[:20000])

val_pred = clf.predict(train_test[20000:].todense())
print(f1_score(train_df['label'].values[20000:], val_pred, average='macro'))
#0.7456517053768087，用时127秒
```
### 3.6 TF-IDF+XGBoost
GBDT也叫梯度提升决策树。[学习资料:XGBoost入门系列第一讲](https://zhuanlan.zhihu.com/p/27816315)

```python
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score
train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=30000)
# 将一篇文章映射为3000维度的向量
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
clf = XGBClassifier(min_child_weight=6,max_depth=15,
                            objective='multi:softmax',num_class=5)
clf.fit(train_test[:20000].todense(), train_df['label'].values[:20000])

val_pred = clf.predict(train_test[20000:].todense())
print(f1_score(train_df['label'].values[20000:], val_pred, average='macro'))
#0.9024422794080935,111分钟
```
### 3.7 TF-IDF+LightGBM
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.metrics import f1_score
train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=30000)
# 将一篇文章映射为3000维度的向量
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
params_sklearn = {
    'learning_rate':0.1,
    'max_bin':150,
    'num_leaves':32,    
    'max_depth':11,
    'reg_alpha':0.1,
    'reg_lambda':0.2,   
    'objective':'multiclass',
    'n_estimators':300,
    #'class_weight':weight
}

clf = lgb.LGBMClassifier(**params_sklearn)
clf.fit(train_test[:20000].todense(), train_df['label'].values[:20000])

val_pred = clf.predict(train_test[20000:].todense())
print(f1_score(train_df['label'].values[20000:], val_pred, average='macro'))
#0.9027526863541893，用时26分12秒
```
序号    | 算法|  f1-score|时间
-------- | -----| ----- | -----
1  |RidgeClassifier（max_features=3000，下同）|0.8895|146秒
2  | 高斯朴素贝叶斯|0.7286|126秒
3  | 决策树|0.7019|193秒
4 |随机森林|0.7456|127秒
5  | GBDT|0.9024|111分钟
6  | lgb|0.9027|26分钟
7  |RidgeClassifier（max_features=4000,stop_words=['3750','648','900']）|0.8994|156秒

### 3.8 全量数据TF-IDF+LightGBM
参考[《零基础入门NLP - 新闻文本分类比赛》](https://zhuanlan.zhihu.com/p/416009420)、[《【学习笔记】新闻文本分类(一)——TF-IDF》](https://blog.csdn.net/Protocols7/article/details/115262349?spm=1001.2014.3001.5501)
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import numpy as np
from scipy import sparse

"""
df_train = pd.read_csv('./train_set.csv',sep='\t')
df_test = pd.read_csv('./test_a.csv',sep='\t')

df_train['text_len'] = df_train['text'].apply(lambda x:len(x.split(' ')))
"""
print(df_train['text_len'].describe())

# 词频5000
tfidfVector=TfidfVectorizer(ngram_range=(1,3),max_features=5000)
tfidfVector.fit(pd.concat([df_train['text'],df_test['text']],axis=0))
#
X_train=tfidfVector.transform(df_train['text'])
X_test=tfidfVector.transform(df_test['text'])
#
sparse.save_npz('X_train_tfidf.npz',X_train)
sparse.save_npz('X_test_tfidf.npz',X_test)#大小207m，如果是50000维，大小364m
#训练时间32min
```

```python
params_sklearn = {
    'learning_rate':0.1,
    'max_bin':150,
    'num_leaves':32,    
    'max_depth':11,
    'reg_alpha':0.1,
    'reg_lambda':0.2,   
    'objective':'multiclass',
    'n_estimators':100,
    #'class_weight':weight
}

clf=LGBMClassifier(**params_sklearn)
"""
clf=LGBMClassifier(n_jobs=-1,min_child_samples=21,max_depth=-1,subsample=0.7217, 
colsample_bytree=0.6,reg_alpha=0.001,reg_lambda=0.5,num_leaves=67,
learning_rate=0.088,n_estimators=100)"""

#早停参数，精度在一定步数内没有变化，就可以停止训练了
clf.fit(X_train.todense(),df_train['label'].values)

df=pd.DataFrame()
df['label']=clf.predict(X_test)
df.to_csv('/TFIDF_lgb.csv')
```
最终运行时间32分钟，分数0.9358（参数懒得认真调了，也没用交叉验证啥的）
# 一、文本挖掘原理
## 1 分词的基本原理
&#8195;&#8195;现代分词都是基于统计的分词，而统计的样本内容来自于一些标准的语料库。假如有一个句子：“小明来到荔湾区”，我们期望语料库统计后分词的结果是："小明/来到/荔湾/区"，而不是“小明/来到/荔/湾区”。那么如何做到这一点呢？
&#8195;&#8195;从统计的角度，我们期望"小明/来到/荔湾/区"这个分词后句子出现的概率要比“小明/来到/荔/湾区”大。如果用数学的语言来说说，如果有一个句子$S$,它有m种分词选项如下：$$A_{11}A_{12}...A_{1n_1}$$$$A_{21}A_{22}...A_{2n_2}$$$$......  ......$$$$A_{m1}A_{m2}...A_{mn_m}$$

&#8195;&#8195;其中下标$n_i$代表第$i$种分词的词个数。如果我们从中选择了最优的第$r$种分词方法，那么这种分词方法对应的统计分布概率应该最大，即：$$r = \underbrace{arg\;max}_iP(A_{i1},A_{i2},...,A_{in_i}) $$

&#8195;&#8195;但是我们的概率分布$P(A_{i1},A_{i2},...,A_{in_i})$并不好求出来，因为它涉及到$n_i$个分词的联合分布。在NLP中，为了简化计算，我们通常使用<font color='red'>马尔科夫假设，即每一个分词出现的概率仅仅和前一个分词有关</font>，即：$$P(A_{ij}|A_{i1},A_{i2},...,A_{i(j-1)}) = P(A_{ij}|A_{i(j-1)})$$
&#8195;&#8195;则我们的联合分布就好求了，即：$$P(A_{i1},A_{i2},...,A_{in_i}) = P(A_{i1})P(A_{i2}|A_{i1})P(A_{i3}|A_{i2})...P(A_{in_i}|A_{i(n_i-1)})$$

&#8195;&#8195;而通过朴素贝叶斯算法，通过数据统计，<font color='red'>用频率值作为概率值的估计，近似的计算出所有的分词之间的二元条件概率。</font>比如任意两个词$w_1,w_2$，它们的条件概率分布可以近似的表示为：$$P(w_2|w_1) = \frac{P(w_1,w_2)}{P(w_1)} \approx \frac{freq(w_1,w_2)}{freq(w_1)}$$$$P(w_1|w_2) = \frac{P(w_2,w_1)}{P(w_2)} \approx \frac{freq(w_1,w_2)}{freq(w_2)}$$

&#8195;&#8195;<font color='red'>其中$freq(w_1,w_2)$表示$w_1,w_2$在语料库中相邻一起出现的次数</font>，而其中$freq(w_1),freq(w_2)$分别表示$w_1,w_2$在语料库中出现的统计次数。

&#8195;&#8195;利用语料库建立的统计概率，对于一个新的句子，我们就可以通过计算各种分词方法对应的联合分布概率，找到最大概率对应的分词方法，即为最优分词。
### 1.2 N元模型
&#8195;&#8195;<font color='red'>n-gram模型：假设一个词的概率由前面N-1个词决定

&#8195;&#8195;例如只依赖于前一个词的模型为二元模型(Bi-Gram model)，即：$$P(A_{i1},A_{i2},...,A_{in_i}) = P(A_{i1})P(A_{i2}|A_{i1})P(A_{i3}|A_{i1}，A_{i2})...P(A_{in_i}|A_{i(n_i-2)}，A_{i(n_i-1)})$$

&#8195;&#8195;而依赖于前两个词的模型为三元模型。以此类推，一直到通用的$N$元模型。

&#8195;&#8195;在实际应用中，$N$一般都较小，一般都小于4，主要原因是<font color='red'>N元模型概率分布的空间复杂度为$O(|V|^N)$，其中$|V|$为语料库大小</font>，而$N$为模型的元数，当$N$增大时，复杂度呈指数级的增长。(二元模型前后两个词都有V种选择)常用汉字三四千，但是常用词是20w，二元模型就是400亿可能。

&#8195;&#8195;$N$元模型的分词方法虽然很好，但是要在实际中应用也有很多问题:
- 某些生僻词，或者相邻分词联合分布在语料库中没有，概率为0。这种情况我们一般会使用　<font color='red'>拉普拉斯平滑，即给它一个较小的概率值。</font>
- 第二个问题是如果句子长，分词有很多情况，计算量也非常大，这时我们可以用下一节维特比算法来优化算法时间复杂度。
### 1.3 维特比算法与分词
&#8195;&#8195;为了简化原理描述，我们本节的讨论都是以二元模型为基础。

&#8195;&#8195;对于一个有很多分词可能的长句子，我们当然可以用暴力方法去计算出所有的分词可能的概率，再找出最优分词方法。但是用维特比算法可以大大简化求出最优分词的时间。具体介绍参考[《文本挖掘》](https://maxiang.io/note/#)。
### 1. 4常用分词工具
&#8195;&#8195;简单的英文分词不需要任何工具，通过空格和标点符号就可以分词了，而进一步的英文分词推荐使用nltk。对于中文分词，则推荐用结巴分词（jieba）。

&#8195;&#8195;分词是文本挖掘的预处理的重要的一步，分词完成后，我们可以继续做一些其他的特征工程，比如向量化（vectorize），TF-IDF以及Hash trick。
## 2.文本挖掘预处理之向量化与Hash Trick
&#8195;&#8195;分词后，如果我们是做文本分类聚类，则后面关键的特征预处理步骤有向量化或向量化的特例Hash Trick。
### 2.1 词袋模型
&#8195;&#8195;词袋模型(Bag of Words,简称BoW)。<font color='red'>词袋模型假设我们不考虑文本中词与词之间的上下文关系，仅仅只考虑所有词的权重。而权重与词在文本中出现的频率有关。

&#8195;&#8195;词袋模型首先会进行分词，在分词之后，通过统计每个词在文本中出现的次数，我们就可以得到该文本基于词的特征，如果<font color='red'>将各个文本样本的这些词与对应的词频放在一起，就是我们常说的向量化。</font>向量化完毕后一般也会使用<font color='red'>TF-IDF进行特征的权重修正，再将特征进行标准化</font>。 再进行一些其他的特征工程后，就可以将数据带入机器学习算法进行分类聚类了。

&#8195;&#8195;<font color='deeppink'>总结下词袋模型的三部曲：分词（tokenizing），TF-IDF修订词特征值（counting）与标准化（normalizing）。</font>
　　　　
&#8195;&#8195;当然，词袋模型有很大的局限性，因为它仅仅考虑了词频，没有考虑上下文的关系，因此会丢失一部分文本的语义。但是大多数时候，<font color='deeppink'>如果我们的目的是分类聚类，则词袋模型表现的很好。

### 2.2 词袋模型之向量化
&#8195;&#8195;在词袋模型的统计词频这一步，我们会得到该文本中所有词的词频，==有了词频，我们就可以用词向量表示这个文本==。例如直接用scikit-learn的CountVectorizer类来完成文本的词频统计与向量化，代码如下：

&#8195;&#8195;完整代码参见我的github:https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/hash_trick.ipynb

```python
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer=CountVectorizer()
corpus=["I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"] 
print vectorizer.fit_transform(corpus)
```

　　　　我们看看对于上面4个文本的处理输出如下：

```python
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
```
&#8195;&#8195;可以看出4个文本的词频已经统计出，在输出中，==左边的括号中的第一个数字是文本的序号，第2个数字是词的序号，注意词的序号是基于所有的文档的。第三个数字就是我们的词频。==

&#8195;&#8195;我们可以进一步看看每个文本的词向量特征和各个特征代表的词，代码如下：

```python
print vectorizer.fit_transform(corpus).toarray()
print vectorizer.get_feature_names()
```

　　　　输出如下：		
```python
[[0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0]
	 [0 0 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0]
	 [1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0]
	 [0 0 0 0 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1]]
	[u'and', u'apple', u'car', u'china', u'come', u'in', u'is', u'love', u'papers', u'polupar', u'science', u'some', u'tea', u'the', u'this', u'to', u'travel', u'work', u'write']
```

&#8195;&#8195;可以看到我们一共有19个词，所以4个文本都是19维的特征向量。而每一维的向量依次对应了下面的19个词。另外由于词"I"在英文中是停用词，不参加词频的统计。

&#8195;&#8195;由于大部分的文本都只会使用词汇表中的很少一部分的词，因此我们的词向量中会有大量的0。也就是说<font color='deeppink'>词向量是稀疏的。在实际应用中一般使用稀疏矩阵来存储。

&#8195;&#8195;将文本做了词频统计后，我们一般会通过TF-IDF进行词特征值修订，这部分我们后面再讲。

&#8195;&#8195;向量化的方法很好用，也很直接，但是在有些场景下很难使用，比如分词后的词汇表非常大，达到100万+，此时如果我们直接使用向量化的方法，将对应的样本对应特征矩阵载入内存，有可能将内存撑爆，在这种情况下我们怎么办呢？第一反应是我们要进行特征的降维，说的没错！而<font color='deeppink'>Hash Trick就是非常常用的文本特征降维方法。
### 2.3 Hash Trick
&#8195;&#8195;在大规模的文本处理中，由于特征的维度对应分词词汇表的大小，所以维度可能非常恐怖，此时需要进行降维，不能直接用我们上一节的向量化方法。而最常用的文本降维方法是Hash Trick。
&#8195;&#8195;在Hash Trick里，我们会定义一个特征Hash后对应的哈希表的大小，这个哈希表的维度会远远小于我们的词汇表的特征维度，因此可以看成是降维。具体的方法是，对应任意一个特征名，我们会用Hash函数找到对应哈希表的位置，然后将该特征名对应的词频统计值累加到该哈希表位置。如果用数学语言表示,假如<font color='deeppink'>哈希函数$h$使第$i$个特征哈希到位置$j$,即$h(i)=j$,则第$i$个原始特征的词频数值$\phi(i)$将累加到哈希后的第$j$个特征的词频数值$\bar{\phi}$上，即：$$\bar{\phi}(j) = \sum_{i\in \mathcal{J}; h(i) = j}\phi(i)$$

&#8195;&#8195;其中$\mathcal{J}$是原始特征的维度。
&#8195;&#8195;但是上面的方法有一个问题，有可能两个原始特征的哈希后位置在一起导致词频累加特征值突然变大，为了解决这个问题，出现了hash Trick的变种signed hash trick,此时除了哈希函数$h$,我们多了一个一个哈希函数：$$\xi : \mathbb{N} \to {\pm 1}$$

&#8195;&#8195;此时我们有$$\bar{\phi}(j) = \sum_{i\in \mathcal{J}; h(i) = j}\xi(i)\phi(i)$$

&#8195;&#8195;这样做的好处是，哈希后的特征仍然是一个无偏的估计，不会导致某些哈希位置的值过大。
&#8195;&#8195;在scikit-learn的HashingVectorizer类中，实现了基于signed hash trick的算法，这里我们就用HashingVectorizer来实践一下Hash Trick，为了简单，我们使用上面的19维词汇表，并哈希降维到6维。当然在实际应用中，19维的数据根本不需要Hash Trick，这里只做一个演示，代码如下：

```python
from sklearn.feature_extraction.text import HashingVectorizer 
vectorizer2=HashingVectorizer(n_features = 6,norm = None)
print vectorizer2.fit_transform(corpus)
　　　　#输出如下：

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
```

&#8195;&#8195;大家可以看到结果里面有负数，这是因为我们的哈希函数$\xi$可以哈希到1或者-1导致的。

&#8195;&#8195;和PCA类似，Hash Trick降维后的特征我们已经不知道它代表的特征名字和意义。此时我们不能像上一节向量化时候可以知道每一列的意义，所以Hash Trick的解释性不强。
　　　　
### 2.4 向量化与Hash Trick小结
&#8195;&#8195;一般来说，<font color='deeppink'>只要词汇表的特征不至于太大，大到内存不够用，肯定是使用一般意义的向量化比较好。因为向量化的方法解释性很强，我们知道每一维特征对应哪一个词，进而我们还可以使用TF-IDF对各个词特征的权重修改，进一步完善特征的表示。

&#8195;&#8195;而Hash Trick用大规模机器学习上，此时我们的词汇量极大，使用向量化方法内存不够用，而使用Hash Trick降维速度很快，降维后的特征仍然可以帮我们完成后续的分类和聚类工作。当然由于分布式计算框架的存在，其实一般我们不会出现内存不够的情况。因此，实际工作中我使用的都是特征向量化。
## 3. 文本挖掘预处理之TF-IDF
### 3.1 TF-IDF概述
&#8195;&#8195;TF-IDF是Term Frequency - Inverse Document Frequency的缩写，即“词频-逆文本频率”。它由两部分组成，TF和IDF。

&#8195;&#8195;IDF反应了一个词在所有文本中出现的频率（也可以理解为词的信息量），如果一个词在很多的文本中出现，那么它的IDF值应该低，比如上文中的“to”。而反过来如果一个词在比较少的文本中出现，那么它的IDF值应该高。比如一些专业的名词如“Machine Learning”。这样的词IDF值应该高。一个极端的情况，如果一个词在所有的文本中都出现，那么它的IDF值应该为0。

&#8195;&#8195;一个词$x$的IDF的基本公式如下：$$IDF(x) = log\frac{N}{N(x)}$$

&#8195;&#8195;其中，<font color='deeppink'>$N$代表语料库中文本的总数，而$N(x)$代表语料库中包含词$x$的文本总数。</font>为什么IDF的基本公式应该是是上面这样的而不是像$N/N(x)$这样的形式呢？这就涉及到信息论相关的一些知识了。感兴趣的朋友建议阅读吴军博士的《数学之美》第11章。

&#8195;&#8195;某一个生僻词在语料库中没有，这样我们的分母为0， IDF没有意义了。所以常用的IDF我们需要做一些平滑，使语料库中没有出现的词也可以得到一个合适的IDF值。平滑的方法有很多种，<font color='red'>最常见的IDF平滑后的公式之一为：$$IDF(x) = log\frac{N+1}{N(x)+1} + 1$$

&#8195;&#8195;有了IDF的定义，我们就可以<font color='red'>计算某一个词的TF-IDF值了：$$TF-IDF(x) = TF(x) * IDF(x)$$

&#8195;&#8195;<font color='red'>这个值可以表示一个词在文档中的权重。</font>其中$TF(x)$指词$x$在 当前文本中的词频。IDF是一个全量信息，综合全局文档得出每个词的IDF值。
&#8195;&#8195;<font color='red'>TF-IDF的缺点是没有考虑词的组合搭配，优点是运算量小，符合直觉，解释性强。</font>在Solr  elastic-Search  和luence这些搜索引擎中广泛使用。

### 3.2 用scikit-learn进行TF-IDF预处理
在scikit-learn中，有两种方法进行TF-IDF的预处理，完整代码参见[github](https://github.com/ljpzzz/machinelearning/blob/master/natural-language-processing/tf-idf.ipynb)。

1. 用CountVectorizer类向量化之后再调用TfidfTransformer类进行预处理。
2. 直接用TfidfVectorizer完成向量化与TF-IDF预处理。

　　　　首先我们来看第一种方法，CountVectorizer+TfidfTransformer的组合，代码如下：

```python
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
:	:
(3, 18)	0.356579823338
(3, 11)	0.356579823338
(3, 8)	0.356579823338
(3, 10)	0.356579823338
```
在输出中，==左边的括号中的第一个数字是文本的序号，第2个数字是词的序号，注意词的序号是基于所有的文档的。第三个数字就是TF-IDF值。==
现在我们用TfidfVectorizer一步到位，代码如下：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer()
re = tfidf2.fit_transform(corpus)
print re
```
### 3.4 TF-IDF小结
&#8195;&#8195;TF-IDF是非常常用的文本挖掘预处理基本步骤，但是如果<font color='red'>预处理中使用了Hash Trick，则一般就无法使用TF-IDF了，因为Hash Trick后我们已经无法得到哈希后的各特征的IDF的值。</font>使用了IF-IDF并标准化以后，我们就可以使用各个文本的词特征向量作为文本的特征，进行分类或者聚类分析。

&#8195;&#8195;当然TF-IDF不光可以用于文本挖掘，在信息检索等很多领域都有使用。因此值得好好的理解这个方法的思想。
&#8195;&#8195;n-gram语言模型是从统计学角度实现，比较简单，效果有限。
